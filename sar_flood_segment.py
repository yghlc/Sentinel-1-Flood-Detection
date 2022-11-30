#!/usr/bin/env python
# Filename: sar_flood_segment.py 
"""
introduction: Object-based image analysis,

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 24 November, 2022
"""

import os, sys
from optparse import OptionParser
import time
import raster_tools
from datetime import datetime
import numpy as np

from raster_tools import image_read_pre_process,permant_water_pixles
from utility import mk_outdirectory
import utility
from utility import  get_sar_file_list
import raster_statistic
import vector_gpd
from grey_image_segment import segment_a_grey_image
from skimage import measure
from sklearn import cluster

# for projection in lat, lon, this should be very small
tile_min_overlap = 1e-10
from multiprocessing import Pool
from scipy.spatial.distance import cdist

# define some threshold ()
thr_perma_water_area = 10000 # pixel                         # required minimum area of permanent water surface in a SAR scene.
thr_min_size_of_permaWater = 1000   # pixel  (i.e 0.01 km^2)  #
thr_dis_flood_to_permaWater = 2000 # 2000 pixel (i.e 20 km)  # a flooded region should be within a ditance of river or lake?



def sar_sigma0_to_8bit(img_path,sar_img_data, save_8bit_path, min_percent=0.01, max_percent=0.99, hist_bin_count=10000, src_nodata=None,dst_nodata=0):

    if os.path.isfile(save_8bit_path):
        return True

    img_array_8bit = raster_tools.image_numpy_allBands_to_8bit_hist(sar_img_data, per_min=min_percent,
                                                                 per_max=max_percent, bin_count=hist_bin_count,
                                                                 src_nodata=src_nodata,
                                                                 dst_nodata=dst_nodata)

    # save to disk
    return raster_tools.save_numpy_array_to_rasterfile(img_array_8bit,save_8bit_path,img_path,
                                             nodata=dst_nodata,compress='lzw',tiled='yes',bigtiff='if_safer')

def run_segmentation(org_raster, save_8bit_path,segment_shp_path, save_dir, process_num = 1, b_vector=False ):
    # get initial polygons
    # because the label from segmentation for superpixels are not unique, so we may need to get mean dem diff based on polygons, set org_raster=None
    label_path = segment_a_grey_image(save_8bit_path, save_dir, process_num, org_raster=None)

    if b_vector is False:
        return label_path
    if os.path.isfile(segment_shp_path) and vector_gpd.is_field_name_in_shp(segment_shp_path, 'S0_mean'):
        print('%s exists, skip' % segment_shp_path)
        return label_path
    else:
        # remove segment_shp_path if it exist, but don't have demD_mean
        if os.path.isfile(segment_shp_path):
            utility.delete_shape_file(segment_shp_path)

        # remove nodata (it was copy from the input image)
        raster_tools.unset_nodata(label_path)

        # convert the label to shapefile # (remove -8 to use 4 connectedness.), if use 8-connect, may result in many invalid polygons
        layer_name = os.path.splitext(os.path.basename(segment_shp_path))[0]
        command_string = 'gdal_polygonize.py -8 %s -b 1 -f GPKG %s %s' % (label_path, segment_shp_path,layer_name)  # "ESRI Shapefile"
        utility.os_system_exit_code(command_string)

        # get dem elevation information for each polygon
        raster_statistic.zonal_stats_multiRasters(segment_shp_path, org_raster, tile_min_overlap=tile_min_overlap,
                                                  stats=['mean', 'std', 'count'], prefix='S0',  # SAR Sigma0
                                                  process_num=process_num)

    return label_path

def cal_one_region_attribute(org_img_data,reg):
    reg_array = org_img_data[reg.coords[:, 0], reg.coords[:, 1]]  # row, col
    return np.nanmean(reg_array),np.nanstd(reg_array)

def calculate_region_to_nearest_water_dis(regions, water_regions):
    # centroid array  # Centroid coordinate tuple (row, col)
    reg_centroid_arr = np.array([ item.centroid for item in regions])
    # print(reg_centroid_arr.shape)
    water_centroid_arr = np.array([ item.centroid for item in water_regions])
    # print(water_centroid_arr.shape)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    all_dist = cdist(reg_centroid_arr,water_centroid_arr)
    # print(all_dist.shape)
    min_dist = np.min(all_dist,axis=1)
    # print(min_dist.shape)

    return min_dist


def get_permanent_water_regions(permanent_water_path,nan_loc, verbose=True):
    # get water regions
    water_raster_np, water_nodata = raster_tools.read_raster_one_band_np(permanent_water_path)
    if water_nodata is not None:
        water_raster_np[water_raster_np == water_nodata] = 0
    water_raster_np[nan_loc] = 0
    labels = measure.label(water_raster_np, background=0, connectivity=2)       # 1 is water, 0 is not
    regions = measure.regionprops(labels)
    # only keep large ones
    regions = [item for item in regions if item.area > thr_min_size_of_permaWater]
    if verbose:
        print('the count of permanent water regions:', len(regions))

    return regions

def get_permanent_water_attributes(sar_img_data, sar_grd_file, sar_nan_loc, water_mask_file, save_dir,verbose=True):
    t2 = time.time()
    p_water_loc, p_water_count, p_water_min, p_water_max, p_water_mean, p_water_median, p_water_std, grd_p_water_file = \
        permant_water_pixles(sar_img_data, sar_grd_file, water_mask_file, save_dir)

    print(datetime.now(), 'preparation of permanent water surface cost %f seconds' % (time.time() - t2))

    water_regions = get_permanent_water_regions(grd_p_water_file, sar_nan_loc, verbose=verbose)

    name_no_ext = os.path.splitext(os.path.basename(sar_grd_file))[0]
    water_region_path = os.path.join(save_dir,name_no_ext + '_PerWaterRegions.tif')
    # for test: save water regions
    water_region_np = np.zeros_like(sar_img_data)
    for idx, reg in enumerate(water_regions):
        water_region_np[reg.coords[:, 0], reg.coords[:, 1]] = idx + 1
        if verbose:
            print(idx+1, reg.area)
    water_region_np[sar_nan_loc] = -9999
    water_region_np = water_region_np.astype(np.int32)
    raster_tools.save_numpy_array_to_rasterfile(water_region_np,water_region_path,sar_grd_file,nodata=-9999,compress='lzw',
                                                tiled='yes', bigtiff='if_safer',verbose=verbose)

    return  p_water_loc, p_water_count, p_water_mean,  p_water_std, water_regions, water_region_path

def get_object_attributes(org_img_data,nan_loc, label_path, dem_path = None, water_regions=None, verbose=True, process_num=1):
    raster_tools.unset_nodata(label_path)
    label_raster_np, label_nodata =  raster_tools.read_raster_one_band_np(label_path)
    # print(label_raster_np, label_nodata)

    labels = measure.label(label_raster_np,background=label_nodata,connectivity=2)  # 2-connectivity, 8 neighbours

    # comparing with those polygons and their attributes after gdal_polygonize.py, the polygon attributes are calculated
    # in or "touched by" the polygon
    # get regions
    regions = measure.regionprops(labels)
    if verbose:
        print('the count of regions:', len(regions))

    # print(regions)
    # calculate attribute: mean, std of SAR sigma0
    # print(org_img_data.shape)
    if dem_path is not None:
        dem_raster_np, dem_nodata = raster_tools.read_raster_one_band_np(dem_path)
        dem_raster_np = dem_raster_np.astype(np.float32)
        if dem_nodata is not None:
            dem_raster_np[dem_raster_np==dem_nodata] = np.nan
        dem_raster_np[nan_loc] = np.nan
        dem_raster_np = raster_tools.map_to_interval(dem_raster_np,0,1) # normalized to 0, 1
    else:
        dem_raster_np = None


    # total_pixel_count = 0
    reg_means = []
    # reg_areas = []
    reg_stds = []
    reg_dem_means = []


    # regions from skimage.measure.regionprops doesn't support parallel calculation
    # if process_num == 1:
    for idx, reg in enumerate(regions):
        reg_array = org_img_data[reg.coords[:,0],reg.coords[:,1]]   # row, col
        reg_means.append(np.nanmean(reg_array))
        reg_stds.append(np.nanstd(reg_array))

            # reg_areas.append(reg.area)       # pixel count
    # elif process_num > 1:
    #     theadPool = Pool(process_num)
    #     parameters_list = [(org_img_data, reg) for idx, reg in enumerate(regions)]
    #     results = theadPool.starmap(cal_one_region_attribute, parameters_list)
    #     for res in results:
    #         reg_mean, reg_std = res
    #         reg_means.append(reg_mean)
    #         reg_stds.extend(reg_std)
    #     theadPool.close()
    # else:
    #     raise ValueError('Wrong value of process_num: %s'%str(process_num))

    # calculate to the distance to water surface
    reg_to_water_diss = calculate_region_to_nearest_water_dis(regions, water_regions)  # unit: pixel
    # reg_to_water_diss = raster_tools.map_to_interval(reg_to_water_diss,0,1)

    # get attributes for permanent water regions
    water_reg_means = []
    water_reg_dem_means = []
    water_reg_to_water_diss = []
    for idx, reg in enumerate(water_regions):
        reg_array = org_img_data[reg.coords[:,0],reg.coords[:,1]]   # row, col
        water_reg_means.append(np.nanmean(reg_array))
        # water_reg_means.append(np.nanstd(reg_array))

    # calculate DEM values
    if dem_raster_np is not None:
        for idx, reg in enumerate(regions):
            reg_dem_array = dem_raster_np[reg.coords[:, 0], reg.coords[:, 1]]  # row, col
            reg_dem_means.append(np.nanmean(reg_dem_array))

        for idx, reg in enumerate(water_regions):
            reg_dem_array = dem_raster_np[reg.coords[:,0],reg.coords[:,1]]   # row, col
            water_reg_dem_means.append(np.nanmean(reg_dem_array))

        # # calculate the distance, 0? not good
        # center = np.array([reg.centroid])
        # print(center)
        # points = np.zeros((reg.area, 2)) #np.array(reg.coords[:,0],reg.coords[:,1])
        # points[:,0] = reg.coords[:,0]
        # points[:,1] = reg.coords[:,1]
        # print(points)
        # dis_point2center = cdist(points, center)
        # print(dis_point2center)
        # water_reg_to_water_diss.append(np.nanmean(dis_point2center))
    # water_reg_to_water_diss = raster_tools.map_to_interval(water_reg_to_water_diss,0,1)

    return reg_means,reg_stds,reg_dem_means, reg_to_water_diss, regions, water_reg_means, water_reg_dem_means,water_reg_to_water_diss


def k_mean_cluster(feature_array, n_clusters=8,verbose=True):
    # feature_array = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    # print(feature_array.shape)
    if feature_array.ndim == 1:
        feature_array = feature_array.reshape(-1, 1)    # change to 2d array
    feature_array = np.nan_to_num(feature_array)    # convert nan to 0

    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0).fit(feature_array)
    # >> > kmeans.labels_
    # array([1, 1, 1, 0, 0, 0], dtype=int32)
    # >> > kmeans.predict([[0, 0], [12, 3]])
    # array([1, 0], dtype=int32)
    # >> > kmeans.cluster_centers_
    # array([[10., 2.],
    #        [1., 2.]])

    # print('size:', len(kmeans.labels_),kmeans.labels_)
    # print(kmeans.cluster_centers_)
    if verbose:
        print('K-mean, Number of iterations run:',kmeans.n_iter_)

    return kmeans


def save_k_mean_labels(kmean,regions,img_data,ori_raster,save_path,nan_loc=None,verbose=True):
    # print(kmean.cluster_centers_)
    # n_count, n_feature = kmean.cluster_centers_.shape
    # print(n_count, n_feature)
    # print(kmean.cluster_centers_[:,0])

    magnitudes = [ np.linalg.norm(item) for item in kmean.cluster_centers_ ]
    # print('magnitudes',magnitudes)
    labels = [ idx for idx in range(len(kmean.cluster_centers_))]

    # create new label for clusters, smaller label for smaller magnitudes
    label_sorted = [label for _, label in sorted(zip(magnitudes,labels))]
    # print(labels)
    # print(magnitudes)
    # print(label_sorted)
    label2newLabel = {}
    for idx, labe in enumerate(label_sorted):
        label2newLabel[labe] = idx + 1
    # print(label2newLabel)

    save_np = np.zeros_like(img_data)
    for label, reg in zip(kmean.labels_, regions):
        # save_np[reg.coords[:,0],reg.coords[:,1]] = label + 1
        save_np[reg.coords[:,0],reg.coords[:,1]] = label2newLabel[label]
    max_label = np.max(kmean.labels_)
    if max_label < np.iinfo(np.uint8).max - 1:
        save_np = save_np.astype(np.uint8)
    elif max_label < np.iinfo(np.uint16).max - 1:
        save_np = save_np.astype(np.uint16)
    elif max_label < np.iinfo(np.int32).max - 1:
        save_np = save_np.astype(np.int32)
    else:
        raise ValueError('the label value is out of range')

    if nan_loc is not None:
        save_np[nan_loc] = 0

    raster_tools.save_numpy_array_to_rasterfile(save_np, save_path, ori_raster, nodata=0,
                                                compress='lzw', tiled='yes', bigtiff='if_safer',verbose=verbose)

    return label2newLabel


def kmean_predict(kmean, feature_array):
    if feature_array.ndim == 1:
        feature_array = feature_array.reshape(-1, 1)    # change to 2d array
    feature_array = np.nan_to_num(feature_array)    # convert nan to 0
    out_label = kmean.predict(feature_array)
    return out_label

def identify_similar_cluser(main_cluster_label,kmean,similarity):
    main_cluster_center = kmean.cluster_centers_[main_cluster_label]
    select_labels = []
    for idx, center in enumerate(kmean.cluster_centers_):
        if np.linalg.norm(main_cluster_center - center) <= similarity:
            select_labels.append(idx)
    select_labels.remove(main_cluster_label)
    return select_labels

def save_flood_clusters(kmean_label_path,out_labels, save_path, regions, sel_reg_idxs=None, per_water_loc=None, nan_loc=None,dst_nodata=128,verbose=True):
    kmean_label_np, nodata = raster_tools.read_raster_one_band_np(kmean_label_path)
    save_np = np.zeros_like(kmean_label_np)
    save_np[ np.isin(kmean_label_np,out_labels) ] = 1
    # remove permanent water surface
    if per_water_loc is not None:
        save_np[per_water_loc] = 0

    if sel_reg_idxs is not None:
        select_np = np.zeros_like(save_np)
        for idx in sel_reg_idxs:
            reg = regions[idx]
            select_np[reg.coords[:, 0], reg.coords[:, 1]] = 1
        save_np = save_np*select_np

    if nan_loc is not None:
        save_np[nan_loc] = dst_nodata
    save_np = save_np.astype(np.uint8)
    raster_tools.save_numpy_array_to_rasterfile(save_np,save_path,kmean_label_path,nodata=dst_nodata,
                                                compress='lzw', tiled='yes', bigtiff='if_safer',verbose=verbose)
    return True

def feature_list_to_feature_array(feature_list):
    # each feature should have the same length
    n_samples = len(feature_list[0])
    n_feature =  len(feature_list)
    for idx in range(1,n_feature):
        if len(feature_list[idx]) != n_samples:
            raise ValueError('expected sample count: %d but get %d'%(n_samples, len(feature_list[idx])))
    feature_array = np.zeros((n_samples,n_feature))
    for idx, fea in enumerate(feature_list):
        feature_array[:,idx] = np.array(feature_list[idx])
    feature_array = np.nan_to_num(feature_array)
    return feature_array


def k_mean_cluster_classification(img_data, grd, regions, sar_features_list, n_cluster, p_water_features_list, nan_loc,cluster_label_path,verbose=True):

    t2 = time.time()
    feature_array = feature_list_to_feature_array(sar_features_list)

    kmeans = k_mean_cluster(feature_array, n_clusters=n_cluster,verbose=verbose)
    # print(km_clusters)
    label2newLabel = save_k_mean_labels(kmeans, regions, img_data, grd, cluster_label_path, nan_loc=nan_loc,verbose=verbose)
    print(datetime.now(), 'k-mean cluster analysis, cost %f seconds' % (time.time() - t2))

    # classification: for super-pixels, based on pixels value from permanent body or a global threshold
    water_feature_array = feature_list_to_feature_array(p_water_features_list)
    out_label = list(kmean_predict(kmeans, water_feature_array))
    if verbose:
        print('labels from kmean prediction of permanent water surface', out_label)

    out_transform = kmeans.transform(water_feature_array) # get distance to the cluster centers
    if verbose:
        print('regions of permanent water surface, transform:', out_transform)
    water_cluster_dis = np.min(out_transform,axis=1)
    similarity_dis = np.mean(water_cluster_dis)

    sim_labels = []
    for c_label in out_label:
        sim_labels.extend(identify_similar_cluser(c_label, kmeans, similarity_dis))
    out_label.extend(sim_labels)
    if verbose:
        print('out_label before re-labeling:', out_label)
    out_label = [label2newLabel[item] for item in out_label]
    if verbose:
        print('out_label:', out_label)

    return out_label


def segment_flood_from_SAR_amplitude(sar_image_list, save_dir,n_cluster=20, dst_nodata=128, src_nodata=None, water_mask_file=None,
                                     dem_file=None,g_water_thr=None, verbose=False,process_num=1):
    '''

    :param sar_image_list: a list of SAR Sigma0 image
    :param save_dir: directory for saving results
    :param n_cluster: the number of cluster that k-mean to form
    :param dst_nodata: nodata for output images
    :param src_nodata: nodata for input images
    :param water_mask_file: mask of permanent water surface
    :param dem_file: the DEM file
    :param g_water_thr: a threshold for detecting water pixels, calcuated from other regions
    :param verbose: if set, it will be verbose
    :param process_num: number of process to run the script in parallel
    :return:
    '''

    t0 = time.time()
    total_count = len(sar_image_list)
    mk_outdirectory(save_dir)
    for idx, grd in enumerate(sar_image_list):
        t1 = time.time()
        print(datetime.now(), '(%d/%d) Processing: %s' % (idx + 1, total_count, os.path.basename(grd)))
        file_name_noext = utility.get_name_no_ext(grd)

        save_fd_path = os.path.join(save_dir, file_name_noext + '_FD_result.tif')  # final results
        if os.path.isfile(save_fd_path):
            print('Flood detection results for %s exist, skip'%grd)
            continue

        t2 = time.time()
        # image process, mask nodata region
        img_data, min, max, mean, median = image_read_pre_process(grd, src_nodata=src_nodata,b_normalized=True)
        nan_loc = np.where(np.isnan(img_data))
        print(datetime.now(),'read and preprocess, size:',img_data.shape,'min, max, mean, median',min, max, mean, median,
              'cost %f seconds'%(time.time()-t2))

        # get attributes of permanent water surface
        p_water_loc, p_water_count, p_water_mean,  p_water_std, water_regions, water_regions_path = \
            get_permanent_water_attributes(img_data,grd,nan_loc,water_mask_file,save_dir,verbose=verbose)
        if p_water_count < thr_perma_water_area:
            # do something
            print(datetime.now(), 'no big river or lakes in the scene')
            continue

        # get the corresponding elevation
        if dem_file is not None:
            grd_dem_file = raster_tools.get_elevation_raster(grd, dem_file, save_dir)
        else:
            grd_dem_file = None

        segment_shp_path = os.path.join(save_dir, file_name_noext + '.gpkg')

        # to 8 bit, then segment
        save_8bit_path = os.path.join(save_dir,file_name_noext + '_8bit.tif')
        sar_sigma0_to_8bit(grd, img_data, save_8bit_path, min_percent=0.01, max_percent=0.99,
                           hist_bin_count=10000, src_nodata=None, dst_nodata=0)

        t2 = time.time()
        seg_label = run_segmentation(grd, save_8bit_path, segment_shp_path, save_dir, process_num=process_num, b_vector=False)
        print(datetime.now(),'segmentation of the SAR image cost %f seconds'%(time.time()-t2))

        t2 = time.time()
        reg_means,reg_stds, reg_dem_means, reg_to_water_diss, regions, water_reg_means, water_reg_dem_means,water_reg_to_water_diss \
            = get_object_attributes(img_data,nan_loc, seg_label, dem_path=grd_dem_file, water_regions = water_regions, verbose=verbose, process_num=process_num)
        print(datetime.now(), 'got region attributes, cost %f seconds' % (time.time() - t2))

        # cluster based on super-pixels, using k-mean
        cluster_label_path = os.path.join(save_dir, file_name_noext + '_kmean_label.tif')
        if grd_dem_file is None:
            sar_features_list = [reg_means]
            water_feature_list = [[p_water_mean]]   # take all permanent water region as a whole
        else:
            sar_features_list = [reg_means,reg_dem_means] # , ,reg_to_water_diss
            water_feature_list = [water_reg_means,water_reg_dem_means] # , , water_reg_to_water_diss

        flood_labels = k_mean_cluster_classification(img_data, grd, regions, sar_features_list, n_cluster, water_feature_list,
                                      nan_loc, cluster_label_path,verbose=verbose)

        # select regions reg_to_water_dis < thr
        sel_region_idx = [idx for idx, dis in enumerate(reg_to_water_diss) if dis < thr_dis_flood_to_permaWater]
        if verbose:
            print('select %d regions from %d ones'%(len(sel_region_idx), len(reg_to_water_diss)), 'max and min distance:', np.max(reg_to_water_diss), np.min(reg_to_water_diss))

        # save results
        save_flood_clusters(cluster_label_path,flood_labels,save_fd_path, regions, sel_reg_idxs=sel_region_idx,
                            per_water_loc=p_water_loc,nan_loc=nan_loc,dst_nodata=dst_nodata,verbose=verbose)
        raster_tools.set_water_color_map(save_fd_path)

        # save_metadata
        #save_metadata(granule, infile_path, img_raster_obj, tile_size, array_size, otsus, lms, 20, save_dir)

        # remove unnecessary files
        files_to_rm = [save_8bit_path,segment_shp_path,seg_label, grd_dem_file,water_regions_path,cluster_label_path]
        # for rm_file in files_to_rm:
        #     if os.path.isfile(rm_file):
        #         utility.delete_file_or_dir(rm_file)


        print(datetime.now(), 'Complete, took %s seconds' % (time.time() - t1))

    total_time = time.time() - t0
    print(datetime.now(), 'Flood detection complete, took %s seconds' % (total_time))

    return True


def test_flood_segment_from_SAR_amplitude():
    work_dir = os.path.expanduser('~/Data/tmp_data/flood_detection/Nebraska')
    # img_path = os.path.join(work_dir, 'Nebraska_S1_Sigma0', 'S1B_IW_GRDH_1SDV_20190317T002127_20190317T002156_015387_01CD01_01DD_Sigma0_VH.tif')
    img_path = os.path.join(work_dir, 'Nebraska_S1_Sigma0', 'S1B_IW_GRDH_1SDV_20190317T002127_20190317T002156_015387_01CD01_01DD_Sigma0_VV_sub3.tif')
    sar_image_list = [img_path]
    n_cluster = 20
    save_dir = os.path.join(work_dir,'fd_segmentation')
    water_mask_tif = os.path.expanduser('~/Data/global_surface_water/extent_epsg4326_theUS/surface_water_theUS_3_2020.tif')
    # dem_file = os.path.expanduser('~/Data/flooding_area/DEM/SRTM_Nebraska/nebraska_SRTM.tif')
    dem_file = None

    segment_flood_from_SAR_amplitude(sar_image_list,save_dir,n_cluster=n_cluster,src_nodata=0,
                                     water_mask_file=water_mask_tif,dem_file=dem_file, verbose=False,process_num=4)

def main(options, args):
    # test_flood_segment_from_SAR_amplitude()

    sar_image_list = get_sar_file_list(args[0])
    sar_image_list = [os.path.abspath(item) for item in sar_image_list]
    save_dir = os.path.abspath(options.save_dir)
    water_mask = options.water_mask
    dem_file = options.elevation_file
    verbose = options.verbose

    src_nodata = options.src_nodata
    dst_nodata = options.out_nodata
    process_num = options.process_num
    global_water_threshold = options.global_water_threshold
    n_clusters = options.kmean_cluster

    print(datetime.now(), 'Found %d SAR Sigma0 images from %s:' % (len(sar_image_list), args[0]))
    print(datetime.now(), 'Will save flood detection results to %s' % save_dir)

    segment_flood_from_SAR_amplitude(sar_image_list, save_dir, n_cluster=n_clusters, src_nodata=src_nodata, dst_nodata=dst_nodata,
                                     water_mask_file=water_mask,dem_file=dem_file, process_num=process_num)



if __name__ == '__main__':
    usage = "usage: %prog [options] sar_files.txt or image_directory "
    parser = OptionParser(usage=usage, version="1.0 2022-11-02")
    parser.description = 'Introduction: flood detection from SAR imagery '

    parser.add_option("-d", "--save_dir",
                      action="store", dest="save_dir", default='FD_results',
                      help="the folder to save pre-processed results")

    parser.add_option("-n", "--src_nodata",
                      action="store", dest="src_nodata", type=float,
                      help="no data value for input data")

    parser.add_option("-m", "--out_nodata",
                      action="store", dest="out_nodata", type=int, default=128,
                      help="no data value for the output raster")

    parser.add_option("-k", "--kmean_cluster",
                      action="store", dest="kmean_cluster", type=int, default=20,
                      help="The number of clusters to form")

    parser.add_option("-g", "--global_water_threshold",
                      action="store", dest="global_water_threshold", type=float,
                      help="a threshold (<) calculated from other regions for identifying water pixels")

    parser.add_option("-p", "--process_num",
                      action="store", dest="process_num", type=int, default=1,
                      help="the process to run the detection in parallel")

    parser.add_option("-w", "--water_mask",
                      action="store", dest="water_mask",
                      help="a file containing the permanent water surface")

    parser.add_option("-e", "--elevation_file",
                      action="store", dest="elevation_file",
                      help="path to the DEM")

    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="setting this to enable outputting log message and png files of histogram")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)