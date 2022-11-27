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

def get_object_attributes(org_img_data, label_path, process_num=1):
    raster_tools.unset_nodata(label_path)
    label_raster_np, label_nodata =  raster_tools.read_raster_one_band_np(label_path)
    # print(label_raster_np, label_nodata)

    labels = measure.label(label_raster_np,background=label_nodata,connectivity=2)  # 2-connectivity, 8 neighbours

    # comparing with those polygons and their attributes after gdal_polygonize.py, the polygon attributes are calculated
    # in or "touched by" the polygon
    # get regions
    regions = measure.regionprops(labels)
    print('the count of regions:', len(regions))

    # print(regions)
    # calculate attribute: mean, std of SAR sigma0
    # print(org_img_data.shape)

    # total_pixel_count = 0
    reg_means = []
    # reg_areas = []
    reg_stds = []

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

    return reg_means,reg_stds,regions # reg_stds, reg_areas,


def k_mean_cluster(feature_array, n_clusters=8):
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

    print('size:', len(kmeans.labels_),kmeans.labels_)
    # print(kmeans.cluster_centers_)
    print('K-mean, Number of iterations run:',kmeans.n_iter_)

    return kmeans


def save_k_mean_labels(kmean,regions,img_data,ori_raster,save_path,nan_loc=None):
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
                                                compress='lzw', tiled='yes', bigtiff='if_safer')

    return label2newLabel


def kmean_predict(kmean, feature_array):
    if feature_array.ndim == 1:
        feature_array = feature_array.reshape(-1, 1)    # change to 2d array
    feature_array = np.nan_to_num(feature_array)    # convert nan to 0
    out_label = kmean.predict(feature_array)
    return out_label

def save_flood_clusters(kmean_label_path,out_labels, save_path, per_water_loc=None, nan_loc=None,dst_nodata=128):
    kmean_label_np, nodata = raster_tools.read_raster_one_band_np(kmean_label_path)
    save_np = np.zeros_like(kmean_label_np)
    if nan_loc is not None:
        save_np[nan_loc] = dst_nodata
    save_np[ np.isin(kmean_label_np,out_labels) ] = 1
    # remove permanent water surface
    if per_water_loc is not None:
        save_np[per_water_loc] = 0
    raster_tools.save_numpy_array_to_rasterfile(save_np,save_path,kmean_label_path,nodata=dst_nodata,
                                                compress='lzw', tiled='yes', bigtiff='if_safer')
    return True


def segment_flood_from_SAR_amplitude(sar_image_list, save_dir,n_cluster=20, dst_nodata=128, src_nodata=None, water_mask_file=None,g_water_thr=None,
                                    verbose=False,process_num=1):
    '''

    :param sar_image_list: a list of SAR Sigma0 image
    :param save_dir: directory for saving results
    :param n_cluster: the number of cluster that k-mean to form
    :param dst_nodata: nodata for output images
    :param src_nodata: nodata for input images
    :param water_mask_file: mask of permanent water surface
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

        flood_seg_path = os.path.join(save_dir, file_name_noext + '_fd.tif')  # final results
        if os.path.isfile(flood_seg_path):
            print('Flood detection results for %s exist, skip'%grd)
            continue

        t2 = time.time()
        # image process, mask nodata region
        img_data, min, max, mean, median = image_read_pre_process(grd, src_nodata=src_nodata)
        nan_loc = np.where(np.isnan(img_data))
        print(datetime.now(),'read and preprocess, size:',img_data.shape,'min, max, mean, median',min, max, mean, median,
              'cost %f seconds'%(time.time()-t2))

        segment_shp_path = os.path.join(save_dir, file_name_noext + '.gpkg')

        # to 8 bit, then segment
        save_8bit_path = os.path.join(save_dir,file_name_noext + '_8bit.tif')
        sar_sigma0_to_8bit(grd, img_data, save_8bit_path, min_percent=0.01, max_percent=0.99,
                           hist_bin_count=10000, src_nodata=None, dst_nodata=0)

        t2 = time.time()
        seg_label = run_segmentation(grd, save_8bit_path, segment_shp_path, save_dir, process_num=process_num, b_vector=False)
        print(datetime.now(),'segmentation cost %f seconds'%(time.time()-t2))

        t2 = time.time()
        reg_means,reg_stds,regions = get_object_attributes(img_data,seg_label,process_num=process_num)
        print(datetime.now(), 'got region attributes, cost %f seconds' % (time.time() - t2))

        # cluster based on super-pixels, using k-mean
        t2 = time.time()
        feature_array = np.array(reg_means)
        # feature_array = np.zeros((len(reg_means),2))
        # feature_array[:,0] = np.array(reg_means)
        # feature_array[:,1] = np.array(reg_stds)
        kmeans = k_mean_cluster(feature_array,n_clusters=n_cluster)
        # print(km_clusters)
        save_cluster_path = os.path.join(save_dir,file_name_noext + '_kmean_label.tif')
        label2newLabel = save_k_mean_labels(kmeans, regions, img_data, grd, save_cluster_path,nan_loc=nan_loc)
        print(datetime.now(), 'k-mean cluster analysis, cost %f seconds' % (time.time() - t2))

        # classification: for super-pixels, based on pixels value from permanent body or a global threshold
        p_water_loc, p_water_count, p_water_min, p_water_max, p_water_mean, p_water_median, p_water_std = \
            permant_water_pixles(img_data, grd, water_mask_file, save_dir)
        if p_water_count < 5000:
            # do something
            continue
        water_feature_array = np.array([p_water_mean])
        out_label = kmean_predict(kmeans,water_feature_array)
        out_label = [label2newLabel[item] for item in out_label]
        print('out_label:',out_label)

        # save results
        save_fd_path =  os.path.join(save_dir,file_name_noext + '_FD_result.tif')
        save_flood_clusters(save_cluster_path,out_label,save_fd_path,per_water_loc=p_water_loc,nan_loc=nan_loc,dst_nodata=dst_nodata)
        raster_tools.set_water_color_map(save_fd_path)

        # save_metadata
        #save_metadata(granule, infile_path, img_raster_obj, tile_size, array_size, otsus, lms, 20, save_dir)

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

    segment_flood_from_SAR_amplitude(sar_image_list,save_dir,n_cluster=n_cluster,src_nodata=0, water_mask_file=water_mask_tif,process_num=4)

def main(options, args):
    # test_flood_segment_from_SAR_amplitude()

    sar_image_list = get_sar_file_list(args[0])
    sar_image_list = [os.path.abspath(item) for item in sar_image_list]
    save_dir = os.path.abspath(options.save_dir)
    water_mask = options.water_mask
    verbose = options.verbose

    src_nodata = options.src_nodata
    dst_nodata = options.out_nodata
    process_num = options.process_num
    global_water_threshold = options.global_water_threshold
    n_clusters = options.kmean_cluster

    print(datetime.now(), 'Found %d SAR Sigma0 images from %s:' % (len(sar_image_list), args[0]))
    print(datetime.now(), 'Will save flood detection results to %s' % save_dir)

    segment_flood_from_SAR_amplitude(sar_image_list, save_dir, n_cluster=n_clusters, src_nodata=src_nodata, dst_nodata=dst_nodata, water_mask_file=water_mask,
                                     process_num=process_num)



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

    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="setting this to enable outputting log message and png files of histogram")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)