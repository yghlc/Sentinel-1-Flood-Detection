#!/usr/bin/env python
# Filename: flood_depth.py 
"""
introduction: estimate flood depth

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 01 January, 2023
"""

import os,sys
from optparse import OptionParser
import glob
from datetime import datetime
import time
import numpy as np

import utility
import raster_tools
import vector_gpd
import math

import rasterio
from multiprocessing import Pool

metadata_path = 'flood_depth_meta.json'

min_area = 100 # pixels
water_height_diff_thr = 1

def update_metadata_path(flood_depth_path):
    global metadata_path
    save_dir = os.path.dirname(flood_depth_path)
    filename = utility.get_name_no_ext(flood_depth_path)
    metadata_path = os.path.join(save_dir, filename+'_meta.json')



def floodmap_to_polygons(flood_map_tif,save_shp_path, min_area_thr, work_dir='./'):

    if os.path.isfile(save_shp_path):
        print('warning, %s already exist'%save_shp_path)
        return True

    tmp_tif = os.path.join(work_dir,'tmp.tif')

    # convert raster to 1, 0, ( previously, nodata is 128), so ignore the nodata
    command_str = 'gdal_calc.py --hideNoData --calc="A==1" --outfile=%s -A %s '%(tmp_tif,flood_map_tif)
    utility.os_system_exit_code(command_str)

    # set 0 as nodata
    command_str = 'gdal_edit.py -a_nodata 0 %s'%(tmp_tif)
    utility.os_system_exit_code(command_str)

    # to shapefile
    out_shp = utility.get_name_by_adding_tail(save_shp_path,'all')
    vector_gpd.raster2shapefile(tmp_tif,out_shp,connect8=True,format='ESRI Shapefile')

    # remove small polygons
    polys = vector_gpd.read_polygons_gpd(out_shp,b_fix_invalid_polygon=False)
    area_list = [item.area for item in polys]
    out_area_shp = utility.get_name_by_adding_tail(save_shp_path,'rmArea')
    vector_gpd.remove_polygons_based_values(out_shp,area_list,min_area_thr,True,out_area_shp)

    # fill holes
    vector_gpd.fill_holes_in_polygons_shp(out_area_shp,save_shp_path)

    # delete tmp.tif
    utility.delete_file_or_dir(tmp_tif)
    utility.delete_shape_file(out_shp)
    utility.delete_shape_file(out_area_shp)

    return True

def read_pixels_along_a_line(line, line_width, raster_path):
    line_poly = line.buffer(line_width)
    try:
        out_image, out_transform, nodata = raster_tools.read_raster_in_polygons_mask(raster_path,line_poly)
        # print('out_image shape',out_image.shape)
        data = out_image[out_image != nodata]
        # print('out image exclude nodata', data, data.size, np.max(data), np.min(data))
        return data
    except ValueError:
        print(datetime.now(),"ValueError, the line does not overlap valid pixels")
        return None

def cal_water_height(index, pixel_heights, height_diff_thr = 1):
    # assume a flat water surface, ref:
    # Cian, F., Marconcini, M., Ceccato, P., & Giupponi, C. (2018). Flood depth estimation by means of high-resolution
    # SAR images and lidar data. Natural Hazards and Earth System Sciences, 18(11), 3063-3084.

    # hist, bins = np.histogram(pixel_heights)
    # print('h_hist',hist, bins)
    # # get distribution
    # print('out image exclude nodata',pixel_heights.size, np.max(pixel_heights), np.min(pixel_heights),
    #       np.mean(pixel_heights),np.std(pixel_heights))

    if pixel_heights is None:
        return None

    height = np.sort(pixel_heights)     # sort, small to large
    # print(height)
    h_arrays = np.array_split(height,100)
    # print(height_100_subs)
    water_height = None
    n = 95
    s = 5
    for i in range(0,n-s):
        if h_arrays[n-i].size < 1 or h_arrays[n-s-i].size < 1:
            continue
        if h_arrays[n-i][-1] - h_arrays[n-s-i][-1] <= height_diff_thr:
            water_height = (h_arrays[n-i][-1] + h_arrays[n-s-i][-1])/2.0
            break
        if (n-5-i)==50:
            print('Warning: the estimation of water height maybe not correct, idx: %d'%index)

    if water_height is None:
        print('Warning: using the median value as water height, idx: %d' % index)
        water_height = np.median(height)


    return water_height

def estimate_water_surface_height_one(index, line, line_width, water_height_diff_thr, dem_path):
    line_ele_data = read_pixels_along_a_line(line, line_width, dem_path)
    water_height = cal_water_height(index, line_ele_data, water_height_diff_thr)
    return water_height


def estimate_water_surface_height(water_polys_shp,line_width, dem_path,process_num=1):

    polygons = vector_gpd.read_polygons_gpd(water_polys_shp,b_fix_invalid_polygon=True)
    polygon_outlines = [vector_gpd.polygon_to_outline(item) for item in polygons]

    if process_num == 1:
        # out_image, out_transform, nodata = raster_tools.read_raster_in_polygons_mask(dem_path,polygons[0])
        line_ele_data_list = [read_pixels_along_a_line(line, line_width, dem_path) for line in polygon_outlines]
        # estimate water surface
        water_height_list = [cal_water_height(idx, item,water_height_diff_thr) for idx,item in enumerate(line_ele_data_list)]
    elif process_num > 1:
        theadPool = Pool(process_num)
        parameters_list = [(idx, line, line_width, water_height_diff_thr, dem_path) for idx, line in enumerate(polygon_outlines)]
        results = theadPool.starmap(estimate_water_surface_height_one, parameters_list)
        water_height_list = [res for res in results]
        theadPool.close()
    else:
        raise ValueError('input process_num is wrong: %s'%str(process_num))

    # save to the shp file
    idx_list = [item for item in range(len(water_height_list))]
    attributes = {'idx': idx_list,'water_h':water_height_list}
    vector_gpd.add_attributes_to_shp(water_polys_shp,attributes)


    return polygons, water_height_list

def test_estimate_water_surface_height():

    test_dir = os.path.expanduser('~/Data/tmp_data/flood_detection/code_debuging')
    # flood_polys_shp = os.path.join(test_dir, 'fd_depth', 's1_houston_code_debug_VH_LM_polys_test.shp')
    flood_polys_shp = os.path.join(test_dir, 'fd_depth', 's1_houston_code_debug_VH_LM_test_one.shp')
    dem_tif = os.path.expanduser('~/Data/flooding_area/DEM/Houston/Houston_SRTM.tif')

    res = 0.000089831528412

    estimate_water_surface_height(flood_polys_shp, res/2,dem_tif)

def cal_flood_depth_one(idx, w_poly,w_h,flood_map,flood_raster_transform,new_dem_tif,depth_np):
    if w_h is None:
        return None
    minx, miny, maxx, maxy = vector_gpd.get_polygon_bounding_box(w_poly)

    xs = [minx, maxx]
    ys = [maxy, miny]  # maxy (uppper left),  miny (lower right)
    rows, cols = rasterio.transform.rowcol(flood_raster_transform, xs, ys)  # ,op=math.floor
    # there is one pixel offset
    rows = [item + 1 for item in rows]
    cols = [item + 1 for item in cols]

    sub_flood_np = flood_map[rows[0]:rows[1], cols[0]:cols[1]]  # +1?
    # print('sub_flood_np shape:',sub_flood_np.shape)

    # read dem within the polygon
    dem_2d, dem_transform, dem_nodata = raster_tools.read_raster_in_polygons_mask(new_dem_tif, w_poly)
    dem_2d = np.squeeze(dem_2d)

    # # adjust the row, col a little bit
    # if rows[1] - rows[0] < dem_2d.shape[0]:
    #     rows[1] += 1
    # if cols[1] - cols[0] < dem_2d.shape[1]:
    #     cols[1] += 1

    # adjust the dem a little bit, the dem array tend to a little bit large (1 pixel)
    if rows[1] - rows[0] < dem_2d.shape[0] or cols[1] - cols[0] < dem_2d.shape[1]:
        dem_2d = dem_2d[0:rows[1] - rows[0], 0:cols[1] - cols[0]]

    # calculate the depth
    depth = w_h - dem_2d
    # print('depth shape:', depth.shape)
    if sub_flood_np.shape != depth.shape:
        raise ValueError('idx: %d,the size of sub_flood_np %s and depth is different %s' % (idx, str(sub_flood_np.shape), str(depth.shape)))

    # save to the entire map
    water_loc = np.where(np.logical_and(sub_flood_np == 1, dem_2d != dem_nodata))
    water_loc_org = (water_loc[0] + rows[0], water_loc[1] + cols[0])
    depth_np[water_loc_org] = depth[water_loc]


def cal_flood_depth(flood_map_tif,dem_tif,water_polygons,water_height_list,save_path,process_num=1):

    flood_map, nodata = raster_tools.read_raster_one_band_np(flood_map_tif)
    flood_raster_transform = raster_tools.get_transform_from_file(flood_map_tif)
    non_data_loc = np.where(flood_map == nodata)

    # resample and crop the DEM
    save_dir = os.path.dirname(save_path)
    name, ext = os.path.splitext(os.path.basename(flood_map_tif))
    new_dem_tif = os.path.join(save_dir,name + '_dem' + ext)

    out = raster_tools.resample_crop_raster(flood_map_tif, dem_tif, output_raster=new_dem_tif, resample_method='bilinear') #near
    if out is False:
        return False
    dem_data_np, dem_nodata = raster_tools.read_raster_one_band_np(new_dem_tif)
    dem_nodata_loc = np.where(dem_data_np == dem_nodata)

    depth_nodata = -9999.0
    depth_np = np.zeros_like(flood_map).astype(np.float32)
    # depth_np[:] = depth_nodata

    # for process_num, because when run in parallel, data in depth_np is random.
    # currently, no solution. modifying numpy array in parallel cause problems.
    process_num = 1

    if process_num==1:
        for idx, (w_poly, w_h) in enumerate(zip(water_polygons,water_height_list)):
            cal_flood_depth_one(idx, w_poly, w_h, flood_map, flood_raster_transform, new_dem_tif, depth_np)
    elif process_num > 1:
        theadPool = Pool(process_num)
        parameters_list = [(idx, w_poly, w_h, flood_map, flood_raster_transform, new_dem_tif, depth_np)
                           for idx, (w_poly, w_h) in enumerate(zip(water_polygons,water_height_list))]
        results = theadPool.starmap(cal_flood_depth_one, parameters_list)
        theadPool.close()
    else:
        raise ValueError('input process_num is wrong: %s' % str(process_num))



    depth_np[non_data_loc] = depth_nodata
    depth_np[dem_nodata_loc] = depth_nodata # mask dem nodata regions
    raster_tools.save_numpy_array_to_rasterfile(depth_np,save_path,flood_map_tif,nodata=depth_nodata,
                                                compress='lzw', tiled='yes', bigtiff='if_safer')
    raster_tools.set_ColorInterp_grey(save_path)


def test_cal_flood_depth():

    test_dir = os.path.expanduser('~/Data/tmp_data/flood_detection/code_debuging')
    flood_tif = os.path.join(test_dir, 'fd_results', 's1_houston_code_debug_VH_LM.tif')
    dem_tif = os.path.expanduser('~/Data/flooding_area/DEM/Houston/Houston_SRTM.tif')
    # flood_polys_shp = os.path.join(test_dir, 'fd_depth', 's1_houston_code_debug_VH_LM_test_one.shp')
    flood_polys_shp = os.path.join(test_dir, 'fd_depth', 's1_houston_code_debug_VH_LM_test_two.shp')
    save_path = os.path.join(test_dir, 'fd_depth', 's1_houston_code_debug_VH_LM_oneRegion_depth.tif')

    res = 0.000089831528412
    water_polys, water_height_list = estimate_water_surface_height(flood_polys_shp, res / 2, dem_tif)
    cal_flood_depth(flood_tif, dem_tif, water_polys, water_height_list,save_path)


def estimate_flood_depth(flood_map_tif, dem_tif, save_path,process_num=1,b_verbose=True):
    '''
    estimate flood depth
    :param flood_map_tif: flood binary map in raster format
    :param dem_tif: dem
    :param save_path: output
    :param process_num: process number
    :param b_verbose: b verbose
    :return: True if successful, False otherwise
    '''
    t0 = time.time()

    update_metadata_path(save_path)

    # check projection
    flood_prj = raster_tools.get_projection(flood_map_tif,'epsg')

    dem_prj = raster_tools.get_projection(dem_tif,'epsg')
    if flood_prj != dem_prj:
        raise ValueError('The projection between %s and %s is different'%(flood_map_tif,dem_tif))

    xres, yres = raster_tools.get_xres_yres_file(flood_map_tif)
    if (xres - yres) > 1e-9:
        raise ValueError('resolution in x (%f) and y (%f) direction is different' % (xres, yres))

    utility.write_metadata(['flood-map', 'DEM', 'resolution'], [flood_map_tif, dem_tif,xres], filename=metadata_path)
    utility.write_metadata(['min-region-in-pixel', 'water-height-diff-thr'], [min_area, water_height_diff_thr], filename=metadata_path)

    area_thr = min_area*xres*yres
    # print(area_thr)

    # flood map to polygons
    save_dir = os.path.dirname(save_path)
    save_shp_path = os.path.join(save_dir,utility.get_name_no_ext(flood_map_tif) + '_polys.shp')
    t1 = time.time()
    floodmap_to_polygons(flood_map_tif, save_shp_path,area_thr,work_dir=save_dir)
    if b_verbose:
        print(datetime.now(), 'flood map to polygons, took %f seconds'%(time.time() - t1) )

    # get water height for each flood regions
    t1 = time.time()
    water_polys, water_height_list = estimate_water_surface_height(save_shp_path, xres/2.0, dem_tif, process_num=process_num)
    if b_verbose:
        print(datetime.now(), 'estimate the height of water surface, took %f seconds'%(time.time() - t1) )

    # calculate flood depth for each region
    t1 = time.time()
    cal_flood_depth(flood_map_tif,dem_tif, water_polys, water_height_list,save_path,process_num=process_num)
    if b_verbose:
        print(datetime.now(), 'calculate flood depth, took %f seconds'%(time.time() - t1))

    print(datetime.now(), 'Estimate depth for %s complete, took %f seconds' % (os.path.basename(flood_map_tif), time.time() - t0))


def test_estimate_flood_depth():
    test_dir = os.path.expanduser('~/Data/tmp_data/flood_detection/code_debuging')
    flood_tif = os.path.join(test_dir, 'fd_results', 's1_houston_code_debug_VH_LM.tif')
    dem_tif = os.path.expanduser('~/Data/flooding_area/DEM/Houston/Houston_SRTM.tif')

    save_path = os.path.join(test_dir, 'fd_depth', utility.get_name_no_ext(flood_tif) +'_depth.tif')
    estimate_flood_depth(flood_tif, dem_tif, save_path, process_num=1, b_verbose=True)


def main(options, args):
    # test_estimate_flood_depth()
    # test_estimate_water_surface_height()
    # test_cal_flood_depth()

    flood_map = args[0]
    dem_tif = args[1]
    process_num = options.process_num
    b_verbose = options.verbose
    save_dir = options.save_dir if options.save_dir is not None else './'
    if options.min_pixel_count is not None:
        global min_area
        min_area = options.min_pixel_count
    if options.water_height_diff_thr is not None:
        global water_height_diff_thr
        water_height_diff_thr = options.water_height_diff_thr


    utility.is_file_exist(flood_map)
    utility.is_file_exist(dem_tif)

    if os.path.isdir(save_dir) is False:
        utility.mkdir(save_dir)

    save_path = os.path.join(save_dir, utility.get_name_no_ext(flood_map) +'_depth.tif')
    estimate_flood_depth(flood_map, dem_tif, save_path, process_num=process_num, b_verbose=b_verbose)


if __name__ == "__main__":
    usage = "usage: %prog [options] flood_map.tif dem.tif  "
    parser = OptionParser(usage=usage, version="1.0 2023-01-01")
    parser.description = 'Introduction: estimate flood depth '

    parser.add_option("-d", "--save_dir",
                      action="store", dest="save_dir",default='depth_results',
                      help="the folder to save pre-processed results")

    parser.add_option("-n", "--src_nodata",
                      action="store", dest="src_nodata", type=float,
                      help="no data value for input data")

    parser.add_option("-p", "--process_num",
                      action="store", dest="process_num", type=int, default=1,
                      help="the process to run the detection in parallel")

    parser.add_option("-m", "--min_pixel_count",
                      action="store", dest="min_pixel_count", type=int,
                      help="ignore flood regions smaller than min_pixel_count")

    parser.add_option("-w", "--water_height_diff_thr",
                      action="store", dest="water_height_diff_thr", type=float,
                      help="a threshold for estimating water height")

    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose",default=False,
                      help="setting this to enable outputting log message and png files of histogram")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)
