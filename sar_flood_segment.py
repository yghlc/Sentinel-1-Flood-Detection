#!/usr/bin/env python
# Filename: sar_flood_segment.py 
"""
introduction: Object-based image analysis,

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 24 November, 2022
"""

import os, sys
import time
import raster_tools
from datetime import datetime
import numpy as np

from raster_tools import image_read_pre_process
from utility import mk_outdirectory
import utility
from grey_image_segment import segment_a_grey_image

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


def segment_flood_from_SAR_amplitude(sar_image_list, save_dir,dst_nodata=128, src_nodata=None, water_mask_file=None,g_water_thr=None,
                                    verbose=False,process_num=1):
    '''

    :param sar_image_list: a list of SAR Sigma0 image
    :param save_dir: directory for saving results
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

        # image process, mask nodata region
        img_data, min, max, mean, median = image_read_pre_process(grd, src_nodata=src_nodata)
        print(datetime.now(),'read and preprocess, size:',img_data.shape,'min, max, mean, median',min, max, mean, median)

        # to 8 bit, then segment
        save_8bit_path = os.path.join(save_dir,file_name_noext + '_8bit.tif')
        sar_sigma0_to_8bit(grd, img_data, save_8bit_path, min_percent=0.01, max_percent=0.99,
                           hist_bin_count=10000, src_nodata=None, dst_nodata=0)


        # cluster based on super-pixels, using k-mean
        # get initial polygons
        # because the label from segmentation for superpixels are not unique, so we may need to get mean dem diff based on polygons, set org_raster=None
        label_path = segment_a_grey_image(save_8bit_path, save_dir, process_num, org_raster=None)

        # classification: for super-pixels


        # p_water_loc, p_water_count, p_water_min, p_water_max, p_water_mean, p_water_median, p_water_std = \
        #     permant_water_pixles(img_data, grd, water_mask_file, save_dir)
        # 
        # # apply thresholding and save results
        # # ---------------------------------------------------------------------------
        # ## Save Meta data file
        # granule = os.path.basename(grd)
        # infile_path = os.path.dirname(grd)      # image directory
        # img_raster_obj = raster_tools.open_raster_read(grd)
        # # save_metadata(granule, infile_path, img_raster_obj, tile_size, array_size, otsus, lms, 20, save_dir)
        # 
        # inan = np.where(np.isnan(img_data))
        # lm_map[inan] = dst_nodata  ## convert no data values
        # # lm_map[p_water_loc] = 0 ## apply permanent water mask
        # lm_map = lm_map.astype(np.uint8)
        # map_type = 'LM'
        # tiff_outname = write_geotiff(save_dir, img_raster_obj, granule, lm_map, map_type, nodata=dst_nodata, compress='lzw',b_colormap=True)  ## Write geotiff

        print(datetime.now(), 'Complete, took %s seconds' % (time.time() - t1))

    total_time = time.time() - t0
    print(datetime.now(), 'Flood detection complete, took %s seconds' % (total_time))

    return True


def test_flood_segment_from_SAR_amplitude():
    work_dir = os.path.expanduser('~/Data/tmp_data/flood_detection/Nebraska')
    img_path = os.path.join(work_dir, 'Nebraska_S1_Sigma0', 'S1B_IW_GRDH_1SDV_20190317T002127_20190317T002156_015387_01CD01_01DD_Sigma0_VH.tif')
    sar_image_list = [img_path]
    save_dir = os.path.join(work_dir,'fd_segmentation')
    water_mask_tif = os.path.expanduser('~/Data/global_surface_water/extent_epsg4326_theUS/surface_water_theUS_3_2020.tif')

    segment_flood_from_SAR_amplitude(sar_image_list,save_dir,src_nodata=0, water_mask_file=water_mask_tif,process_num=4)

def main():
    test_flood_segment_from_SAR_amplitude()

if __name__ == '__main__':
    main()