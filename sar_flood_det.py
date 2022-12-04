#!/usr/bin/env python
# Filename: sar_flood_det.py 
"""
introduction: detection of flood from SAR imagery

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 02 November, 2022
"""
import os,sys
from optparse import OptionParser
import glob
from datetime import datetime
import time

import numpy as np

from SAR_Flood_Detection_v02 import Run_amplitude_algorithm,save_metadata,write_geotiff,mk_outdirectory

import raster_tools
from raster_tools import image_read_pre_process,permant_water_pixles

from BimodalThreshold_module_v02 import BimodalThreshold

from utility import get_sar_file_list
import utility


def flood_detection_from_SAR_amplitude(sar_image_list, save_dir,dst_nodata=128, src_nodata=None, water_mask_file=None,g_water_thr=None,
                                       ptf=False,v=0.1,verbose=False,process_num=1):
    t0 = time.time()
    total_count = len(sar_image_list)
    mk_outdirectory(save_dir)
    for idx, grd in enumerate(sar_image_list):
        t1 = time.time()
        print(datetime.now(), '(%d/%d) Processing: %s' % (idx + 1, total_count, os.path.basename(grd)))

        filename, ext = os.path.splitext(os.path.basename(grd))
        final_LM_map = os.path.join(save_dir, filename + '_' + 'LM' + ext)
        if os.path.isfile(final_LM_map):
            print('%s already exists, skip'%final_LM_map)
            continue

        # image process, mask nodata region
        img_data, min, max, mean, median = image_read_pre_process(grd, src_nodata=src_nodata,b_normalized=True)
        print(datetime.now(),'read and preprocess, size:',img_data.shape,'min, max, mean, median',min, max, mean, median)

        if ptf:
            img_data = np.where(img_data > 0., img_data ** v, 0.)  # power transform

        p_water_loc, p_water_count, p_water_min, p_water_max, p_water_mean, p_water_median, p_water_std, grd_p_water_file = \
            permant_water_pixles(img_data, grd, water_mask_file, save_dir)

        # run bimodal threshold
        tile_size = 1456*3
        array_size = 182*3
        B_thresh = 0.65
        b_otsu = False
        bt = BimodalThreshold(grd,save_dir,tile_size,array_size,B_thresh,b_otsu=b_otsu)

        otsus, lms = bt.otsu_and_lm_for_an_image(img_data,verbose=verbose,process_num=process_num)

        # apply thresholding and save results
        # ---------------------------------------------------------------------------
        ## Save Meta data file
        granule = os.path.basename(grd)
        infile_path = os.path.dirname(grd)      # image directory
        img_raster_obj = raster_tools.open_raster_read(grd)
        save_metadata(granule, infile_path, img_raster_obj, tile_size, array_size, otsus, lms, 20, save_dir)
        print("OTSUS: ", otsus)  # this prints otsus results for each subtile - the lower threshold
        print("LMS: ", lms)  # This prints out the LM results for each subtile - the upper threshold

        # ## Create permanent water body masks
        # mask_outfilename = create_water_masks(surface_water_dir, surface_water_fname, pixel_size_lon_deg,
        #                                       pixel_size_lat_deg, minLon, maxLon, minLat, maxLat)

        # ---------------------------------------------------------------------------
        ### Write output as Geotiff - lm
        lm = np.mean(lms)  # Calculate the mean of the upper threshold (lms) for all regions:
        for tmp in sorted(lms):
            if tmp > p_water_mean and tmp > p_water_median:
                print(datetime.now(), 'Choose %.4f as the threshold, LM mean is %.4f'%(tmp, lm))
                lm = tmp
                break
        if p_water_count > 5000 and lm > p_water_mean + 3*p_water_std:
            print(datetime.now(),'Warning, the mean of LM is too large')
            if g_water_thr is not None:
                print(datetime.now(), 'use the threshold for global water instead')
                lm_map = np.where(img_data < g_water_thr, 1, 0)
            else:
                print(datetime.now(), 'use the (mean+2*std) over permanent water as threshold')
                lm_map = np.where(img_data < p_water_mean + 2*p_water_std, 1, 0)
        else:
            lm_map = np.where(img_data < lm, 1, 0)  # lm_map is a binary image of pixels above the threshold; Converts values greater than the mean (i.e. industrial(?)) to 1, all else to 0; a binary image of water(1) and non-water(0)
        inan = np.where(np.isnan(img_data))
        lm_map[inan] = dst_nodata  ## convert no data values
        # lm_map[p_water_loc] = 0 ## apply permanent water mask
        lm_map = lm_map.astype(np.uint8)
        map_type = 'LM'
        tiff_outname = write_geotiff(save_dir, img_raster_obj, granule, lm_map, map_type, nodata=dst_nodata, compress='lzw',b_colormap=True)  ## Write geotiff

        if len(otsus) > 0:
            # ---------------------------------------------------------------------------
            ### Write output as Geotiff - otsu
            otsu_mean = np.mean(otsus)  # Calculate the mean of the upper threshold (lms) for all regions:
            otsu_map = np.where(img_data < otsu_mean, 1,0)  # is a binary image of pixels above the threshold; Converts values greater than the mean (i.e. industrial(?)) to 1, all else to 0; a binary image of water(1) and non-water(0)
            otsu_map[inan] = dst_nodata  ## convert no data values
            # otsu_map[p_water_loc] = 0  ## apply permanent water mask
            otsu_map = otsu_map.astype(np.uint8)
            map_type = 'OTSU'
            tiff_outname = write_geotiff(save_dir, img_raster_obj, granule, otsu_map, map_type, nodata=dst_nodata, compress='lzw', b_colormap=True)

        utility.delete_file_or_dir(grd_p_water_file)

        print(datetime.now(), 'Complete, took %s seconds' % (time.time() - t1))

    total_time = time.time() - t0
    print(datetime.now(), 'Flood detection complete, took %s seconds' % (total_time))



def main(options, args):

    sar_image_list = get_sar_file_list(args[0])
    sar_image_list = [os.path.abspath(item) for item in sar_image_list]
    save_dir = os.path.abspath(options.save_dir)
    water_mask = options.water_mask
    verbose = options.verbose

    src_nodata = options.src_nodata
    dst_nodata = options.out_nodata
    process_num = options.process_num
    global_water_threshold = options.global_water_threshold

    print(datetime.now(), 'Found %d SAR Sigma0 images from %s:'%(len(sar_image_list),args[0]))
    print(datetime.now(), 'Will save flood detection results to %s'%save_dir)

    flood_detection_from_SAR_amplitude(sar_image_list, save_dir, dst_nodata=dst_nodata,src_nodata=src_nodata,
                                       water_mask_file=water_mask, g_water_thr=global_water_threshold, ptf=False, v=0.1,verbose=verbose,process_num=process_num)

    # Run_amplitude_algorithm(sar_image_list, save_dir,os.path.dirname(water_mask),os.path.basename(water_mask),verbose=verbose)



if __name__ == "__main__":
    usage = "usage: %prog [options] sar_files.txt or image_directory "
    parser = OptionParser(usage=usage, version="1.0 2022-11-02")
    parser.description = 'Introduction: flood detection from SAR imagery '

    parser.add_option("-d", "--save_dir",
                      action="store", dest="save_dir",default='FD_results',
                      help="the folder to save pre-processed results")

    parser.add_option("-n", "--src_nodata",
                      action="store", dest="src_nodata", type=float,
                      help="no data value for input data")

    parser.add_option("-m", "--out_nodata",
                      action="store", dest="out_nodata", type=int, default=128,
                      help="no data value for the output raster")

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
                      action="store_true", dest="verbose",default=False,
                      help="setting this to enable outputting log message and png files of histogram")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)

