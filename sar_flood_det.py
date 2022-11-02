#!/usr/bin/env python
# Filename: sar_flood_det.py 
"""
introduction: detection of flood from SAR imagery

add time: 02 November, 2022
"""
import os,sys
from optparse import OptionParser
import glob
from datetime import datetime
import time

from SAR_Flood_Detection_v02 import Run_amplitude_algorithm

def get_sar_file_list(file_or_dir):
    if os.path.isdir(file_or_dir):
        sar_Sigma_files = glob.glob(os.path.join(file_or_dir, '*Sigma0_VV.tif'))  # Process VV files
        if len(sar_Sigma_files) == 0:  ## Process VH files, if VV is empty
            sar_Sigma_files = glob.glob(os.path.join(file_or_dir, '*Sigma0_VH.tif'))
    else:
        with open(file_or_dir,'r') as f_obj:
            sar_Sigma_files = [line.strip() for line in f_obj.readlines()]
    if len(sar_Sigma_files) == 0:
        raise ValueError("No SAR Sigma0 in %s"%file_or_dir)
    return sar_Sigma_files


def Run_amplitude_algorithm_v2(sar_image_list, save_dir,water_mask_file=None):
    t0 = time.time()
    total_count = len(sar_image_list)
    for idx, grd in enumerate(sar_image_list):
        t1 = time.time()
        print(datetime.now(), 'Processing the SAR image %s / %s' % (idx + 1, total_count))
        # to be completed

        print(datetime.now(), 'Complete, took %s seconds' % (time.time() - t1))

    total_time = time.time() - t0
    print(datetime.now(), 'Flood detection complete, took %s seconds' % (total_time))



def main(options, args):

    sar_image_list = get_sar_file_list(args[0])
    save_dir = options.save_dir
    water_mask = options.water_mask
    verbose = options.verbose

    print(datetime.now(), 'Found %d SAR Sigma0 images from %s:'%(len(sar_image_list),args[0]))
    print(datetime.now(), 'Will save flood detection results to %s'%save_dir)


    # Run_amplitude_algorithm_v2(sar_image_list, save_dir,water_mask_file=water_mask)
    Run_amplitude_algorithm(sar_image_list, save_dir,os.path.dirname(water_mask),os.path.basename(water_mask),verbose=verbose)



if __name__ == "__main__":
    usage = "usage: %prog [options] sar_files.txt or image_directory "
    parser = OptionParser(usage=usage, version="1.0 2022-11-02")
    parser.description = 'Introduction: flood detection from SAR imagery '

    parser.add_option("-d", "--save_dir",
                      action="store", dest="save_dir",default='asf_data',
                      help="the folder to save pre-processed results")


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

