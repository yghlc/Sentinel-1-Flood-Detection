#!/bin/bash

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

# please change this accordingly
scriptDir=~/codes/PycharmProjects
save_dir=${PWD}
# the resolution that pre-processing and flood detection will work on
res=10
# change these for different regions and time
ext_shp=extent/houston_valid_image_extent.shp
s_date=2017-08-29
e_date=2017-08-30

# temporal directory for saving intermediate data output by SNAP
# for better IO performance in different system
tmpDir=/tmp

# The pre-processing need SNAP and GDAL, their path should be saved to "env_setting.json"
# please place env_setting.json in the working directory.

#####################################################################################
# download Sentinel-1 data
down_py=${scriptDir}/yghlc_Sentinel-1-Pre-Processing/asf_download.py

## if username and password are set in ~./netrc, the python script can automatically read them
#username=$(awk '/urs.earthdata.nasa.gov/{getline; print $2}' ~/.netrc)
#password=$(awk '/urs.earthdata.nasa.gov/{getline; getline; print $2}' ~/.netrc)
${down_py} ${ext_shp} -d ${save_dir}/sentinel-1  -s ${s_date} -e ${e_date} #-u ${username} -p ${password}

# Alternatively, save granule id of SAR images into a text file, then download them
#file_ids=houston_20170830_ids.txt
#${down_py} ${file_ids} -d ${save_dir}/sentinel-1

#####################################################################################
# pre-processing using SNAP
# Apply Orbit File, Remove GRD Border Noise, Calibration, Speckle Filter, and Terrain Correction
rtc_py=${scriptDir}/yghlc_Sentinel-1-Pre-Processing/snap_GRD_process.py
outdir=${save_dir}/pre-processed
${rtc_py} ${save_dir}/sentinel-1  -d ${outdir} -t ${tmpDir}  --save_pixel_size=${res}

#####################################################################################
# run flood detection
fd_py=${scriptDir}/yghlc_Sentinel-1-Flood-Detection/sar_flood_det.py
fd_dir=${save_dir}/FD_results_thresholding
water_mask=~/Data/global_surface_water/extent_epsg4326_theUS/surface_water_theUS_3_2020.tif
${fd_py} ${save_dir}/pre-processed/final --src_nodata=0.0 -w ${water_mask} -d ${fd_dir}  --process_num=8 #--verbose

## Alternatively, save selected SAR images into a text file, then run flood detection on selected ones.
#img_list=houston_201708_Sigma0.txt
#${fd_py} ${img_list} --src_nodata=0.0 -w ${water_mask} -d ${fd_dir}  --process_num=8 #--verbose



