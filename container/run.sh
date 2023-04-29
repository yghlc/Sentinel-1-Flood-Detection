#!/bin/bash

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

#save_dir=/data/flooding_area/test_docker_containers/Houston
#water_mask=/data/global_surface_water/extent_epsg4326_theUS/surface_water_theUS_3_2020.tif
#sar_img_dir=${save_dir}/pre-processed/final
#fd_dir=${save_dir}/FD_results_thresholding
#
#sar_flood_det.py ${sar_img_dir} --src_nodata=0.0 -w ${water_mask}  -d ${fd_dir}  --process_num=1

## flood detection
sar_flood_det.py  s1_pre-processing_FD_input.json