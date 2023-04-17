#!/bin/bash

fd_py=~/codes/PycharmProjects/yghlc_Sentinel-1-Flood-Detection/flood_depth.py

#fd_dir=depth_results
#dem=~/Bhaltos2/lingcaoHuang/flooding_area/DEM/Houston/Houston_SRTM.tif

#fd_dir=depth_results_3DEP
#dem=~/Bhaltos2/lingcaoHuang/flooding_area/DEM/Houston/dem_3DEP/Houston_3DEP_3m_epsg4326.tif

fd_dir=depth_results_3DEP_10m
dem=~/Bhaltos2/lingcaoHuang/flooding_area/DEM/Houston/dem_3DEP_10m/Houston_3DEP_10m_epsg4326.tif

proc_num=4
#proc_num=1

for tif in FD_results_thresholding/*3220*VH*.tif FD_results_thresholding/*D734*VH*.tif; do 
	echo $tif
	${fd_py} ${tif} ${dem} -d ${fd_dir} --process_num=${proc_num}  --verbose
	#exit
done

