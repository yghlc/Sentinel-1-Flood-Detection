#!/bin/bash

set -eE -o functrace

hostdir=~/Bhaltos2/lingcaoHuang
input=/data/flooding_area/test_docker_containers/s1_pre-processing_FD_input_docker.json
docker_home=/root

# download Sentinel-1 data
docker run --rm -v ${hostdir}:/data -v $HOME:${docker_home}  -it sentinel-1-pre-processing asf_download.py ${input}

# preprocessing of Sentinel-1 data
docker run --rm -v ${hostdir}:/data -v $HOME:${docker_home}  -it sentinel-1-pre-processing snap_GRD_process.py  ${input} 

#flood detection
docker run --rm -v ${hostdir}:/data -v ${HOME}:${docker_home}  -it sentinel-1-flood-detection sar_flood_det.py ${input}

