#!/bin/bash

set -eE -o functrace

hostdir=/home/lihu9680/Bhaltos2/lingcaoHuang
input=/data/flooding_area/test_docker_containers/s1_pre-processing_FD_input_docker.json
docker_home=/root

# download 
docker run --rm -v ${hostdir}:/data -v $HOME:${docker_home}  -it sentinel-1-pre-processing asf_download.py ${input}

# preprocessing
docker run --rm -v ${hostdir}:/data -v $HOME:${docker_home}  -it sentinel-1-pre-processing snap_GRD_process.py  ${input} 

#flood detection
docker run --rm -v ${hostdir}:/data -v ${HOME}:${docker_home}  -it sentinel-1-flood-detection sar_flood_det.py ${input}

