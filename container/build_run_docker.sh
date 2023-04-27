#!/bin/bash

############################################################
## build and manage docker images
#build docker image, repository name must be lowercase
docker build -t  sentinel-1-flood-detection .

#lists the images you have locally.
docker images

#remove local image, You can refer to an image by its ID or its name  (-f: force)
docker rmi  -f sentinel-1-pre-processing

############################################################
## run and manage docker containers
# create and run container, --rm : remove the container on exit
docker run --rm -it sentinel-1-flood-detection


# noted: need large CPU memory (>16 GB) to run,
# "docker info | grep Memory" shows the total memory, adjust this if necessary.

# run, mount /data (-v PATH-on-host-machine:PATH-inside-container)
# TESIA
docker run --rm -v /home/lihu9680/Bhaltos2/lingcaoHuang:/data  -it sentinel-1-flood-detection
# my Laptop:
docker run --rm -v ${HOME}/Data:/data -v ${HOME}:/home/user  -it sentinel-1-flood-detection


# tag and push to docker hub
docker tag sentinel-1-flood-detection sentinel-1-flood-detection:v1
docker push sentinel-1-flood-detection:v1


### launch a new terminal to the container, e9ef58868d14 is the container by "nvidia-docker ps" or "nvidia-docker ps -a"
#docker exec -it e9ef58868d14 bash

### start the container at the background
#4cc63f4a50d1 is got by "nvidia-docker ps -q -l"
#docker start e9ef58868d14

### attach to the container
#docker attach e9ef58868d14




