#!/bin/bash

# it turns out that docker have permission issues (different user inside and outside)
# even I set user name as "hlc" in the docker. When "git pull" inside docker, permission denied.  hlc Jan 10 2019

#build docker image, repository name must be lowercase
docker build -t  sentinel-1-flood-detection .

# create and run container

docker run -it sentinel-1-flood-detection

# --rm : remove the container on exit
# by mount home folder, the container can load environment settings in .bashrc
docker run --rm -v $HOME/:/home/hlc/ -it sentinel-1-flood-detection
#cryo06
docker run --rm -v $HOME/:/home/hlc/ -v /docker:/docker -v /DATA3:/DATA3 -v /DATA4:/DATA4 -it sentinel-1-flood-detection
nvidia-docker run --rm -v $HOME/:/home/hlc/ -v /docker:/docker -v /DATA3:/DATA3 -v /DATA4:/DATA4 -it sentinel-1-flood-detection
#cryo03
nvidia-docker run --rm -v $HOME/:/home/hlc/ -v /500G:/500G -v /DATA1:/DATA1 -it sentinel-1-flood-detection

# tag and push to docker hub
docker tag sentinel-1-flood-detection sentinel-1-flood-detection:v1
docker push sentinel-1-flood-detection:v1


### launch a new terminal to the container, e9ef58868d14 is the container by "nvidia-docker ps" or "nvidia-docker ps -a"
#nvidia-docker exec -it e9ef58868d14 bash

### start the container at the background
#4cc63f4a50d1 is got by "nvidia-docker ps -q -l"
#nvidia-docker start e9ef58868d14

### attach to the container
#nvidia-docker attach e9ef58868d14

### install isce after entering the docker container
/home/hlc/programs/isce_v2.2/temp/isce-2.2.0/setup
#./install.sh -p /home/hlc/programs/isce_v2.2
./install.sh  -v -c /home/hlc/programs/isce_v2.2/SConfigISCE



