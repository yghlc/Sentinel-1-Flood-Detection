FROM ubuntu:22.04
MAINTAINER Lingcao Huang (lingcao.huang@colorado.edu)

# Install base utilities
RUN apt-get update && apt-get install -y  \
        build-essential \
        pkg-config \
        zip \
        unzip \
        git \
	    wget \
	    &&\
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/sh ~/miniconda.sh -b -p /opt/miniconda3  && rm ~/miniconda.sh

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# install python package
RUN conda install -c conda-forge -y geopandas
# this will also install GDAL
RUN conda install -c conda-forge -y rasterio
RUN conda install -c conda-forge -y scikit-image
RUN conda install -c conda-forge -y simplekml
RUN conda install -c conda-forge -y asf_search

RUN pip install opencv-python
RUN pip install elevation

# t: tarball, i index-cache p: package  y: Do not ask for confirmation
RUN conda clean -tipy

# install SNAP
RUN wget --quiet https://download.esa.int/step/snap/9.0/installers/esa-snap_all_unix_9_0_0.sh -O ~/esa-snap.sh && \
    /bin/sh ~/esa-snap.sh -q -dir /opt/snap  && rm ~/esa-snap.sh

# update SNAP when a container launch?
#RUN /opt/snap/bin/snap --nosplash --nogui --modules --update-all

# clone github repo
RUN git clone https://github.com/yghlc/Sentinel-1-Pre-Processing.git /opt/Sentinel-1-Pre-Processing  && \
    git clone https://github.com/yghlc/Sentinel-1-Flood-Detection.git /opt/Sentinel-1-Flood-Detection

