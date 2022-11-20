#!/usr/bin/env python
# Filename: raster_tools.py 
"""
introduction: Based on rasterio, to read and write raster data;  some simple calculation

ref: https://github.com/yghlc/DeeplabforRS/blob/master/raster_io.py

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 14 November, 2022
"""

import os, sys
import rasterio
import numpy as np
import math

from rasterio.coords import BoundingBox
from rasterio.mask import mask
from rasterio.features import rasterize
from rasterio.features import shapes

import skimage.measure
import time
#Color interpretation https://rasterio.readthedocs.io/en/latest/topics/color.html
from rasterio.enums import ColorInterp


def open_raster_read(raster_path):
    src = rasterio.open(raster_path)
    return src

def get_width_heigth_bandnum(opened_src):
    return opened_src.height,  opened_src.width,  opened_src.count

# def get_xres_yres(opened_src):
#     return opened_src.height,  opened_src.width,  opened_src.count

def get_driver_format(file_path):
    with rasterio.open(file_path) as src:
        return src.driver

def get_projection(file_path, format=None):
    # https://rasterio.readthedocs.io/en/latest/api/rasterio.crs.html
    # convert the different type, to epsg, proj4, and wkt
    with rasterio.open(file_path) as src:
        if format is not None:
            if format == 'proj4':
                return src.crs.to_proj4() # string like '+init=epsg:32608', differnt from GDAL output
            elif format == 'wkt':
                return src.crs.to_wkt()     # string,  # its OGC WKT representation
            elif format == 'epsg':
                return src.crs.to_epsg()    # to epsg code, iint
            else:
                raise ValueError('Unknown format: %s'%str(format))
        return src.crs

def get_xres_yres_file(file_path):
    with rasterio.open(file_path) as src:
        xres, yres  = src.res       # Returns the (width, height) of pixels in the units of its coordinate reference system.
        return xres, yres

def get_height_width_bandnum_dtype(file_path):
    with rasterio.open(file_path) as src:
        return src.height, src.width, src.count, src.dtypes[0]

def get_transform_from_file(file_path):
    with rasterio.open(file_path) as src:
        return src.transform

def get_nodata(file_path):
    with rasterio.open(file_path) as src:
        return src.nodata

def boundary_to_window(boundary):
    # boundary: (xoff,yoff ,xsize, ysize)
    # window structure; expecting ((row_start, row_stop), (col_start, col_stop))
    window = ((boundary[1],boundary[1]+boundary[3])  ,  (boundary[0],boundary[0]+boundary[2]))
    return window

def copy_one_patch_image_data_2d(patch, entire_img_data):
    #(xoff,yoff ,xsize, ysize)
    row_s = patch[1]
    row_e = patch[1] + patch[3]
    col_s = patch[0]
    col_e = patch[0] + patch[2]
    # entire_img_data is in opencv format:  height, width
    patch_data = entire_img_data[row_s:row_e, col_s:col_e]
    patch_data = np.nan_to_num(patch_data)  # nan to 0
    return patch_data

def read_raster_all_bands_np(raster_path, boundary=None):
    # boundary: (xoff,yoff ,xsize, ysize)

    with rasterio.open(raster_path) as src:
        indexes = src.indexes

        if boundary is not None:
            data = src.read(indexes, window=boundary_to_window(boundary))
        else:
            data = src.read(indexes)  # output (band_count, height, width)

        # print(data.shape)
        # print(src.nodata)
        # if src.nodata is not None and src.dtypes[0] == 'float32':
        #     data[ data == src.nodata ] = np.nan

        return data, src.nodata


def read_raster_one_band_np(raster_path, band=1, boundary=None):
    # boundary: (xoff,yoff ,xsize, ysize)
    with rasterio.open(raster_path) as src:

        if boundary is not None:
            data = src.read(band, window=boundary_to_window(boundary))
        else:
            data = src.read(band)  # output (height, width)

        # if src.nodata is not None and src.dtypes[0] == 'float32':
        #     data[ data == src.nodata ] = np.nan
        return data, src.nodata


def save_numpy_array_to_rasterfile(numpy_array, save_path, ref_raster, format='GTiff', nodata=None,
                                   compress=None, tiled=None, bigtiff=None, boundary=None, verbose=True):
    '''
    save a numpy to file, the numpy has the same projection and extent with ref_raster
    Args:
        numpy_array:
        save_path:
        ref_raster:
        format:
        boundary: (xoff,yoff ,xsize, ysize)

    Returns:

    '''
    if numpy_array.ndim == 2:
        band_count = 1
        height, width = numpy_array.shape
        # reshape to 3 dim, to write the disk
        numpy_array = numpy_array.reshape(band_count, height, width)
    elif numpy_array.ndim == 3:
        band_count, height, width = numpy_array.shape
    else:
        raise ValueError('only accept ndim is 2 or 3')

    dt = np.dtype(numpy_array.dtype)

    if verbose:
        print('dtype:', dt.name)
        print(numpy_array.dtype)
        print('band_count,height,width', band_count, height, width)
        # print('saved numpy_array.shape',numpy_array.shape)

    with rasterio.open(ref_raster) as src:
        # [print(src.colorinterp[idx]) for idx in range(src.count)]
        # test: save it to disk
        out_meta = src.meta.copy()
        out_meta.update({"driver": format,
                         "height": height,
                         "width": width,
                         "count": band_count,
                         "dtype": dt.name
                         })
        if nodata is not None:
            out_meta.update({"nodata": nodata})

        if compress is not None:
            out_meta.update(compress=compress)
        if tiled is not None:
            out_meta.update(tiled=tiled)
        if bigtiff is not None:
            out_meta.update(bigtiff=bigtiff)

        if boundary is not None:
            if boundary[2] != width or boundary[3] != height:
                raise ValueError(
                    'boundary (%s) is not consistent with width (%d) and height (%d)' % (str(boundary), width, height))
            window = boundary_to_window(boundary)
            new_transform = src.window_transform(window)
            out_meta.update(transform=new_transform)

        colorinterp = [src.colorinterp[idx] for idx in range(src.count)]
        # print(colorinterp)

        with rasterio.open(save_path, "w", **out_meta) as dest:
            dest.write(numpy_array)
            # Get/set raster band color interpretation: https://github.com/mapbox/rasterio/issues/100
            if src.count == band_count:
                dest.colorinterp = colorinterp
            else:
                dest.colorinterp = [ColorInterp.undefined] * band_count

    if verbose:
        print('save to %s' % save_path)

    return True


#ref: https://github.com/yghlc/DeeplabforRS/blob/master/split_image.py
def sliding_window(image_width,image_height, patch_w,patch_h,adj_overlay_x=0,adj_overlay_y=0):
    """
    get the subset windows of each patch
    Args:
        image_width: width of input image
        image_height: height of input image
        patch_w: the width of the expected patch
        patch_h: the height of the expected patch
        adj_overlay_x: the extended distance (in pixel of x direction) to adjacent patch, make each patch has overlay with adjacent patch
        adj_overlay_y: the extended distance (in pixel of y direction) to adjacent patch, make each patch has overlay with ad
    Returns: The list of boundary of each patch

    """

    count_x = int(image_width/patch_w)
    count_y = int(image_height/patch_h)

    leftW = int(image_width)%int(patch_w)
    leftH = int(image_height)%int(patch_h)
    if leftW < patch_w/3 and count_x > 0:
        # count_x = count_x - 1
        leftW = patch_w + leftW
    else:
        count_x = count_x + 1
    if leftH < patch_h/3 and count_y > 0:
        # count_y = count_y - 1
        leftH = patch_h + leftH
    else:
        count_y = count_y + 1


    patch_boundary = []
    for i in range(0,count_x):
        # f_obj.write('column %d:'%i)
        for j in range(0,count_y):
            w = patch_w
            h = patch_h
            if i==count_x -1:
                w = leftW
            if j == count_y - 1:
                h = leftH

            # extend the patch
            xoff = max(i*patch_w - adj_overlay_x,0)  # i*patch_w
            yoff = max(j*patch_h - adj_overlay_y, 0) # j*patch_h
            xsize = min(i*patch_w + w + adj_overlay_x,image_width) - xoff   #w
            ysize = min(j*patch_h + h + adj_overlay_y, image_height) - yoff #h

            new_patch = (xoff,yoff ,xsize, ysize)
            patch_boundary.append(new_patch)

    # remove duplicated patches
    patch_boundary_unique = set(patch_boundary)
    if len(patch_boundary_unique) != len(patch_boundary):
        patch_boundary = patch_boundary_unique

    return patch_boundary


# clips the data to lower and upper quantiles
def quantile_clip(band, lower_quantile=None, upper_quantile=None):
    if lower_quantile != None:
        lower_q = np.nanquantile(band, lower_quantile)
        new_band = np.where(band < lower_q, lower_q, band)
        band = new_band
    if upper_quantile != None:
        upper_q = np.nanquantile(band, upper_quantile)
        new_band = np.where(band > upper_q, upper_q, band)
        band = new_band

    return band


# maps the values in an image to a different interval defined by [a, b]
def map_to_interval(band, a, b):
    arr_min = np.min(band)
    arr_max = np.max(band)
    frac = (b-a) / (arr_max-arr_min)
    new_band = a + frac*(band - arr_min)
    band = new_band
    return band


def threshold_clip(band, lower=None, upper=None):
    if lower != None:
        new_band = np.where(band < lower, lower, band)
        band = new_band
    if upper != None:
        new_band = np.where(band > upper, upper, band)
        band = new_band
    return band

def main():
    pass


if __name__=='__main__':
    main()

