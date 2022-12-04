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

import utility

from subprocess import Popen, PIPE, STDOUT
def run_pOpen(cmd_str):
    ps = Popen(cmd_str, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    out, err = ps.communicate()
    returncode = ps.returncode
    if returncode != 0:
        print(out.decode())
        # print(p.stdout.read())
        print(err)
        sys.exit(1)

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

def read_raster_in_polygons_mask(raster_path, polygons, nodata=None, all_touched=True, crop=True,
                                 bands = None, save_path=None):
    # using mask to get pixels in polygons
    # see more information of the parameter in the function: mask

    if isinstance(polygons, list) is False:
        polygon_list = [polygons]
    else:
        polygon_list = polygons

    with rasterio.open(raster_path) as src:
        # crop image and saved to disk
        out_image, out_transform = mask(src, polygon_list, nodata=nodata, all_touched=all_touched, crop=crop,
                                        indexes=bands)

        # print(out_image.shape)
        if out_image.ndim == 2:
            height, width = out_image.shape
            band_count = 1
        else:
            band_count, height, width = out_image.shape
        if nodata is None:  # if it None, copy from the src file
            nodata = src.nodata
        if save_path is not None:
            # save it to disk
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                             "height": height,
                             "width": width,
                             "count": band_count,
                             "transform": out_transform,
                             "nodata": nodata})  # note that, the saved image have a small offset compared to the original ones (~0.5 pixel)
            if out_image.ndim == 2:
                out_image = out_image.reshape((1, height, width))
            with rasterio.open(save_path, "w", **out_meta) as dest:
                dest.write(out_image)

        return out_image, out_transform, nodata


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
def map_to_interval(band, a, b, data_min=None, data_max = None):
    if data_min is not None:
        arr_min = data_min
    else:
        arr_min = np.min(band)
    if data_max is not None:
        arr_max = data_max
    else:
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

def get_image_bound_box(file_path, buffer=None):
    # get the bounding box: (left, bottom, right, top)
    with rasterio.open(file_path) as src:
        # the extent of the raster
        raster_bounds = src.bounds
        if buffer is not None:
            # Create new instance of BoundingBox(left, bottom, right, top)
            new_box_obj = BoundingBox(raster_bounds.left-buffer, raster_bounds.bottom-buffer,
                       raster_bounds.right+buffer, raster_bounds.top+ buffer)
            # print(raster_bounds, new_box_obj)
            return new_box_obj
        return raster_bounds

def get_image_proj_extent(imagepath):
    """
    get the extent of a image
    Args:
        imagepath:image path

    Returns:(ulx:Upper Left X,uly: Upper Left Y,lrx: Lower Right X,lry: Lower Right Y)

    """
    bound = get_image_bound_box(imagepath)
    ulx = bound.left
    uly = bound.top
    lrx = bound.right
    lry = bound.bottom
    return (ulx,uly,lrx,lry)


def subset_image_projwin(output,imagefile,ulx,uly,lrx,lry,resample_m='bilinear',dst_nondata=0,xres=None,yres=None,
                         o_format='GTiff',compress=None, tiled=None, bigtiff=None,thread_num=None):
    #bug fix: the origin (x,y) has a difference between setting one when using gdal_translate to subset image 2016.7.20
    # CommandString = 'gdal_translate  -r bilinear  -eco -projwin ' +' '+str(ulx)+' '+str(uly)+' '+str(lrx)+' '+str(lry)\
    # + ' '+imagefile + ' '+output
    xmin = ulx
    ymin = lry
    xmax = lrx
    ymax = uly

    CommandString = 'gdalwarp -r %s '%resample_m + ' -of ' + o_format
    CommandString += ' -te ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax)

    # if src_nodata != None:
    #     CommandString += ' -srcnodata ' +  str(src_nodata)
    # if dst_nodata != None:
    #     CommandString += ' -dstnodata ' + str(dst_nodata)

    if xres !=None and yres!=None:
        CommandString += ' -tr ' + str(xres) + ' ' +str(yres)

    if compress != None:
        CommandString += ' -co ' + 'compress=%s'%compress       # lzw
    if tiled != None:
        CommandString += ' -co ' + 'TILED=%s'%tiled     # yes
    if bigtiff != None:
        CommandString += ' -co ' + 'bigtiff=%s' % bigtiff  # IF_SAFER

    if thread_num != None:
        CommandString += ' -multi -wo NUM_THREADS=%d '%thread_num

    CommandString += ' '+ imagefile + ' ' + output

    return run_pOpen(CommandString)


def subset_image_baseimage(output_file,input_file,baseimage,same_res=False,resample_m='bilinear'):
    """
    subset a image base on the extent of another image
    Args:
        output_file:the result file
        input_file:the image need to subset
        baseimage:the base image which provide the extend for subset
        same_res: if true, then will resample the output to the resolution of baseimage, otherwise, keep the resolution

    Returns:True is successful, False otherwise

    """
    (ulx,uly,lrx,lry) = get_image_proj_extent(baseimage)
    if ulx is False:
        return False
    # check the save folder is valid or not
    save_dir = os.path.dirname(output_file)

    if same_res:
        xres, yres = get_xres_yres_file(baseimage)
    else:
        xres, yres = get_xres_yres_file(input_file) # the resolution should keep the same

    if subset_image_projwin(output_file,input_file,ulx,uly,lrx,lry,xres=xres,yres=yres,resample_m=resample_m,
                            compress='lzw', tiled='yes', bigtiff='if_safer') is False:
        return False
    return True

def resample_crop_raster(ref_raster, input_raster, output_raster=None, resample_method='near'):

    if output_raster is None:
        name, ext = os.path.splitext(os.path.basename(input_raster))
        output_raster = name + '_res_sub' + ext

    # xres, yres = raster_io.get_xres_yres_file(ref_raster)
    # resample_raster = os.path.basename(input_raster)
    # resample_raster = io_function.get_name_by_adding_tail(resample_raster,'resample')
    #
    # # resample
    # RSImageProcess.resample_image(input_raster,resample_raster,xres,yres,resample_method)
    # if os.path.isfile(resample_raster) is False:
    #     raise ValueError('Resample %s failed'%input_raster)

    # check projection
    prj4_ref =  get_projection(ref_raster,format='proj4')
    prj4_input = get_projection(input_raster,format='proj4')
    if prj4_ref != prj4_input:
        raise ValueError('projection inconsistent: %s and %s'%(ref_raster, input_raster))

    if os.path.isfile(output_raster):
        print('Warning, %s exists'%output_raster)
        return output_raster

    # crop
    subset_image_baseimage(output_raster, input_raster, ref_raster, same_res=True,resample_m=resample_method)
    if os.path.isfile(output_raster):
        return output_raster
    else:
        return False

def get_max_min_histogram_percent_oneband(data, bin_count, min_percent=0.01, max_percent=0.99, nodata=None,
                                          hist_range=None):
    '''
    get the max and min when cut of % top and bottom pixel values
    :param data: one band image data, 2d array.
    :param bin_count: bin_count of calculating the histogram
    :param min_percent: percent
    :param max_percent: percent
    :param nodata:
    :param hist_range: [min, max] for calculating the histogram
    :return: min, max value, histogram (hist, bin_edges)
    '''
    if data.ndim != 2:
        raise ValueError('Only accept 2d array')
    data_1d = data.flatten()
    if nodata is not None:
        data_1d = data_1d[data_1d != nodata] # remove nodata values

    data_1d = data_1d[~np.isnan(data_1d)]   # remove nan value
    hist, bin_edges = np.histogram(data_1d, bins=bin_count, density=False, range=hist_range)

    # get the min and max based on percent cut.
    if min_percent >= max_percent:
        raise ValueError('min_percent >= max_percent')
    found_min = 0
    found_max = 0

    count = hist.size
    sum = np.sum(hist)
    accumulate_sum = 0
    for ii in range(count):
        accumulate_sum += hist[ii]
        if accumulate_sum/sum >= min_percent:
            found_min = bin_edges[ii]
            break

    accumulate_sum = 0
    for ii in range(count-1,0,-1):
        # print(ii)
        accumulate_sum += hist[ii]
        if accumulate_sum / sum >= (1 - max_percent):
            found_max = bin_edges[ii]
            break

    return found_min, found_max, hist, bin_edges

def image_numpy_to_8bit(img_np, max_value, min_value, src_nodata=None, dst_nodata=None):
    '''
    convert float or 16 bit to 8bit,
    Args:
        img_np:  numpy array
        max_value:
        min_value:
        src_nodata:
        dst_nodata:  if output nodata is 0, then covert data to 1-255, if it's 255, then to 0-254

    Returns: new numpy array

    '''
    print('Convert to 8bit, original max, min: %.4f, %.4f'%(max_value, min_value))
    nan_loc = np.where(np.isnan(img_np))
    if nan_loc[0].size > 0:
        img_np = np.nan_to_num(img_np)

    nodata_loc = None
    if src_nodata is not None:
        nodata_loc = np.where(img_np==src_nodata)

    img_np[img_np > max_value] = max_value
    img_np[img_np < min_value] = min_value

    if dst_nodata == 0:
        n_max, n_min = 255, 1
    elif dst_nodata == 255:
        n_max, n_min = 254, 0
    else:
        n_max, n_min = 255, 0

    # scale the grey values to 0 - 255 for better display
    k = (n_max - n_min)*1.0/(max_value - min_value)
    new_img_np = (img_np - min_value) * k + n_min
    new_img_np = new_img_np.astype(np.uint8)

    # replace nan data as nodata
    if nan_loc[0].size > 0:
        if dst_nodata is not None:
            new_img_np[nan_loc] = dst_nodata
        else:
            new_img_np[nan_loc] = n_min
    # replace nodata
    if nodata_loc is not None and nodata_loc[0].size >0:
        if dst_nodata is not None:
            new_img_np[nodata_loc] = dst_nodata
        else:
            new_img_np[nodata_loc] = src_nodata

    return new_img_np

def image_numpy_allBands_to_8bit_hist(img_np_allbands, min_max_values=None, per_min=0.01, per_max=0.99, bin_count = 10000, src_nodata=None, dst_nodata=None):

    input_ndim = img_np_allbands.ndim
    if input_ndim == 3:
        band_count, height, width = img_np_allbands.shape
    else:
        # add one dimension
        band_count = 1
        img_np_allbands = np.expand_dims(img_np_allbands, axis=0)

    if min_max_values is not None:
        # if we input multiple scales, it should has the same size the band count
        if len(min_max_values) > 1 and len(min_max_values) != band_count:
            raise ValueError('The number of min_max_value is not the same with band account')
        # if only input one scale, then duplicate for multiple band account.
        if len(min_max_values) == 1 and len(min_max_values) != band_count:
            min_max_values = min_max_values * band_count

    # get min, max
    new_img_np = np.zeros_like(img_np_allbands, dtype=np.uint8)
    for band, img_oneband in enumerate(img_np_allbands):
        found_min, found_max, hist, bin_edges = get_max_min_histogram_percent_oneband(img_oneband, bin_count,
                                                                                                min_percent=per_min,
                                                                                                max_percent=per_max,
                                                                                                nodata=src_nodata)
        print('min and max value from histogram (percent cut):', found_min, found_max)
        if min_max_values is not None:
            if found_min < min_max_values[band][0]:
                found_min = min_max_values[band][0]
                print('reset the min value to %s' % found_min)
            if found_max > min_max_values[band][1]:
                found_max = min_max_values[band][1]
                print('reset the max value to %s' % found_max)
        if found_min == found_max:
            print('warning, found_min == find_max, set the output as nodata or found_min')
            new_img_np[band, :] = dst_nodata if dst_nodata is not None else found_min
        else:
            new_img_np[band,:] = image_numpy_to_8bit(img_oneband, found_max, found_min, src_nodata=src_nodata, dst_nodata=dst_nodata)

    if input_ndim == 3:
        return new_img_np
    else:
        # remove the add dimension
        return np.squeeze(new_img_np)

def unset_nodata(raster_path):
    # remove nodato (it was copy from the input image)
    command_str = 'gdal_edit.py -unsetnodata ' + raster_path
    utility.os_system_exit_code(command_str)

def set_water_color_map(raster_path):
    with rasterio.open(raster_path, 'r+') as dst:
        color_map_dict = {0: (230, 230, 230, 255),
                              1: (31, 120, 180, 255),  # light blue for water
                              128: (255, 255, 255, 255),  # nodata
                              255: (31, 120, 180, 255)}  # light blue for water, in some file, 255 is water
        dst.write_colormap(1, color_map_dict)

    return True

def image_read_pre_process(image_path, src_nodata=None, b_normalized=False):
    '''
    return a max and min value for normalization (0-1)
    :param image_path: image path
    :param tile_width:
    :param tile_height:
    :param b_normalized: if True, will be normalized to 0 -1
    :return:
    '''
    data, nodata = read_raster_one_band_np(image_path)
    # if nodata is not set
    if nodata is None:
        nodata = src_nodata

    if nodata is not None:
        data[data==nodata] = np.nan   # set nodata as nan
    else:
        print('Warning, nodata value is not set')

    # if set lower_quantile=0.01, it makes "min_cnt = np.sum(s_s_array == np.min(s_s_array)); min_cnt < 100" failed in "otsu_and_lm_for_a_array"
    data = quantile_clip(data,  upper_quantile=0.99)
    data = threshold_clip(data,lower=0.0)
    min_value = np.nanmin(data)
    max_value = np.nanmax(data)
    if b_normalized:
        data = map_to_interval(data, 0, 1, data_min=min_value, data_max=max_value)
        # calculate the min, max value again
        min_value = np.nanmin(data)
        max_value = np.nanmax(data)

    mean_value = np.nanmean(data)
    medium_value = np.nanmedian(data)

    return data, min_value,max_value, mean_value,medium_value

def permant_water_pixles(sar_image_2d, sar_grd_path,water_mask_file,save_dir):
    # locate pixels for permanent water
    # statistic the sigma0 value of these pixels

    name, ext = os.path.splitext(os.path.basename(sar_grd_path))
    mask_save_path = os.path.join(save_dir,name + '_PerWaterMask'+ext)
    info_json_path = os.path.join(save_dir,name + '_PerWaterMask.json')

    surface_water_crop = resample_crop_raster(sar_grd_path, water_mask_file, output_raster=mask_save_path, resample_method='near')
    if surface_water_crop is False:
        return False

    data, nodata = read_raster_one_band_np(surface_water_crop)

    if data.shape != sar_image_2d.shape:
        raise ValueError('the size of water mask (%s) and the SAR (%s) is different'%(str(data.shape), str(sar_image_2d.shape)))

    # in the surface water, 1 is water, 0 are other, 255 are ocean
    per_water_loc = np.where(data==1)
    per_nonland_loc = np.where(np.logical_or(data==1, data==255))

    # statistics on SAR images
    array_per_water = sar_image_2d[per_water_loc]
    # print(array_per_water.shape)
    array_per_water = array_per_water[~np.isnan(array_per_water) ] # remove nan
    # print(array_per_water.shape)

    pixel_count = array_per_water.size
    min = np.min(array_per_water)
    max = np.max(array_per_water)
    mean = np.mean(array_per_water)
    median = np.median(array_per_water)
    std = np.std(array_per_water)

    # save to json
    water_dict = {'sar_image_dir':os.path.dirname(sar_grd_path),
                  'sar_image_file':os.path.basename(sar_grd_path),
                  'Land_PerWater_PixelCount':pixel_count,
                  'sar_value_min_onPerWater':float(min),
                  'sar_value_max_onPerWater':float(max),
                  'sar_value_mean_onPerWater':float(mean),
                  'sar_value_median_onPerWater':float(median),
                  'sar_value_std_onPerWater':float(std)}

    utility.save_dict_to_txt_json(info_json_path,water_dict)

    return per_nonland_loc, pixel_count, min, max, mean, median, std, mask_save_path

def get_elevation_raster(sar_grd_path,dem_file,save_dir):
    utility.is_file_exist(dem_file)
    name, ext = os.path.splitext(os.path.basename(sar_grd_path))
    save_path = os.path.join(save_dir,name + '_DEM'+ext)
    dem_resample_crop = resample_crop_raster(sar_grd_path, dem_file, output_raster=save_path, resample_method='near')
    if dem_resample_crop is False:
        return None
    return save_path


def main():
    pass


if __name__=='__main__':
    main()


