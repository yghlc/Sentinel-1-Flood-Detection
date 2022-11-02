#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : {Ryan Cassotto, Conor Simmons }
# Edited By   : {Clayton Brengman}
# Created Date: 2022/10/24
# version ='2.0'
# Version 2 uses fixed window and sub-windows. It also incorporates permanent water mask.
# ---------------------------------------------------------------------------
""" Sentinel 1 Data Flood Detection Routine"""  
# ---------------------------------------------------------------------------
from image_proc_module_v01 import Image_proc
from BimodalThreshold_module_v01 import BimodalThreshold
import rasterio
import glob
import numpy as np
import os,sys
import argparse,ast
from subprocess import Popen, PIPE, STDOUT

import shutil

#gdal_dir='/home/rcassotto/anaconda3/bin/'
gdal_dir='/usr/local/bin/'
NoDataValue = 128

def update_env_setting():
    gdal_info = shutil.which("gdalinfo")
    global gdal_dir
    gdal_dir = os.path.dirname(gdal_info)  + '/'
    print('update gdal_dir to %s' %gdal_dir)
# when import or run this model, will call update_env_setting
update_env_setting()


def run_pOpen(cmd_str):
    ps = Popen(cmd_str, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    out, err = ps.communicate()
    returncode = ps.returncode
    if returncode != 0:
        print(out.decode())
        # print(p.stdout.read())
        print(err)
        sys.exit(1)




# ---------------------------------------------------------------------------
# making the output directory
def mk_outdirectory(outpath_full):
    if not os.path.exists(outpath_full):
        print("Making output directory: ", outpath_full)
        os.makedirs(outpath_full)
    return

# ---------------------------------------------------------------------------
## Get image georeference information (e.g. lat, lon, pixel size, etc)
def get_image_geo_info(in_sigma_file):
        image_data = rasterio.open(in_sigma_file)
        lon = (image_data.bounds.left, image_data.bounds.right)
        lat = (image_data.bounds.top, image_data.bounds.bottom)
        minLat = np.min(lat)
        maxLat = np.max(lat)
        minLon = np.min(lon)
        maxLon = np.max(lon)
        medianLat = np.median(lat)
 
        gt = image_data.transform
        pixel_size_lon_deg = gt[0]
        pixel_size_lat_deg = gt[4]
        pixel_size_lat_m = gt[4]*111e3
        pixel_size_lon_m = gt[0]*111e3*np.cos(np.radians(medianLat))
        lon_arr = np.arange(minLon+0.5*pixel_size_lon_deg, maxLon, pixel_size_lon_deg, dtype=float)# lon array with 0.5 chip offset
        lat_arr = np.arange(maxLat+0.5*pixel_size_lat_deg, minLat, pixel_size_lat_deg, dtype=float)
        
        return lat_arr,lon_arr,pixel_size_lat_m,pixel_size_lon_m,pixel_size_lat_deg,pixel_size_lon_deg,image_data, minLon, maxLon, minLat, maxLat


# ---------------------------------------------------------------------------
# save metadata
def save_metadata(granule, infile_path, image_data, window, sub_window, otsus, lms, n_fraction_denom, out_dir):

        Line1 = ( "Input Image: " + granule + "\n")
        Line2 = ( "Input Image Path: " + infile_path + "\n")
        Line3 = ( "Image Height: " + str(image_data.height) + "\n")
        Line4 = ( "Image Width: " + str(image_data.width) + "\n")
        Line5 = ( "Final Block Size: " + str(window) + "\n")
        Line6 = ( "Final S-array Size: " + str(sub_window) + "\n")
        Line7 = ( "Mean OTSU value: " + str(np.mean(otsus)) + "\n")
        Line8 = ( "Mean LM value: " + str(np.mean(lms)) + "\n")
        Line9 = ( "OTSU Values: " + str(otsus) + "\n")
        Line10 = ( "LM Values: " + str(lms) + "\n")
        Line11 = ( "S-array denominator: " + str(n_fraction_denom) + "\n")
        Line13 = ( "Output Path: " + out_dir + "\n")
        if "Sigma0_VV" in granule:
            Line12 = ( "Output Image: " + granule.replace('Sigma0_VV.tif','FD_Results.tif') + "\n")
        else:
            Line12 = ( "Output Image: " + granule.replace('Sigma0_VH.tif','FD_Results.tif') + "\n")

            
        metafile_outname = ( out_dir +'/' + granule + '_FD_Results_meta.txt' )      
        meta_file = open (metafile_outname, "w")
        meta_file.write(Line1)
        meta_file.write(Line2)
        meta_file.write(Line3)
        meta_file.write(Line4)
        meta_file.write(Line5)
        meta_file.write(Line6)
        meta_file.write(Line7)
        meta_file.write(Line8)
        meta_file.write(Line9)
        meta_file.write(Line10)
        meta_file.write(Line11)
        meta_file.write(Line12)
        meta_file.write(Line13)
        meta_file.close()
        return
    
    
# ---------------------------------------------------------------------------
# Write out geotiffs
def write_geotiff(out_dir, image_data, granule, outmap, map_type,nodata=None,compress=None,b_colormap=False):
    if "Sigma0_VV" in granule:
        tiff_outname = os.path.join(out_dir,granule.replace('_Sigma0_VV.tif','_FD_Results_' + map_type + '.tif'))  
    else:
        tiff_outname = os.path.join(out_dir,granule.replace('_Sigma0_VH.tif','_FD_Results_' + map_type + '.tif'))         
    profile = image_data.profile.copy()  # copy geotiff meta data from input file   
    print('Saving results to ', tiff_outname)
    dt = np.dtype(outmap.dtype)
    # update meta
    profile.update({"dtype": dt.name})
    if nodata is not None:
        profile.update({"nodata": nodata})
    if compress is not None:
        profile.update(compress=compress)

    with rasterio.open(tiff_outname, 'w', **profile) as dst:
        dst.write(outmap,1)
        if b_colormap:
            color_map_dict = {0: (230, 230, 230, 255),
                              1: (31, 120, 180, 255),  # light blue for water
                              128: (255, 255, 255, 255),  # nodata
                              255: (31, 120, 180, 255)}  # light blue for water, in some file, 255 is water
            dst.write_colormap(1, color_map_dict)

    return tiff_outname

def create_water_masks(surface_water_dir, surface_water_fname, pixel_size_lon_deg, pixel_size_lat_deg, minLon, maxLon, minLat, maxLat):

    #### Read in water extents
    gswp_path_and_filename = os.path.join(surface_water_dir, surface_water_fname)
#    surface_water_in = rasterio.open(gswp_path_and_filename) # opens geotiff with rasterio; saves it as numpy array with M x N shape
#    surface_water = surface_water_in.read(1) # Read band 1

    ### Create a water mask with gdal - convert colortable to a geotiff using gdal_calc
    tmp_mask_path_and_filename = os.path.join(surface_water_dir,'tmp.tif') 
    print('Generating permanent water mask: ', tmp_mask_path_and_filename)
    cmd_gen_mask = gdal_dir + 'gdal_calc.py -A ' + gswp_path_and_filename + ' --outfile=' + tmp_mask_path_and_filename + ' --calc="logical_and(A!=1, A!=255)"'
    print(cmd_gen_mask); print(' ')
    run_pOpen(cmd_gen_mask)

     
    ##### Crop and resize mask to extents and pixel size of input image
    mask_outfilename = gswp_path_and_filename.replace('.tif', '_10m_WaterMask.tif') 
    cmd_resize_crop_mask = gdal_dir + 'gdal_translate -tr ' + str(pixel_size_lon_deg) + ' ' + str(pixel_size_lat_deg) + ' -r bilinear -projwin ' + str(minLon) + ' ' + str(maxLat) + ' ' + str(maxLon) + ' ' + str(minLat) + ' ' + tmp_mask_path_and_filename + ' ' + mask_outfilename
    print(cmd_resize_crop_mask); print(' ')
    run_pOpen(cmd_resize_crop_mask)
    os.remove(tmp_mask_path_and_filename)
    return mask_outfilename


def apply_water_body_mask(results_map_geotiff, mask_outfilename):
    #### gdal_calc to apply the mask
    infile_masked_outfilename = results_map_geotiff.replace('.tif','_PWMasked.tif')
    cmd_apply_mask = gdal_dir + 'gdal_calc.py -A ' + results_map_geotiff + ' -B ' + mask_outfilename + ' --outfile=' + infile_masked_outfilename + ' --calc="A*B" --NoDataValue=%d'%NoDataValue
    print(cmd_apply_mask); print(' ')
    run_pOpen(cmd_apply_mask)
    cmd_compress = gdal_dir + 'gdal_translate -co "compress=lzw" %s tmp.tif'%(infile_masked_outfilename)
    run_pOpen(cmd_compress)
    os.remove(infile_masked_outfilename)
    shutil.move('tmp.tif', infile_masked_outfilename)

    return



def Run_amplitude_algorithm(Sigma_files, out_dir, surface_water_dir, surface_water_fname,verbose=True):
    
    
    for in_sigma_file in Sigma_files:
        # ---------------------------------------------------------------------------
        ## construct output directory name, make output directory     
        granule = in_sigma_file.split('/')[-1] # granule is a string with the original Sigma0 filename (e.g. S1A_IW_GRDH_1SDV_20150921T232856_20150921T232918_007820_00AE3B_E36F_Sigma0_VV)
        head_tail=os.path.split(in_sigma_file)   # partition input path from filename      
        infile_path = head_tail[0]
        mk_outdirectory(out_dir)  # make output directory#
         
        
        # ---------------------------------------------------------------------------
        ## Read in image
        VV_image = Image_proc(in_sigma_file)
        VV_image.read()
        inan = np.where(VV_image.band == 0) # index no data values in Sigma0 images
        

#        # ---------------------------------------------------------------------------
#        ## Get image corner coordinates
        (lat_arr,lon_arr,pixel_size_lat_m,pixel_size_lon_m,pixel_size_lat_deg,pixel_size_lon_deg,image_data, minLon, maxLon, minLat, maxLat) = get_image_geo_info(in_sigma_file)
        geographic_bounds = [minLon, maxLon, minLat, maxLat]

        # ---------------------------------------------------------------------------
        # Run bimodal threshold
        bt = BimodalThreshold(geographic_bounds)
        
        VV_image.quantile_clip(upper_quantile=0.99)
        VV_image.threshold_clip(lower=0.0)
        VV_image.map_to_interval(0,1)
        
        # ---------------------------------------------------------------------------
        ### This approach starts with S x S as 1/20th min image dimension size then increases if no value is found
        n_fraction_denom = 20 # denominator for the fraction to determine s-array size. S-array size = min image dimension * n_fraction_numer / n_fraction_denom
        otsus = [] # initialize ostsus list

        
        ## Mod 10/18/2022
        window=1456*3
        sub_window=182*3
        otsus, lms = bt.otsu_and_lm([VV_image], out_dir, ptf=False, block_dim=window, s=sub_window, verbose=verbose) ## Setting Verbose = True creates diagnostic images of subpanel distributions

        
        # ---------------------------------------------------------------------------
        ## Save Meta data file
        save_metadata(granule, infile_path, image_data, window, sub_window, otsus, lms, n_fraction_denom, out_dir)
        print("OTSUS: ", otsus)  # this prints otsus results for each subtile - the lower threshold
        print("LMS: ", lms)  # This prints out the LM results for each subtile - the upper threshold

        ## Create permanent water body masks
        mask_outfilename = create_water_masks(surface_water_dir, surface_water_fname, pixel_size_lon_deg, pixel_size_lat_deg, minLon, maxLon, minLat, maxLat)
        

        # ---------------------------------------------------------------------------
        ### Write output as Geotiff - lm
        lm = np.mean(lms) # Calculate the mean of the upper threshold (lms) for all regions: 
        lm_map = np.where(VV_image.band > lm, 1, 0) # lm_map is a binary image of pixels above the threshold; Converts values greater than the mean (i.e. industrial(?)) to 1, all else to 0; a binary image of water(1) and non-water(0)
        lm_map[inan] = NoDataValue  ## convert no data values
        lm_map = lm_map.astype(np.uint8)
        map_type='LM'   
        tiff_outname = write_geotiff(out_dir, image_data, granule, lm_map, map_type,nodata=NoDataValue,compress='lzw',b_colormap=True)      ## Write geotiff
        
        apply_water_body_mask(tiff_outname, mask_outfilename)  # mask permanent water bodies

        
        # ---------------------------------------------------------------------------
        ### Write output as Geotiff - otsu
        otsu_mean = np.mean(otsus) # Calculate the mean of the upper threshold (lms) for all regions: 
        otsu_map = np.where(VV_image.band > otsu_mean, 1, 0) # lm_map is a binary image of pixels above the threshold; Converts values greater than the mean (i.e. industrial(?)) to 1, all else to 0; a binary image of water(1) and non-water(0)
        otsu_map[inan] = NoDataValue  ## convert no data values
        otsu_map = otsu_map.astype(np.uint8)
        map_type='OTSU'       
        tiff_outname = write_geotiff(out_dir, image_data, granule, otsu_map, map_type,nodata=NoDataValue,compress='lzw',b_colormap=True)
        apply_water_body_mask(tiff_outname, mask_outfilename)  # mask permanent water bodies

        
        ### Write output as Geotiff - lower of the two mean thresholds
        final_threshold_val = np.min(np.array([lm, otsu_mean]))
        combined_map = np.where(VV_image.band > final_threshold_val, 1, 0) # lm_map is a binary image of pixels above the threshold; Converts values greater than the mean (i.e. industrial(?)) to 1, all else to 0; a binary image of water(1) and non-water(0)
        combined_map[inan] = NoDataValue  ## convert no data values to
        combined_map = combined_map.astype(np.uint8)
        map_type='combined'       
        tiff_outname = write_geotiff(out_dir, image_data, granule, combined_map, map_type,nodata=NoDataValue,compress='lzw',b_colormap=True)
        apply_water_body_mask(tiff_outname, mask_outfilename)  # mask permanent water bodies

        
        
           
           

if __name__ ==  "__main__":
    # ---------------------------------------------------------------------------
    #Implement Arg parse to pull relevant input parameters from input.txt file
    #Use Argparge to get information from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',type=argparse.FileType('r'))
    p = parser.parse_args()

    with p.filename as file:
        contents = file.read()
        args = ast.literal_eval(contents)
        
    Sigma_files = glob.glob(os.path.join(args['sigma_file_loc'], '*Sigma0_VV.tif'))  # Process VV files
    if len(Sigma_files) == 0:   ## Process VH files, if VV is empty
        Sigma_files = glob.glob(os.path.join(args['sigma_file_loc'], '*Sigma0_VH.tif'))
    Run_amplitude_algorithm(Sigma_files,args['output_dir'], args['surface_water_dir'], args['surface_water_fname'])

