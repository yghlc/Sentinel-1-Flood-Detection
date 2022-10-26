#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : {Ryan Cassotto, Clayton Brengman}
# Created Date: 2022/05/26
# version ='1.0'
# ---------------------------------------------------------------------------
"""Sentinel 1 RTC Correction module for IARPA SMART Project"""  
# ---------------------------------------------------------------------------
import os,datetime,time,glob,shutil,zipfile,re
from shapely.geometry import Polygon
from subprocess import Popen, PIPE, STDOUT
import geopandas as gpd
import argparse,ast

# --------------------------------------------------------------------------- 
# Where the Sentinel 1 Toolbox graphing tool exe is located
#baseSNAP = '/software/snap/bin/gpt'
#baseSNAP = '/usr/local/snap/bin/gpt'
baseSNAP = '/home/rcassotto/snap/bin/gpt'

def timestamp(date):
    return time.mktime(date.timetuple())

# ---------------------------------------------------------------------------
# making the output directory
def output_dir(output_dir,gran):
    Output_Directory = os.path.join(output_dir,gran + '_Processed')
    if not os.path.exists(Output_Directory):
        os.makedirs(Output_Directory)
    return Output_Directory

# ---------------------------------------------------------------------------
# Apply precise orbit file
def applyOrbit(new_dir, granule_path_zip, granule):
    aoFlag = ' Apply-Orbit-File '
    oType = '-PcontinueOnFail=\"false\" -PorbitType=\'Sentinel Precise (Auto Download)\' '
    out = '-t ' + new_dir + '/' + granule + '_OB '
    cmd = baseSNAP + aoFlag + out + oType + granule_path_zip
    #print('Applying Precise Orbit file')
    #print(cmd)
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    output = p.stdout.read()
    #print(output)
    #print(" ")
    orbit_corrected_file_path = os.path.join(new_dir, granule + '_OB.dim')
    return orbit_corrected_file_path

# ---------------------------------------------------------------------------
# remove border noise
def applyremovebordernoise(new_dir, in_data_path, baseGran):
    Noise_flag = '  Remove-GRD-Border-Noise '
    out = '-t ' + new_dir + '/' + baseGran + '_OB_GBN '
    in_data_cmd = '-SsourceProduct=' + in_data_path
    cmd = baseSNAP + Noise_flag + out + in_data_cmd
    #print('Removing Border Noise')
    #print(cmd)
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    output = p.stdout.read()
    #print(output)
    #print(" ")
    border_noise_file_path = os.path.join(new_dir, baseGran + '_OB_GBN.dim')
    return border_noise_file_path

# ---------------------------------------------------------------------------
# apply calibrations
def applyCal(new_dir, in_data_path, baseGran):
    calFlag = ' Calibration -PoutputBetaBand=false -PoutputSigmaBand=true '
    out = '-t ' + new_dir + '/' + baseGran + '_OB_GBN_CAL '
    in_data_cmd = '-Ssource=' + in_data_path
    cmd = baseSNAP + calFlag + out + in_data_cmd
    #print('Applying Calibration')
    #print(cmd)
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    output = p.stdout.read()
    #print(output)
    #print(" ")
    calibrated_file_path = os.path.join(new_dir, baseGran + '_OB_GBN_CAL.dim')
    return calibrated_file_path# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------
# apply a speckle filter
def applySpeckle(new_dir, in_data_path, baseGran):
    speckle_flag = ' Speckle-Filter -Pfilter="Refined Lee" '
    out = '-t ' + new_dir + '/' + baseGran + '_OB_GBN_CAL_SP '
    in_data_cmd = '-Ssource=' + in_data_path
    cmd = baseSNAP + speckle_flag + out + in_data_cmd
    #print('Applying Speckle')
    #print(cmd)
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    output = p.stdout.read()
    #print(output)
    #print(" ")
    speckle_filter_file_path = os.path.join(new_dir, baseGran + '_OB_GBN_CAL_SP.dim')
    return speckle_filter_file_path

# ---------------------------------------------------------------------------
# Apply range doppler terrain correction
def applyTC(new_dir, in_data_path, baseGran, pixsiz, extDEM):
    tcFlag = ' Terrain-Correction '
    out = '-t ' + new_dir + '/' + baseGran + '_OB_GBN_CAL_SP_TC '
    in_data_cmd = '-Ssource=' + in_data_path + ' '
    in_data_cmd = in_data_cmd + '-PsaveDEM=false '
#    in_data_cmd = in_data_cmd + '-PsaveProjectedLocalIncidenceAngle=false ' # RKC changed to true on Jan 26, 2022
#    in_data_cmd = in_data_cmd + '-PsaveLocalIncidenceAngle=false ' # RKC added on Jan 26, 2022 
    in_data_cmd = in_data_cmd + '-PsaveIncidenceAngleFromEllipsoid=true ' # RKC added on Jan 26, 2022
    
    in_data_cmd = in_data_cmd + '-PpixelSpacingInMeter=' + str(pixsiz) + ' '
    
    if extDEM != " ":
        in_data_cmd = in_data_cmd + ' -PdemName=\"External DEM\" -PexternalDEMFile=%s -PexternalDEMNoDataValue=0 ' % extDEM
    else:
        in_data_cmd = in_data_cmd + ' -PdemName=\"SRTM 1Sec HGT\" '
    cmd = baseSNAP + tcFlag + out + in_data_cmd
    #print('Applying Terrain Correction -- This will take some time')
    #print(cmd)
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    output = p.stdout.read()
    #print(output)
    #print(" ")
    terrain_correction_file_path = new_dir + '/' + baseGran + '_OB_GBN_CAL_SP_TC.dim'
    return terrain_correction_file_path

# ---------------------------------------------------------------------------
# write files to tiff
def Sigma0_FF_2_gtif(new_dir, Sigma0_directory, granule):
    Sigma0_VV_path = Sigma0_directory + '/' + 'Sigma0_VV.img'
    Sigma0_VH_path = Sigma0_directory + '/' + 'Sigma0_VH.img'
    Incidnc_Angle_path = Sigma0_directory + '/' 'incidenceAngleFromEllipsoid.img'  # new addition 10/12/2022
    Sigma0_VV_save = new_dir + '/' + granule + '_' + Sigma0_VV_path.split('/')[-1].split('.')[0] + '.tif'
    Sigma0_VH_save = new_dir + '/' + granule + '_' + Sigma0_VH_path.split('/')[-1].split('.')[0] + '.tif'
    Incidnc_Angle_save = new_dir + '/' + granule + '_incidenceAngleFromEllipsoid.tif'  # new addition 10/12/2022
    cmd_VV = '/usr/local/bin/gdal_translate -of GTiff ' + Sigma0_VV_path + ' ' + Sigma0_VV_save
    cmd_VH = '/usr/local/bin/gdal_translate -of GTiff ' + Sigma0_VH_path + ' ' + Sigma0_VH_save
    cmd_Incidnc = '/usr/local/bin/gdal_translate -of GTiff ' + Incidnc_Angle_path + ' ' + Incidnc_Angle_save  # new addition 10/12/2022
    p_VV = Popen(cmd_VV, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    p_VH = Popen(cmd_VH, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    p_Incidnc = Popen(cmd_Incidnc, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT) # new addition 10/12/2022
    print('Incidence Angle outfilename: ', Incidnc_Angle_save)

# ---------------------------------------------------------------------------    
#Clean files
def clean_dirs(Output_Directory,Final_Out_Dir):
    basename = os.path.basename(Output_Directory)[:-10]
    print(Output_Directory)
    keep_files = [os.path.join(Output_Directory,basename + '_OB_GBN_CAL_SP_TC.dim'),
                  os.path.join(Output_Directory,basename +'_Sigma0_VH.tif'),
                  os.path.join(Output_Directory,basename + '_Sigma0_VV.tif'),
                  os.path.join(Output_Directory,basename + '_incidenceAngleFromEllipsoid.tif')]
    keep_files_out = [os.path.join(Final_Out_Dir,basename + '_OB_GBN_CAL_SP_TC.dim'),
                      os.path.join(Final_Out_Dir,basename +'_Sigma0_VH.tif'),
                      os.path.join(Final_Out_Dir,basename + '_Sigma0_VV.tif'),
                      os.path.join(Final_Out_Dir,basename + '_incidenceAngleFromEllipsoid.tif')]
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(Output_Directory)) for f in fn]
    remove = []
    for file in files:
        if file not in keep_files:
            remove.append(file)
    for file in remove:
        if os.path.isfile(file):
            os.remove(file)
    
    if not os.path.exists(Final_Out_Dir):
        os.makedirs(Final_Out_Dir)
    
    for file,file_out in zip(keep_files,keep_files_out):
        if os.path.isfile(file):
            shutil.move(file,file_out)
            if '.img' in file_out:
                cmd = '/usr/local/bin/gdal_translate -of GTiff ' + file_out + ' ' + file_out[:-4] + '.tif'
                Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)

    shutil.rmtree(Output_Directory)    # Remove *_Processed directory
    
# ---------------------------------------------------------------------------            
def check_overlap(region_model,GRD_input_list):
    df = gpd.read_file(region_model)
    df_poly     = df.geometry[0].__geo_interface__['coordinates']
    rm_polylist = [tuple(reversed(y)) for x in df_poly for y in x]
    rm_poly = Polygon(rm_polylist)

    keep = []
    for file in GRD_input_list:
        unzipped_file     = zipfile.ZipFile(file, "r")
        manifest          = unzipped_file.read(os.path.join(os.path.basename(file)[:-4] + '.SAFE','manifest.safe'))
        result            = re.search('<gml:coordinates>(.*)</gml:coordinates>', str(manifest))
        grd_coverage_str  = result.group(1)
        grd_coverage_tmp  = tuple(grd_coverage_str.split(' '))
        grd_polylist = []
        for coord in grd_coverage_tmp:
            a,b = coord.split(',')
            grd_polylist.append(tuple((float(a),float(b))))
        grd_poly          = Polygon(grd_polylist)
        
        overlap_area     = grd_poly.intersection(rm_poly).area
        percent_overlap  = overlap_area / rm_poly.area
        if percent_overlap > 0.15:
            keep.append(file)
        
    return keep

def Process_GRD_File(args,GRD_input_list,extDEM_path=" "):
    """ Function that preprocesses the amplitude data
    :param GRD_input_list: Input list of GRDH data
    :param pixsiz: desired pixel size
    :param extDEM_path: If using an external DEM this is the path
    :return: pre processed geotifs in GRD_Processed folder
    """

    start_time = datetime.datetime.now()
    if args['pixsize'] == " ":
        pixsiz = 10.0
    else:
        pixsiz = args['pixsize']
    if extDEM_path == " ":
        print("No external DEM file specified")
    else:
        print("external DEM has been specified")
   # GRD_list_updated = check_overlap(args['region_model'],GRD_input_list)  # This was added by Clayton to eliminate scenes with small overlap
#    for i,GRD_file in enumerate(GRD_list_updated):   # checks for overlap; eliminates scenes with small overlap
#        print('Processing GRD File %s / %s' % (i,len(GRD_list_updated)))   # checks for overlap; eliminates scenes with small overlap
    for i,GRD_file in enumerate(GRD_input_list):
        print('Processing GRD File %s / %s' % (i,len(GRD_input_list)))   
        granule = GRD_file.split('/')[-1].split('.')[0]
        Output_Directory = output_dir(args['output_dir'],granule)
        if len(glob.glob(os.path.join(Output_Directory,'*VV*'))) == 0:           
            # orbit correction
            Orbit_Correction = applyOrbit(Output_Directory, GRD_file, granule)
            # border noise removal
            Border_Noise_Removal = applyremovebordernoise(Output_Directory, Orbit_Correction, granule)
            # Calibration to sigma nought
            Calibration = applyCal(Output_Directory, Border_Noise_Removal, granule)
            # speckle filter
            Speckle_Filter = applySpeckle(Output_Directory, Calibration, granule)
            # terrain correction
            Terrain_Correction = applyTC(Output_Directory, Speckle_Filter, granule, pixsiz, extDEM_path)
            # write out data to geotiffs VV and VH
            Sigma0_directory = Terrain_Correction.replace('.dim', '.data')
            Sigma0_FF_2_gtif(Output_Directory, Sigma0_directory, granule)  
            clean_dirs(Output_Directory,os.path.join(args['output_dir'],'final'))
        else:
            print('%s already has output files...skipping' % (GRD_file))
    end_time = datetime.datetime.now()
    Total_time = timestamp(end_time) - timestamp(start_time)
    return Total_time

if __name__ ==  "__main__":
    
    # ---------------------------------------------------------------------------
    #Use Argparge to get information from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',type=argparse.FileType('r'))
    p = parser.parse_args()

    with p.filename as file:
        contents = file.read()
        args = ast.literal_eval(contents)
        
    if len(args['DEM']) > 0:
        DEM = args['DEM']
        
    GRD_files = glob.glob(os.path.join(args['grd_file_loc'],'*GRDH*.zip'))
    
    if len(args['DEM'])>0:
        time_spent = Process_GRD_File(args,GRD_files, extDEM_path=DEM) # Original RTC.py code for external DEM
    else:
        time_spent = Process_GRD_File(args,GRD_files) 

    print('Process complete, took %s' % (time_spent))
