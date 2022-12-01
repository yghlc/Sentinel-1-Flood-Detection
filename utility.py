#!/usr/bin/env python
# Filename: utility.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 24 November, 2022
"""

import os,sys
import json
import shutil
import subprocess
import glob

import time
import math

# ---------------------------------------------------------------------------
# making the output directory
def mk_outdirectory(outpath_full):
    if not os.path.exists(outpath_full):
        print("Making output directory: ", outpath_full)
        os.makedirs(outpath_full)
    return

def outputlogMessage(message):
    timestr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime() )
    outstr = timestr +': '+ message
    print(outstr)

def is_file_exist(file_path):
    # check if a file exists
    if os.path.isfile(file_path):
        return True
    else:
        raise IOError("File : %s not exist"%os.path.abspath(file_path))


def get_name_no_ext(file_path):
    # get file name without extension
    filename_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    return filename_no_ext

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isexists = os.path.exists(path)
    if not isexists:
        try:
            os.makedirs(path)
            print(path + ' Create Success')
            return True
        except IOError:
            print('creating %s failed'%path)
            assert False
    else:
        print(path + '  already exist')
        return False


def save_list_to_txt(file_name, save_list):
    with open(file_name, 'w') as f_obj:
        for item in save_list:
            f_obj.writelines(item + '\n')

def read_list_from_txt(file_name):
    with open(file_name,'r') as f_obj:
        lines = f_obj.readlines()
        lines = [item.strip() for item in lines]
        return lines

def save_dict_to_txt_json(file_name, save_dict):

    # check key is string, int, float, bool or None,
    strKey_dict = {}
    for key in save_dict.keys():
        # print(type(key))
        if type(key) not in [str, int, float, bool, None]:
            strKey_dict[str(key)] = save_dict[key]
        else:
            strKey_dict[key] = save_dict[key]

    # ,indent=2 makes the output more readable
    json_data = json.dumps(strKey_dict,indent=2)
    with open(file_name, "w") as f_obj:
        f_obj.write(json_data)

def read_dict_from_txt_json(file_path):
    if os.path.getsize(file_path) == 0:
        return None
    with open(file_path) as f_obj:
        data = json.load(f_obj)
        return data

def get_file_list_by_ext(ext,folder,bsub_folder):
    """

    Args:
        ext: extension name of files want to find, can be string for a single extension or list for multi extension
        eg. '.tif'  or ['.tif','.TIF']
        folder:  This is the directory, which needs to be explored.
        bsub_folder: True for searching sub folder, False for searching current folder only

    Returns: a list with the files abspath ,eg. ['/user/data/1.tif','/user/data/2.tif']
    Notes: if input error, it will exit the program
    """

    extension = []
    if isinstance(ext, str):
        extension.append(ext)
    elif isinstance(ext, list):
        extension = ext
    else:
        raise ValueError('input extension type is not correct')
    if os.path.isdir(folder) is False:
        raise IOError('input error, directory %s is invalid'%folder)
    if isinstance(bsub_folder,bool) is False:
        raise ValueError('input error, bsub_folder must be a bool value')

    files = []
    sub_folders = []
    sub_folders.append(folder)

    while len(sub_folders) > 0:
        current_sear_dir = sub_folders[0]
        file_names = os.listdir(current_sear_dir)
        file_names = [os.path.join(current_sear_dir,item) for item in file_names]
        for str_file in file_names:
            if os.path.isdir(str_file):
                sub_folders.append(str_file)
                continue
            ext_name = os.path.splitext(str_file)[1]
            for temp in extension:
                if ext_name == temp:
                    # files.append(os.path.abspath(os.path.join(current_sear_dir,str_file)))
                    files.append(str_file)
                    break
        if bsub_folder is False:
            break
        sub_folders.pop(0)

    return files


def delete_file_or_dir(path):
    """
    remove a file or folder
    Args:
        path: the name of file or folder

    Returns: True if successful, False otherwise
    Notes: if IOError occurs or path not exist, it will exit the program
    """
    try:
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        else:
            print('%s not exist'%path)
            assert False
    except IOError:
        print('remove file or dir failed : ' + str(IOError))
        assert False

    return True

def delete_shape_file(input):
    arg1 = os.path.splitext(input)[0]
    exts = ['.shx', '.shp','.prj','.dbf','.cpg']
    for ext in exts:
        file_path = arg1 + ext
        if os.path.isfile(file_path):
            delete_file_or_dir(file_path)
    if os.path.isfile(input):
        delete_shape_file(input)
    return True

def copy_file_to_dst(file_path, dst_name, overwrite=False):
    """
    copy file to a destination file
    Args:
        file_path: the copied file
        dst_name: destination file name

    Returns: True if successful or already exist, False otherwise.
    Notes:  if IOError occurs, it will exit the program
    """
    if os.path.isfile(dst_name) and overwrite is False:
        outputlogMessage("%s already exist, skip copy file"%dst_name)
        return True

    if file_path==dst_name:
        outputlogMessage('warning: shutil.SameFileError')
        return True

    try:
        shutil.copy(file_path,dst_name)
    # except shutil.SameFileError:
    #     basic.outputlogMessage('warning: shutil.SameFileError')
    #     pass
    except IOError:
        raise IOError('copy file failed: '+ file_path)



    if not os.path.isfile(dst_name):
        outputlogMessage('copy file failed, from %s to %s.'%(file_path,dst_name))
        return False
    else:
        outputlogMessage('copy file success: '+ file_path)
        return True

def copy_shape_file(input, output):

    assert is_file_exist(input)

    arg1 = os.path.splitext(input)[0]
    arg2 = os.path.splitext(output)[0]
    # arg_list = ['cp_shapefile', arg1, arg2]
    # return basic.exec_command_args_list_one_file(arg_list, output)

    copy_file_to_dst(arg1+'.shx', arg2 + '.shx', overwrite=True)
    copy_file_to_dst(arg1+'.shp', arg2 + '.shp', overwrite=True)
    copy_file_to_dst(arg1+'.prj', arg2 + '.prj', overwrite=True)
    copy_file_to_dst(arg1+'.dbf', arg2 + '.dbf', overwrite=True)

    outputlogMessage('finish copying %s to %s'%(input,output))

    return True


def os_system_exit_code(command_str):
    '''
    run a common string, check the exit code
    :param command_str:
    :return:
    '''
    res = os.system(command_str)
    if res != 0:
        sys.exit(1)

def exec_command_args_list_one_file(args_list, output):
    """
    execute a command string
    Args:
        args_list: a list contains args

    Returns:

    """
    ps = subprocess.Popen(args_list)
    returncode = ps.wait()
    if os.path.isfile(output):
        return output
    else:
        outputlogMessage('return codes: ' + str(returncode))
        return False

def get_name_by_adding_tail(basename,tail):
    """
    create a new file name by add a tail to a exist file name
    Args:
        basename: exist file name
        tail: the tail name

    Returns: a new name if successfull
    Notes: if input error, it will exit program

    """
    text = os.path.splitext(basename)
    if len(text)<2:
        outputlogMessage('ERROR: incorrect input file name: %s'%basename)
        assert False
    return text[0]+'_'+tail+text[1]

def get_sar_file_list(file_or_dir):
    if os.path.isdir(file_or_dir):
        sar_Sigma_files = glob.glob(os.path.join(file_or_dir, '*Sigma0_VV.tif'))  # Process VV files
        if len(sar_Sigma_files) == 0:  ## Process VH files, if VV is empty
            sar_Sigma_files = glob.glob(os.path.join(file_or_dir, '*Sigma0_VH.tif'))
    else:
        with open(file_or_dir,'r') as f_obj:
            sar_Sigma_files = [line.strip() for line in f_obj.readlines()]
            sar_Sigma_files = [ os.path.expanduser(item) for item in sar_Sigma_files]
    if len(sar_Sigma_files) == 0:
        raise ValueError("No SAR Sigma0 in %s"%file_or_dir)
    return sar_Sigma_files


def meters_to_degrees_onEarth(distance):
    return (distance/6371000.0)*180.0/math.pi

if __name__ == '__main__':
    pass