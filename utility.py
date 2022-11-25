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

# ---------------------------------------------------------------------------
# making the output directory
def mk_outdirectory(outpath_full):
    if not os.path.exists(outpath_full):
        print("Making output directory: ", outpath_full)
        os.makedirs(outpath_full)
    return

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



if __name__ == '__main__':
    pass