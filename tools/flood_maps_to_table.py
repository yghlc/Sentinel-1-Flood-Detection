#!/usr/bin/env python
# Filename: flood_maps_to_table.py 
"""
introduction:  list the metadata of flood maps into a table

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 19 January, 2023
"""

import os,sys
from optparse import OptionParser
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
import utility


def get_flood_map_meta_path(flood_map_dir,map_tif):
    filename = utility.get_name_no_ext(map_tif)
    str_list = filename.split('_')
    pre_name = '_'.join(str_list[:-1])  # remove the last one
    meta_file_name = pre_name + '_FD_Results_meta.json'
    return os.path.join(flood_map_dir,meta_file_name)


def test_get_flood_map_meta_path():
    flood_map_dir = os.path.expanduser('~/Bhaltos2/lingcaoHuang/flooding_area/NebraskaDomain/FD_results_thresholding')
    map_tif = os.path.join(flood_map_dir,'S1B_IW_GRDH_1SDV_20190920T001316_20190920T001345_018114_0221BD_29EF_Sigma0_VV_LM.tif')
    meta_path = get_flood_map_meta_path(flood_map_dir,map_tif)
    print(meta_path)


def save_flood_meta_to_table(flood_map_dir,flood_maps_json, save_table_path):
    all_meta_dict = {}

    save_attributes = ["Input-Image","Image-Height","Image-Width","Pixel-mean","Pixel-median","Mean-LM-value",
                       "LM-threshold","LM-Flood-Pixel-Percentage"]

    for idx, flood_json in enumerate(flood_maps_json):
        # meta_path = get_flood_map_meta_path(flood_map_dir,flood_tif)
        meta_dict = utility.read_dict_from_txt_json(flood_json)
        for attr in save_attributes:
            all_meta_dict.setdefault(attr,[]).append(meta_dict[attr])

    #  save to table
    save_table_pd = pd.DataFrame(all_meta_dict)
    with pd.ExcelWriter(save_table_path) as writer:
        save_table_pd.to_excel(writer, sheet_name='flood-meta')
        # # set format
        # workbook = writer.book
        # format = workbook.add_format({'num_format': '#0.000'})
        # acc_talbe_sheet = writer.sheets['accuracy table']
        # acc_talbe_sheet.set_column('G:I',None,format)
        # acc_iou_talbe_sheet = writer.sheets['accuracy table IOU version']
        # acc_iou_talbe_sheet.set_column('G:I', None, format)
        print('save table to %s' % os.path.abspath(save_table_path))



def main(options, args):
    flood_map_dir = args[0]
    save_table_path = options.save_path
    if save_table_path is None:
        save_table_path = os.path.basename(flood_map_dir) + '.xlsx'

    flood_map_jsons = utility.get_file_list_by_pattern(flood_map_dir,'*.json')
    flood_map_jsons = sorted(flood_map_jsons)
    if len(flood_map_jsons) < 1:
        raise ValueError('No flood map in %s'%flood_map_dir)

    save_flood_meta_to_table(flood_map_dir, flood_map_jsons,save_table_path)





if __name__ == "__main__":
    usage = "usage: %prog [options] flood_map_dir "
    parser = OptionParser(usage=usage, version="1.0 2023-01-19")
    parser.description = 'Introduction: save the metadata of flood maps into a table '

    parser.add_option("-s", "--save_path",
                      action="store", dest="save_path",
                      help="the file name for saving the table")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)