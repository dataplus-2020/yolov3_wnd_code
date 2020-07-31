import glob
import numpy as np
import argparse
import os

import utils.wv_util as wv
from utils.utils_xview import coord_iou, compute_iou
from utils.xview_synthetic_util import preprocess_xview_syn_data_distribution as pps
from utils.data_process_distribution_vis_util import process_wv_coco_for_yolo_patches_no_trnval as pwv
import pandas as pd
from ast import literal_eval
import json
import datetime
from matplotlib import pyplot as plt
import shutil
import cv2
from tqdm import tqdm
import json
from utils.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc
import time




def get_cat_2_image_tif_name():
    '''
    :return:
    catid_images_name_maps
    catid_tifs_name_maps
    copy raw tif to septerate tif folder
    '''
    json_file = os.path.join(args.txt_save_dir, 'xview_all_{}_{}cls_xtlytlwh.json'.format(args.input_size, args.class_num))
    anns_json = json.load(open(json_file))
    cats = anns_json['categories']
    images = anns_json['images']
    annos = anns_json['annotations']
    annos_cat_imgs = [(an['category_id'], an['image_id']) for an in annos]
    annos_cat_imgs = list(set(annos_cat_imgs))

    cat_ids_names = [(c['id'], c['name']) for c in cats]
    cat_ids_names = list(set(cat_ids_names))
    cat_ids_names.sort()
    print('cat_ids_names ', cat_ids_names)
    cat_images_map = {}
    for cid, cname in cat_ids_names:
        save_dir = os.path.join(args.base_tif_folder, 'raw_{}_tifs/'.format(cid))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if cid not in cat_images_map.keys():
            cat_images_map[cid] = []

        for acid, iid in annos_cat_imgs:
            if acid == cid:
                for img in images:
                    if img['id'] == iid:
                        cat_images_map[cid].append(img['file_name'])
                        break
    json_file = os.path.join(args.txt_save_dir,
                             'xview_catid_imagename_maps_{}_{}cls.json'.format(args.input_size, args.class_num))  # topleft
    json.dump(cat_images_map, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=pwv.MyEncoder)

    cat_tif_maps = {}
    for id in cat_images_map.keys():
        c_images = cat_images_map[id]
        c_tifs = [name.split('_')[0]+'.tif' for name in c_images]
        c_tifs = list(set(c_tifs))
        cat_tif_maps[id] = c_tifs

        for tif in c_tifs:
            shutil.copy(os.path.join(args.image_folder, tif),
                        os.path.join(os.path.join(args.base_tif_folder, 'raw_{}_tifs/'.format(id)), tif))

    json_file = os.path.join(args.txt_save_dir,
                             'xview_catid_tifname_maps_{}_{}cls.json'.format(args.input_size, args.class_num))  # topleft
    json.dump(cat_tif_maps, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=pwv.MyEncoder)

    inter = [a for a in cat_tif_maps[0] if a in cat_tif_maps[1]]
    print('inter ', inter) # ['1702.tif']


def get_args(px_thres=None, whr=None): #
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str,
                        help="Path to folder containing image chips (ie 'Image_Chips/') ",
                        default='/media/lab/Yang/data/xView/train_images/')
    parser.add_argument("--base_tif_folder", type=str,
                        help="Path to folder containing tifs ",
                        default='/media/lab/Yang/data/xView/')
    parser.add_argument("--images_save_dir", type=str, help="to save chip trn val images files",
                        default='/media/lab/Yang/data/xView_YOLO/images/')

    parser.add_argument("--json_filepath", type=str, help="Filepath to GEOJSON coordinate file",
                        default='/media/lab/Yang/data/xView/xView_train.geojson')

    parser.add_argument("--xview_yolo_dir", type=str, help="dir to xViewYOLO",
                        default='/media/lab/Yang/data/xView_YOLO/')

    parser.add_argument("--txt_save_dir", type=str, help="to save  related label files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')
    parser.add_argument("--data_list_save_dir", type=str, help="to save selected trn val images and labels",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_cls/data_list/')

    parser.add_argument("--annos_save_dir", type=str, help="to save txt annotation files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')

    parser.add_argument("--fig_save_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/figures/')

    parser.add_argument("--data_save_dir", type=str, help="to save data files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_cls/')

    parser.add_argument("--cat_sample_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/')

    parser.add_argument("--class_num", type=int, default=2, help="Number of Total Categories")  # 60  6
    parser.add_argument("--input_size", type=int, default=608, help="Number of Total Categories")
    parser.add_argument("--seed", type=int, default=17, help="random seed")

    args = parser.parse_args()
    args.images_save_dir = args.images_save_dir + '{}_{}cls/'.format(args.input_size, args.class_num)
    if px_thres:
        args.annos_save_dir = args.annos_save_dir + '{}/{}_cls_xcycwh_px{}whr{}/'.format(args.input_size, args.class_num, px_thres, whr_thres)
    else:
        args.annos_save_dir = args.annos_save_dir + '{}/{}_cls_xcycwh/'.format(args.input_size, args.class_num)
    args.txt_save_dir = args.txt_save_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)
    args.cat_sample_dir = args.cat_sample_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)
    args.data_save_dir = args.data_save_dir.format(args.class_num)
    args.data_list_save_dir = args.data_list_save_dir.format(args.input_size, args.class_num)

    if not os.path.exists(args.txt_save_dir):
        os.makedirs(args.txt_save_dir)

    if not os.path.exists(args.annos_save_dir):
        os.makedirs(args.annos_save_dir)

    if not os.path.exists(args.images_save_dir):
        os.makedirs(args.images_save_dir)

    if not os.path.exists(args.cat_sample_dir):
        os.makedirs(args.cat_sample_dir)

    if not os.path.exists(args.data_save_dir):
        os.makedirs(args.data_save_dir)
    if not os.path.exists(args.data_list_save_dir):
        os.makedirs(args.data_list_save_dir)
    return args


if __name__ == '__main__':
    args = get_args()

    '''
    create chips and label txt and get all images json, convert from *.geojson to *.json
    '''
    # pwv.create_chips_and_txt_geojson_2_json(args)

    '''
    catid_images_name_maps
    catid_tifs_name_maps
    copy raw tif to septerate tif folder 
    '''
    # get_cat_2_image_tif_name()

    '''
    remove bad images according to  /media/lab/Yang/data/xView/sailboat_bad_raw_tif_names.txt
    983.tif
    '''
    # bad_img_path = '/media/lab/Yang/data/xView/sailboat_bad_raw_tif_names.txt'
    # src_dir = os.path.join(args.base_tif_folder, 'raw_1_tifs')
    # pwv.remove_bad_image(bad_img_path, src_dir)

    # bad_img_path = '/media/lab/Yang/data/xView/sailboat_bad_image_names.txt'
    # src_dir = args.images_save_dir
    # pwv.remove_bad_image(bad_img_path, src_dir)

    # bad_img_path = '/media/lab/Yang/data/xView/airplane_bad_raw_tif_names.txt'
    # src_dir = os.path.join(args.base_tif_folder, 'raw_0_tifs')
    # pwv.remove_bad_image(bad_img_path, src_dir)

    '''
    backup ground truth *.txt 
    remove bbox and annotations of bad cropped .jpg 
    manually get ***** /media/lab/Yang/data/xView/sailboat_airplane_removed_cropped_jpg_names.txt
    '''
    # bad_img_names = '/media/lab/Yang/data/xView/sailboat_airplane_removed_cropped_jpg_names.txt'
    # args = get_args(px_thres=23, whr=3)
    # pwv.remove_txt_and_json_of_bad_image(bad_img_names, args)



    whr_thres = 3 # 3.5
    px_thres= 23
    args = get_args(px_thres, whr_thres)
    save_path = args.cat_sample_dir + 'image_with_bbox/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    img_list = np.sort(glob.glob(os.path.join(args.images_save_dir, '*.jpg')))
    for img in img_list:
        lbl_name = os.path.basename(img).replace('.jpg', '.txt')
        lbl_file = os.path.join(args.annos_save_dir, lbl_name)
        gbc.plot_img_with_bbx(img, lbl_file, save_path, label_index=False)
