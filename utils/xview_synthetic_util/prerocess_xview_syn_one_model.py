import glob
import numpy as np
import argparse
import os

import utils.wv_util as wv
from utils.utils_xview import coord_iou, compute_iou
from utils.xview_synthetic_util import preprocess_xview_syn_data_distribution as pps
from utils.xview_synthetic_util import process_syn_xview_background_wv_split as psx
import pandas as pd
from ast import literal_eval
import json
import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm
import shutil
import cv2
import seaborn as sn
import time


def get_annos_of_model_id(model_id=0):
    src_model_dir = args.annos_save_dir[:-1] + '_all_model/'
    des_model_modelid_dir = args.annos_save_dir[:-1] + '_m{}_only_one_with_modelid/'.format(model_id)
    if not os.path.exists(des_model_modelid_dir):
        os.mkdir(des_model_modelid_dir)
    des_model_dir = args.annos_save_dir[:-1] + '_m{}_only_one/'.format(model_id)
    if not os.path.exists(des_model_dir):
        os.mkdir(des_model_dir)

    lbl_model_txts = glob.glob(os.path.join(src_model_dir, '*.txt'))
    for lt in lbl_model_txts:
        if not pps.is_non_zero_file(lt):
            continue
        name = os.path.basename(lt)
        df_txt = pd.read_csv(lt, header=None, sep=' ')
        df_txt = df_txt[df_txt.loc[:, 5] == model_id]
        if df_txt.empty:
            continue
        df_txt.to_csv(os.path.join(des_model_modelid_dir, name), header=False, index=False, sep=' ')

        df_txt_no_model_id = df_txt.loc[:, :-1]
        df_txt_no_model_id.to_csv(os.path.join(des_model_dir, name), header=False, index=False, sep=' ')


def create_test_dataset_by_model_id(model_id):
    base_dir = args.data_save_dir
    base_val_lbl_files = pd.read_csv(os.path.join(base_dir, 'xviewval_lbl_px23whr3_seed17.txt'), header=None)
    base_val_lbl_name = [os.path.basename(f) for f in base_val_lbl_files.loc[:, 0]]

    test_lbl_files = open(os.path.join(base_dir, 'xviewtest_lbl_px23whr3_seed17_m{}_only.txt'.format(model_id)), 'w')
    test_img_files = open(os.path.join(base_dir, 'xviewtest_img_px23whr3_seed17_m{}_only.txt'.format(model_id)), 'w')
    test_lbl_with_modelid_files = open(os.path.join(base_dir, 'xviewtest_lbl_px23whr3_seed17_with_model_m{}_only.txt'.format(model_id)), 'w')

    lbl_model_with_modelid_dir = args.annos_save_dir[:-1] + '_m{}_only_one_with_modelid/'.format(model_id)
    lbl_model_dir = args.annos_save_dir[:-1] + '_m{}_only_one/'.format(model_id)
    lbl_model_txts = glob.glob(os.path.join(lbl_model_dir, '*.txt'))
    for mf in lbl_model_txts:
        m_name = os.path.basename(mf)
        img_name = m_name.replace('.txt', '.jpg')
        if m_name in base_val_lbl_name:
            test_lbl_files.write('%s\n' % mf)
            test_lbl_with_modelid_files.write('%s\n' % os.path.join(lbl_model_with_modelid_dir, m_name))
            test_img_files.write('%s\n' % os.path.join(args.images_save_dir, img_name))
    test_img_files.close()
    test_lbl_files.close()



def get_txt_contain_model_id(model_id=5, copy_img=False):
    src_model_dir = args.annos_save_dir[:-1] + '_all_model/'
    des_model_dir = args.annos_save_dir[:-1] + '_m{}_all_model/'.format(model_id)
    if not os.path.exists(des_model_dir):
        os.mkdir(des_model_dir)
    if copy_img:
        src_img_dir = os.path.join(args.cat_sample_dir, 'image_with_bbox_indices/px23whr3_seed17_images_with_bbox_with_indices/')
        des_img_dir = os.path.join(args.cat_sample_dir, 'image_with_bbox_indices/px23whr3_seed17_images_with_bbox_with_indices_m{}/'.format(model_id))
        if not os.path.exists(des_img_dir):
            os.mkdir(des_img_dir)

    lbl_model_txts = glob.glob(os.path.join(src_model_dir, '*.txt'))
    for lt in lbl_model_txts:
        if not pps.is_non_zero_file(lt):
            continue
        name = os.path.basename(lt)
        df_txt = pd.read_csv(lt, header=None, sep=' ')
        contain_or_not = df_txt.loc[:, 5] == model_id # series
        contain_or_not = contain_or_not.to_numpy()
        if contain_or_not.any():
            shutil.copy(lt, os.path.join(des_model_dir, name))
            if copy_img:
                img_name = name.replace('.txt', '.jpg')
                shutil.copy(os.path.join(src_img_dir, img_name),
                            os.path.join(des_img_dir, img_name))


def get_image_list_contain_model_id(model_id):
    src_model_dir = args.annos_save_dir[:-1] + '_all_model/'
    image_list = []
    lbl_model_txts = glob.glob(os.path.join(src_model_dir, '*.txt'))
    for lt in lbl_model_txts:
        if not pps.is_non_zero_file(lt):
            continue
        name = os.path.basename(lt)
        df_txt = pd.read_csv(lt, header=None, sep=' ')
        contain_or_not = df_txt.loc[:, 5] == model_id # series
        contain_or_not = contain_or_not.to_numpy()
        if contain_or_not.any():
            image_list.append(name.replace('.txt', '.jpg'))
    return image_list

def get_args(px_thres=None, whr_thres=None, seed=17):
    parser = argparse.ArgumentParser()

    parser.add_argument("--xview_yolo_dir", type=str, help="dir to xViewYOLO",
                        default='/media/lab/Yang/data/xView_YOLO/')

    parser.add_argument("--images_save_dir", type=str, help="to save chip trn val images files",
                        default='/media/lab/Yang/data/xView_YOLO/images/')

    parser.add_argument("--txt_save_dir", type=str, help="to save  related label files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')

    parser.add_argument("--data_list_save_dir", type=str, help="to save selected trn val images and labels",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_cls/data_list/')
    parser.add_argument("--data_save_dir", type=str, help="to save data files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_cls/')
    parser.add_argument("--cat_sample_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/')
    parser.add_argument("--annos_save_dir", type=str, help="to save txt annotation files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')
    parser.add_argument("--font3", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 10}")
    parser.add_argument("-ft2", "--font2", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 13}")
    parser.add_argument("--class_num", type=int, default=1, help="Number of Total Categories")  # 60  6 1
    parser.add_argument("--input_size", type=int, default=608, help="Number of Total Categories")  # 300 416

    args = parser.parse_args()
    #fixme ----------==--------change
    args.images_save_dir = args.images_save_dir + '{}_{}cls/'.format(args.input_size, args.class_num)
    if px_thres:
        args.data_save_dir = args.data_save_dir.format(args.class_num) + 'px{}whr{}_seed{}/'.format(px_thres, whr_thres, seed)
        args.annos_save_dir = args.annos_save_dir + '{}/{}_cls_xcycwh_px{}whr{}/'.format(args.input_size, args.class_num, px_thres, whr_thres)
    else:
        args.data_save_dir = args.data_save_dir.format(args.class_num)
        args.annos_save_dir = args.annos_save_dir + '{}/{}_cls_xcycwh/'.format(args.input_size, args.class_num)
    args.txt_save_dir = args.txt_save_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)
    args.cat_sample_dir = args.cat_sample_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)

    args.data_list_save_dir = args.data_list_save_dir.format(args.input_size, args.class_num)
    if not os.path.exists(args.txt_save_dir):
        os.makedirs(args.txt_save_dir)

    if not os.path.exists(args.annos_save_dir):
        os.makedirs(args.annos_save_dir)

    if not os.path.exists(args.images_save_dir):
        os.makedirs(args.images_save_dir)

    if not os.path.exists(args.cat_sample_dir):
        os.makedirs(args.cat_sample_dir)

    return args


if __name__ == '__main__':
    whr_thres = 3
    px_thres = 23
    seed = 17
    args = get_args(px_thres, whr_thres, seed)

    '''
    change end model id 
    '''
    # px_thres = 23
    # whr_thres = 3
    # end_mid = 6
    # dc.change_end_model_id(px_thres,whr_thres, end_mid)

    '''
    get model 0 txt only
    only one model label
    '''
    # # model_id = 0
    # model_id = 4
    # # model_id = 1
    # get_annos_of_model_id(model_id)

    '''
    create test dataset contain only one model
    '''
    # model_id = 1
    # model_id = 4
    # create_test_dataset_by_model_id(model_id)

    '''
    create *.data for only one model
    '''
    # base_cmt = 'px23whr3_seed17'
    # # model_id = 1
    # model_id = 4
    # psx.create_xview_base_data_for_onemodel_only(model_id, base_cmt)

    '''
    get txt which contains model_id == 5
    '''
    # model_id = 4
    # get_txt_contain_model_id(model_id, copy_img=True)

    '''
    get image_list that contain model_id
    '''
    # model_id = 4
    # model_id = 3
    # model_id = 2
    # image_list = get_image_list_contain_model_id(model_id)
    # print(image_list)











