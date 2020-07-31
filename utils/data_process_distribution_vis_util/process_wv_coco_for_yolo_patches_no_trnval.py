"""
Copyright 2018 Defense Innovation Unit Experimental
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# import tensorflow as tf
import glob
import numpy as np
import argparse
import os

import utils.wv_util as wv
from utils.utils_xview import coord_iou, compute_iou
from utils.xview_synthetic_util import preprocess_xview_syn_data_distribution as pps
import pandas as pd
from ast import literal_eval
import json
import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm
import shutil
import cv2
import seaborn as sn
import json
from utils.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc
import time

"""
  A script that processes xView imagery. 
  Args:
      image_folder: A folder path to the directory storing xView .tif files
        ie ("xView_data/")

      json_filepath: A file path to the GEOJSON ground truth file
        ie ("xView_gt.geojson")

      val_percent (-t): The percentage of input images to use for test set

      suffix (-s): The suffix for output TFRecord files.  Default suffix 't1' will output
        xview_train_t1.record and xview_test_t1.record

      augment (-a): A boolean value of whether or not to use augmentation

  Outputs:
    Writes two files to the current directory containing training and test data in
        TFRecord format ('xview_train_SUFFIX.record' and 'xview_test_SUFFIX.record')
"""


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def convert_norm(size, box):
    '''
    https://blog.csdn.net/xuanlang39/article/details/88642010
    '''
    dh = 1. / (size[0])  # h--0--y
    dw = 1. / (size[1])  # w--1--x

    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0  # (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert(box):
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]

    return x, y, w, h


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def drawpoly(img, pts, color, thickness=1, style='dotted'):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness, style)
    return img


def drawrect(img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    drawpoly(img, pts, color, thickness, style)
    return img


def shuffle_images_and_boxes_classes(sh_ims, sh_img_names, sh_box, sh_classes_final, sh_boxids, ks):
    """
    Shuffles images, boxes, and classes, while keeping relative matching indices

    Args:
        sh_ims: an array of images
        sh_img_names: a list of image names
        sh_box: an array of bounding box coordinates ([xmin,ymin,xmax,ymax])
        sh_classes_final: an array of classes
        sh_boxids: a list of bbox ids

    Output:
        Shuffle image, boxes, and classes arrays, respectively
        :param sh_boxids:
        :param ks:
    """
    assert len(sh_ims) == len(sh_box)
    assert len(sh_box) == len(sh_classes_final)

    np.random.seed(0)
    perm = np.random.permutation(len(ks))
    out_m = {}
    out_b = {}
    out_c = {}
    out_n = {}
    out_i = {}

    c = 0
    for ind in perm:
        out_m[c] = sh_ims[ks[ind]]
        out_b[c] = sh_box[ks[ind]]
        out_c[c] = sh_classes_final[ks[ind]]
        out_n[c] = sh_img_names[ks[ind]]
        out_i[c] = sh_boxids[ks[ind]]
        c = c + 1
    return out_m, out_n, out_b, out_c, out_i


def get_raw_airplane_tifs():
    tif_names = pd.read_csv('/media/lab/Yang/data/xView/airplane_tif_names.txt', header=None)
    tif_path = '/media/lab/Yang/data/xView/airplane_tifs/'
    if not os.path.exists(tif_path):
        os.mkdir(tif_path)
    src_path =  '/media/lab/Yang/data/xView/train_images/'
    for name in tif_names.loc[:, 0].to_list():
        shutil.copy(os.path.join(src_path, name), tif_path)


def remove_bad_image(bad_img_path, src_dir):
    bad_raw_tif_names = pd.read_csv(bad_img_path, header=None)
    for name in bad_raw_tif_names.loc[:, 0].to_list():
        bad_tif = os.path.join(src_dir, name)
        if os.path.exists(bad_tif):
            os.remove(bad_tif)


def create_chips_and_txt_geojson_2_json(args):

    coords, chips, classes, features_ids = wv.get_labels(args.json_filepath, args.class_num)
    # gs = json.load(open('/media/lab/Yang/data/xView/xView_train.geojson'))
    # print('chips', chips.shape)

    res = (args.input_size, args.input_size)

    file_names =glob.glob(os.path.join(args.image_folder, "*.tif"))
    file_names.sort()
    # print('file_names ', len(file_names))

    # fixme
    df_img_num_names = pd.DataFrame(columns=['id', 'file_name'])

    txt_norm_dir = args.annos_save_dir
    if not os.path.exists(txt_norm_dir):
        os.makedirs(txt_norm_dir)
    image_names_list = []
    _img_num = 0
    img_num_list = []
    image_info_list = []
    annotation_list = []
    if os.path.exists(args.images_save_dir):
            shutil.rmtree(args.images_save_dir)
            os.mkdir(args.images_save_dir)

    for f_name in tqdm(file_names):
        # Needs to be "X.tif", ie ("5.tif")
        # Be careful!! Depending on OS you may need to change from '/' to '\\'.  Use '/' for UNIX and '\\' for windows
        arr = wv.get_image(f_name)
        name = os.path.basename(f_name)
        '''
         # drop 1395.tif  airplane_bad_raw_tif_names.txt
        '''
        if name == '1395.tif' or  name == '2524.tif':
            continue

        ims, img_names, box, classes_final, box_ids = wv.chip_image(arr, coords[chips == name],
                                                                    classes[chips == name],
                                                                    features_ids[chips == name], res,
                                                                    name.split('.')[0], args.images_save_dir)
        if not img_names:
            continue

        ks = [k for k in ims.keys()]
        # print('ks ', len(ks))
        for k in ks:
            file_name = img_names[k]
            image_names_list.append(file_name)
            ana_txt_name = file_name.split(".")[0] + ".txt"
            f_txt = open(os.path.join(txt_norm_dir, ana_txt_name), 'w')
            img = wv.get_image(os.path.join(args.images_save_dir, file_name))
            image_info = {
                "id": _img_num,
                "file_name": file_name,
                "height": img.shape[0],
                "width": img.shape[1],
                "date_captured": datetime.datetime.utcnow().isoformat(' '),
                "license": 1,
                "coco_url": "",
                "flickr_url": ""
            }
            image_info_list.append(image_info)

            for d in range(box_ids[k].shape[0]):
                # create annotation_info
                bbx = box[k][d]
                annotation_info = {
                    "id": box_ids[k][d],
                    "image_id": _img_num,
                    # "image_name": img_name, #fixme: there isn't 'image_name'
                    "category_id": np.int(classes_final[k][d]),
                    "iscrowd": 0,
                    "area": (bbx[2] - bbx[0] + 1) * (bbx[3] - bbx[1] + 1),
                    "bbox": [bbx[0], bbx[1], bbx[2] - bbx[0], bbx[3] - bbx[1]],  # (w1, h1, w, h)
                    # "segmentation": [],
                }
                annotation_list.append(annotation_info)

                bbx = [np.int(b) for b in box[k][d]]
                cvt_bbx = convert_norm(res, bbx)
                f_txt.write(
                    "%s %s %s %s %s\n" % (np.int(classes_final[k][d]), cvt_bbx[0], cvt_bbx[1], cvt_bbx[2], cvt_bbx[3]))
            img_num_list.append(_img_num)
            _img_num += 1
            f_txt.close()
    print('_img_num', _img_num)
    df_img_num_names['id'] = img_num_list
    df_img_num_names['file_name'] = image_names_list
    df_img_num_names.to_csv(
        os.path.join(args.txt_save_dir, 'xview_all_image_names_ids_{}_{}cls.csv'.format(args.input_size, args.class_num)),
        index=False)

    trn_instance = {'info': 'xView {} cls chips 608 yx185 created {}'.format(args.class_num, time.strftime('%Y-%m-%d_%H.%M', time.localtime())),
                    'license': 'license', 'images': image_info_list,
                    'annotations': annotation_list, 'categories': wv.get_all_categories(args.class_num)}
    json_file = os.path.join(args.txt_save_dir,
                             'xview_all_{}_{}cls_xtlytlwh.json'.format(args.input_size, args.class_num))  # topleft
    json.dump(trn_instance, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)


def plot_image_with_bbox_by_image_name_from_patches(image_name):
    df_cat_color = pd.read_csv(
        args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num, args.class_num),
        delimiter='\t')
    cat_ids = df_cat_color['category_id'].tolist()
    cat_colors = df_cat_color['color'].tolist()
    json_file = os.path.join(args.txt_save_dir,
                             'xview_all_{}_{}cls_xtlytlwh.json'.format(args.input_size, args.class_num))

    lbl_json = json.load(open(json_file))
    image_list = lbl_json['images']
    anns_list = lbl_json['annotations']
    annos = []
    for im in image_list:
        if im['file_name'] == image_name:
            im_id = im['id']
            for an in anns_list:
                if an['image_id'] == im_id:
                    annos.append([an['category_id']] + an['bbox'])
            break
    img_bbx_fig_dir = args.cat_sample_dir + 'img_name_2_gt_bbx_figures/'
    if not os.path.exists(img_bbx_fig_dir):
        os.makedirs(img_bbx_fig_dir)
    img_dir = args.images_save_dir
    img = cv2.imread(os.path.join(img_dir, image_name))
    for an in annos:
        cat_id = an[0]
        gt_bbx = an[1:]
        color = literal_eval(cat_colors[cat_ids.index(cat_id)])
        img = cv2.rectangle(img, (gt_bbx[0], gt_bbx[1]), (gt_bbx[0] + gt_bbx[2], gt_bbx[1] + gt_bbx[3]), color, 2)
        cv2.putText(img, text=str(cat_id), org=(gt_bbx[0] + 10, gt_bbx[1] + 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))
    cv2.imwrite(os.path.join(img_bbx_fig_dir, image_name), img)


def remove_txt_and_json_of_bad_image(bad_img_names, args):
    '''
    backup *.txt
    remove bad *.txt of bad_images
    remove bbox of bad_images in the *.json
    :param bad_img_path:
    :return:
    '''
    '''backup *.txt'''
    src_files = glob.glob(os.path.join(args.annos_save_dir.split('_px')[0], '*.txt'))
    if '_px' in args.annos_save_dir:
        if os.path.exists(args.annos_save_dir):
            shutil.rmtree(args.annos_save_dir)
            os.mkdir(args.annos_save_dir)
        backup_dir = args.annos_save_dir
    else:
        backup_dir = args.annos_save_dir[:-1] + '_backup/'
        if not os.path.exists(backup_dir):
            os.mkdir(backup_dir)
    for sf in src_files:
        shutil.copy(sf, backup_dir)

    ''' remove bad .jpg '''
    bad_raw_img_names = pd.read_csv(bad_img_names, header=None).to_numpy()
    for name in bad_raw_img_names[:, 0]:
        lbl_name = name.replace('.jpg', '.txt')
        bad_lbl = os.path.join(args.annos_save_dir, lbl_name)
        # if lbl_name == '1164_0.txt':
        #     print(os.path.exists(bad_lbl))
        if os.path.exists(bad_lbl):
            os.remove(bad_lbl)

        ''' remove bad .jpg in image_names_ids_{}_{}cls.csv'''
    img_id_file = os.path.join(args.txt_save_dir, 'xview_all_image_names_ids_{}_{}cls.csv'.format(args.input_size, args.class_num))
    df_img_num_names = pd.read_csv(img_id_file)
    for name in bad_raw_img_names[:, 0]:
        bad_name = df_img_num_names[df_img_num_names['file_name'] == name]
        for ix in bad_name.index:
            df_img_num_names = df_img_num_names.drop(index=ix)
    print('image_names_ids len ', df_img_num_names.shape[0])
    reduced_img_id_map_file = os.path.join(args.txt_save_dir, 'xview_reduced_image_names_ids_{}_{}cls.csv'.format(args.input_size, args.class_num))
    if os.path.exists(reduced_img_id_map_file):
        os.remove(reduced_img_id_map_file)
    df_img_num_names.to_csv(reduced_img_id_map_file, index=False)

    ''' remove bbox of bad .jpg in xview_all_{}_{}cls_xtlytlwh.json '''
    json_file = os.path.join(args.txt_save_dir,
                             'xview_all_{}_{}cls_xtlytlwh.json'.format(args.input_size, args.class_num))  # topleft

    exist_images = df_img_num_names['file_name'].tolist()
    annos_json = json.load(open(json_file))
    image_list = [a for a in annos_json['images']]
    image_name_list = [a['file_name'] for a in annos_json['images']]
    image_id_list = [a['id'] for a in annos_json['images']]
    annotation_list = annos_json['annotations']
    print('exist_images', len(exist_images))
    print('image_list', len(image_list))
    print('annotation_list', len(annotation_list))
    bad_ids = []
    for ix, imn in enumerate(image_name_list):
        if imn not in exist_images:
            bad_ids.append(image_id_list[ix])
            image_list.remove(image_list[ix])
    for an in annotation_list:
        if an['image_id'] in bad_ids:
            annotation_list.remove(an)
    print('after----------')
    print('image_list', len(image_list))
    print('annotation_list', len(annotation_list))
    # print([(a['id'], a['file_name']) for a in image_list if a['file_name'] not in exist_images])
    # print([x for x in exist_images if x not in [a['file_name'] for a in annos_json['images']]])
    trn_instance = {'info': 'xView aiplanes chips 608 yx185 created', 'license': 'license', 'images': image_list,
                    'annotations': annotation_list, 'categories': wv.get_all_categories(args.class_num)}
    json_file = os.path.join(args.txt_save_dir,
                             'xview_reduced_{}_{}cls_xtlytlwh.json'.format(args.input_size, args.class_num))  # topleft
    if os.path.exists(json_file):
        os.remove(json_file)
    json.dump(trn_instance, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)

def clean_backup_xview_plane_with_constraints(px_thres=20, whr_thres=4):
    '''
    backupu *.txt first
    then remove labels with some constraints
    :param px_thres: each edge length must be larger than px_thres px_thres=6
    :param whr_thres:
    :return:
    '''
    args = get_args(px_thres, whr_thres)
    # gt_files = np.sort(glob.glob(os.path.join(args.annos_save_dir.split('_px')[0], '*.txt')))
    # if os.path.exists(args.annos_save_dir):
    #     shutil.rmtree(args.annos_save_dir)
    #     os.mkdir(args.annos_save_dir)
    #     for f in gt_files:
    #         shutil.copy(f, args.annos_save_dir)

    txt_files = np.sort(glob.glob(os.path.join(args.annos_save_dir, '*.txt')))
    for f in txt_files:
        if pps.is_non_zero_file(f):
            df_txt = pd.read_csv(f, header=None, sep=' ')
            for i in df_txt.index:
                bbx = df_txt.loc[i, 1:] * args.input_size
                # bbx_wh = max(bbx.loc[3]/bbx.loc[4] if bbx.loc[4] else bbx.loc[3], bbx.loc[4]/bbx.loc[3]  if bbx.loc[3] else bbx.loc[4])
                # if '1044_1.txt' in f:
                #     print(f)
                if not bbx.loc[4] or not bbx.loc[3]:
                    print(bbx)
                    print(f)
                xl = round(bbx.loc[1] - bbx.loc[3]/2)
                yl = round(bbx.loc[2] - bbx.loc[4]/2)
                xr = round(bbx.loc[1] + bbx.loc[3]/2)
                yr = round(bbx.loc[2] + bbx.loc[4]/2)
                bbx_wh = max(bbx.loc[3]/bbx.loc[4], bbx.loc[4]/bbx.loc[3])
                w = round(bbx.loc[3])
                h = round(bbx.loc[4])
                #fixme
                # if bbx.loc[3] <= px_thres or bbx.loc[4] <= px_thres or bbx_wh > whr_thres:
                #     df_txt = df_txt.drop(i)

                if bbx_wh >= whr_thres:
                    df_txt = df_txt.drop(i)
                elif (xl<=0 or xr>=args.input_size-1) and (w <= px_thres or h <= px_thres):
                    df_txt = df_txt.drop(i)
                elif (yl<=0 or yr>=args.input_size-1) and (w <= px_thres or h <= px_thres):
                    df_txt = df_txt.drop(i)
            df_txt.to_csv(f, header=False, index=False, sep=' ')


def check_xview_plane_drops():
    args = get_args()
    before_path = args.annos_save_dir[:-1] + '_backup/'
    txt_path = '/media/lab/Yang/data/xView_YOLO/labels/{}'.format(args.input_size)
    before_constrain = os.listdir(before_path)
    after_constrain = os.listdir(args.annos_save_dir)
    drop_list = [f for f in before_constrain if f not in after_constrain]
    drop_rgb_dir = os.path.join(txt_path, '1_cls_drop', 'rgb')
    if not os.path.exists(drop_rgb_dir):
        os.makedirs(drop_rgb_dir)
    drop_rgb_bbx_dir = os.path.join(txt_path, '1_cls_drop', 'rgb_bbx')
    if not os.path.exists(drop_rgb_bbx_dir):
        os.makedirs(drop_rgb_bbx_dir)
    drop_lbl_dir = os.path.join(txt_path, '1_cls_drop', 'lbl')
    if not os.path.exists(drop_lbl_dir):
        os.makedirs(drop_lbl_dir)
    xview_all_img_dir = '/media/lab/Yang/data/xView_YOLO/images/{}/'.format(args.input_size)
    for f in drop_list:
        shutil.copy(os.path.join(before_path, f), os.path.join(drop_lbl_dir, f))
        im = f.replace('.txt', '.jpg')
        shutil.copy(os.path.join(xview_all_img_dir, im),
                    os.path.join(drop_rgb_dir, im))

        gbc.plot_img_with_bbx(os.path.join(drop_rgb_dir, im), os.path.join(before_path, f), drop_rgb_bbx_dir)


def adjust_empty_annos_json(px_thres=None, whr_thres=None):
    args = get_args(px_thres, whr_thres)
    json_file = os.path.join(args.txt_save_dir.split('_px')[0],
                             'xview_reduced_{}_{}cls_xtlytlwh.json'.format(args.input_size, args.class_num))  # topleft
    annos_json = json.load(open(json_file))
    image_list = annos_json['images']
    print('image_list', len(image_list))

    img_id_file = os.path.join(args.txt_save_dir, 'xview_reduced_image_names_ids_{}_{}cls.csv'.format(args.input_size, args.class_num))
    df_img_id = pd.read_csv(img_id_file)
    lbl_files = np.sort(glob.glob(os.path.join(args.annos_save_dir, '*.txt')))
    # print(lbl_files[0])
    annotation_list = []
    image_list = []
    cnt = 0
    for lbl in lbl_files:
        img_name = os.path.basename(lbl).replace('.txt', '.jpg')
        img_id = df_img_id[df_img_id['file_name'] == img_name]['id'].to_numpy()
        img = wv.get_image(args.images_save_dir + img_name)
        image_info = {
            "id": img_id,
            "file_name": img_name,
            "height": img.shape[0],
            "width": img.shape[1],
            "date_captured": datetime.datetime.utcnow().isoformat(' '),
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        }
        image_list.append(image_info)

        if pps.is_non_zero_file(lbl):
            lbl_arr = pd.read_csv(lbl, header=None, sep=' ').to_numpy()
            boxes = lbl_arr[:, 1:] * args.input_size
            boxes[:, 0] = boxes[:, 0] - boxes[:, 2]/2
            boxes[:, 1] = boxes[:, 1] - boxes[:, 3]/2
            boxes = boxes.astype(np.int)
            # create annotation_info
            for i in range(boxes.shape[0]):
                bbx = boxes[i]
                annotation_info = {
                    "id": cnt,
                    "image_id": img_id,
                    # "image_name": img_name, #fixme: there isn't 'image_name'
                    "category_id": 0,
                    "iscrowd": 0,
                    "area": (bbx[2]) * (bbx[3]),
                    "bbox": [bbx[0], bbx[1], bbx[2], bbx[3]],  # (w1, h1, w, h)
                    # "segmentation": [],
                    }
                cnt += 1
                annotation_list.append(annotation_info)
        else:
            annotation_info = {
                "id": cnt,
                "image_id": img_id,
                # "image_name": img_name, #fixme: there isn't 'image_name'
                "category_id": 0,
                "iscrowd": 0,
                "area": 0,
                "bbox": [],  # (w1, h1, w, h)
                # "segmentation": [],
                }
            cnt += 1
            annotation_list.append(annotation_info)

    trn_instance = {'info': 'xview aiplanes chips 608 yx185 created with empty label files', 'license': 'license', 'images': image_list,
                    'annotations': annotation_list, 'categories': wv.get_all_categories(args.class_num)}
    json_file = os.path.join(args.txt_save_dir,
                             'xview_reduced_{}_{}cls_xtlytlwh_with_empty.json'.format(args.input_size, args.class_num))  # topleft
    json.dump(trn_instance, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)

def recover_xview_val_list():
    '''
    recover xview val list with constriants of px_theres=4, whr_thres=3
    :return:
    '''
    lbl_path = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_backup/'
    all_files = glob.glob(lbl_path + '*.txt')
    all_names = [os.path.basename(v) for v in all_files]
    txt_save_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls/data_list/Feb_backup/first_data_set_backup/'
    val_img_txt = open(os.path.join(txt_save_dir, 'xview_val_img.txt'), 'w')
    val_lbl_txt = open(os.path.join(txt_save_dir, 'xview_val_lbl.txt'), 'w')

    trn_lbl_list = pd.read_csv('/media/lab/Yang/code/yolov3/data_xview/1_cls/first_data_set_backup/xview_train_lbl.txt', header=None).loc[:, 0]
    trn_lbl_names = [os.path.basename(f) for f in trn_lbl_list]
    val_lbl_name = [v for v in all_names if v not in trn_lbl_names]
    img_path = '/media/lab/Yang/data/xView_YOLO/images/608/'
    for v in val_lbl_name:
        val_lbl_txt.write('%s\n' % os.path.join(lbl_path, v))
        val_img_txt.write('%s\n' % os.path.join(img_path, v.replace('.txt', '.jpg')))
    val_lbl_txt.close()
    val_img_txt.close()


def create_xview_names(file_name='xview'):
    args = get_args()
    df_cat = pd.read_csv(args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num), sep='\t')
    cat_names = df_cat['category'].to_list()
    f_txt = open(os.path.join(args.data_save_dir, '{}.names'.format(file_name)), 'w')
    for i in range(len(cat_names)):
        f_txt.write("%s\n" % cat_names[i])
    f_txt.close()


def split_trn_val_with_chips(data_name='xview', comments='', seed=17, px_thres=None, whr_thres=None):
    args = get_args(px_thres, whr_thres)

    data_save_dir = args.data_save_dir
    if comments:
        txt_save_dir = args.data_list_save_dir + comments[1:] + '/'
        if not os.path.exists(txt_save_dir):
            os.makedirs(txt_save_dir)
        data_save_dir = os.path.join(data_save_dir, comments[1:])
        if not os.path.exists(data_save_dir):
            os.makedirs(data_save_dir)
    else:
        txt_save_dir = args.data_list_save_dir
        
    lbl_path = args.annos_save_dir
    images_save_dir = args.images_save_dir
    all_files = glob.glob(os.path.join(lbl_path, '*.txt'))
    all_files.sort()
    all_file_names = [os.path.basename(s).split('_')[0] for s in all_files]
    all_name_set = list(set(all_file_names))

    num_files = len(all_name_set)
    trn_num = int(num_files * (1 - args.val_percent))
    np.random.seed(seed)
    perm_files = np.random.permutation(all_name_set)

    # num_files = len(all_files)
    # trn_num = int(num_files * (1 - args.val_percent))
    # np.random.seed(seed)
    # perm_files = np.random.permutation(all_files)

    trn_img_txt = open(os.path.join(txt_save_dir, '{}train_img{}.txt'.format(data_name, comments)), 'w')
    trn_lbl_txt = open(os.path.join(txt_save_dir, '{}train_lbl{}.txt'.format(data_name, comments)), 'w')
    val_img_txt = open(os.path.join(txt_save_dir, '{}val_img{}.txt'.format(data_name, comments)), 'w')
    val_lbl_txt = open(os.path.join(txt_save_dir, '{}val_lbl{}.txt'.format(data_name, comments)), 'w')

    # for i in range(trn_num):
    #     trn_lbl_txt.write("%s\n" % perm_files[i])
    #     lbl_name = perm_files[i].split('/')[-1]
    #     img_name = lbl_name.replace('.txt', '.jpg')
    #     trn_img_txt.write("%s\n" % os.path.join(images_save_dir, img_name))

    for i in range(trn_num):
        start_files = glob.glob(os.path.join(lbl_path, perm_files[i] + '*.txt'))
        for f in start_files:
            trn_lbl_txt.write("%s\n" % f)
            lbl_name = os.path.basename(f)
            img_name = lbl_name.replace('.txt', '.jpg')
            trn_img_txt.write("%s\n" % os.path.join(images_save_dir, img_name))

    trn_img_txt.close()
    trn_lbl_txt.close()

    # for i in range(trn_num, num_files):
    #     val_lbl_txt.write("%s\n" % perm_files[i])
    #     lbl_name = perm_files[i].split('/')[-1]
    #     img_name = lbl_name.replace('.txt', '.jpg')
    #     val_img_txt.write("%s\n" % os.path.join(images_save_dir, img_name))

    for i in range(trn_num, num_files):
        start_files = glob.glob(os.path.join(lbl_path, perm_files[i] + '*.txt'))
        for f in start_files:
            val_lbl_txt.write("%s\n" % f)
            lbl_name = os.path.basename(f)
            img_name = lbl_name.replace('.txt', '.jpg')
            val_img_txt.write("%s\n" % os.path.join(images_save_dir, img_name))

    val_img_txt.close()
    val_lbl_txt.close()

    shutil.copyfile(os.path.join(txt_save_dir, '{}train_img{}.txt'.format(data_name, comments)),
                    os.path.join(data_save_dir, '{}train_img{}.txt'.format(data_name, comments)))
    shutil.copyfile(os.path.join(txt_save_dir, '{}train_lbl{}.txt'.format(data_name, comments)),
                    os.path.join(data_save_dir, '{}train_lbl{}.txt'.format(data_name, comments)))
    shutil.copyfile(os.path.join(txt_save_dir, '{}val_img{}.txt'.format(data_name, comments)),
                    os.path.join(data_save_dir, '{}val_img{}.txt'.format(data_name, comments)))
    shutil.copyfile(os.path.join(txt_save_dir, '{}val_lbl{}.txt'.format(data_name, comments)),
                    os.path.join(data_save_dir, '{}val_lbl{}.txt'.format(data_name, comments)))


def create_json_for_train_or_val_according_to_all_json(data_name='xview', comments='', seed=1024, px_thres=None, whr_thres=None, typestr='val'):
    args = get_args(px_thres, whr_thres)
    if comments:
        comments = comments.format(seed)
        data_list_save_dir = args.data_list_save_dir + comments[1:] + '/'
        if not os.path.exists(data_list_save_dir):
            os.makedirs(data_list_save_dir)
        data_save_dir = os.path.join(args.data_save_dir, comments[1:])
        if not os.path.exists(data_save_dir):
            os.makedirs(data_save_dir)
    else:
        data_list_save_dir = args.data_list_save_dir
        data_save_dir = args.data_save_dir

    val_img_txt = pd.read_csv(os.path.join(data_save_dir, '{}val_img{}.txt'.format(data_name, comments)), header=None)
    val_img_names = [os.path.basename(f) for f in val_img_txt.loc[:, 0]]

    plane_json = json.load(open(os.path.join(args.txt_save_dir,
                             'xview_reduced_{}_{}cls_xtlytlwh.json'.format(args.input_size, args.class_num))))
    all_image_list = plane_json['images']
    all_annos_list = plane_json['annotations']

    val_img_id_dict = {}
    val_images = []
    for vi in all_image_list:
        vi_name = vi['file_name']
        if vi_name in val_img_names:
            val_img_id_dict[vi_name] = vi['id']
            val_images.append(vi)
    json_img_id_file = os.path.join(data_save_dir, 'xview_val{}_{}cls_img_id_map.json'.format(comments, args.class_num))
    json.dump(val_img_id_dict, open(json_img_id_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)
    shutil.copy(json_img_id_file, data_list_save_dir)

    val_annotations = []
    for va in all_annos_list:
        va_img_id = va['image_id']
        if va_img_id in val_img_id_dict.values():
            val_annotations.append(va)

    json_instance = {'info': 'xView {} aiplanes chips 608 yx185 created'.format(typestr), 'license': 'license', 'images': val_images,
                    'annotations': val_annotations, 'categories': wv.get_all_categories(args.class_num)}
    json_file = os.path.join(data_save_dir,
                             'xview_val{}_{}_{}cls_xtlytlwh.json'.format(comments, args.input_size, args.class_num))  # topleft
    json.dump(json_instance, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)
    shutil.copy(json_file, data_list_save_dir)


def split_trn_val_with_tifs():
    tif_pre_names = [os.path.basename(t).split('.')[0] for t in glob.glob(args.image_folder + '*.tif')]
    tif_num = len(tif_pre_names)
    np.random.seed(args.seed)
    perm_pre_tif_files = np.random.permutation(tif_pre_names)
    trn_tif_num = int(tif_num * (1 - args.val_percent))

    all_files = glob.glob(args.images_save_dir + '*.jpg')
    lbl_path = args.annos_save_dir
    num_files = len(all_files)

    trn_img_txt = open(os.path.join(args.txt_save_dir, 'xviewtrain_img_tifsplit.txt'), 'w')
    trn_lbl_txt = open(os.path.join(args.txt_save_dir, 'xviewtrain_lbl_tifsplit.txt'), 'w')
    val_img_txt = open(os.path.join(args.txt_save_dir, 'xviewval_img_tifsplit.txt'), 'w')
    val_lbl_txt = open(os.path.join(args.txt_save_dir, 'xviewval_lbl_tifsplit.txt'), 'w')

    for i in range(trn_tif_num):
        for j in range(num_files):
            jpg_name = all_files[j].split('/')[-1]
            if jpg_name.startswith(perm_pre_tif_files[i]):
                trn_img_txt.write("%s\n" % all_files[j])
                img_name = all_files[j].split('/')[-1]
                lbl_name = img_name.replace('.jpg', '.txt')
                trn_lbl_txt.write("%s\n" % (lbl_path + lbl_name))

    trn_img_txt.close()
    trn_lbl_txt.close()

    for i in range(trn_tif_num, tif_num):
        for j in range(num_files):
            jpg_name = all_files[j].split('/')[-1]
            if jpg_name.startswith(perm_pre_tif_files[i]):
                val_img_txt.write("%s\n" % all_files[j])
                img_name = all_files[j].split('/')[-1]
                lbl_name = img_name.replace('.jpg', '.txt')
                val_lbl_txt.write("%s\n" % (lbl_path + lbl_name))

    val_img_txt.close()
    val_lbl_txt.close()

    shutil.copyfile(os.path.join(args.txt_save_dir, 'xviewtrain_img_tifsplit.txt'),
                    os.path.join(args.data_save_dir, 'xviewtrain_img_tifsplit.txt'))
    shutil.copyfile(os.path.join(args.txt_save_dir, 'xviewtrain_lbl_tifsplit.txt'),
                    os.path.join(args.data_save_dir, 'xviewtrain_lbl_tifsplit.txt'))
    shutil.copyfile(os.path.join(args.txt_save_dir, 'xviewval_img_tifsplit.txt'),
                    os.path.join(args.data_save_dir, 'xviewval_img_tifsplit.txt'))
    shutil.copyfile(os.path.join(args.txt_save_dir, 'xviewval_lbl_tifsplit.txt'),
                    os.path.join(args.data_save_dir, 'xviewval_lbl_tifsplit.txt'))


def get_train_or_val_imgs_by_cat_id(cat_ids, typestr='val'):
    cat_img_ids_maps = json.load(
        open(os.path.join(args.txt_save_dir, 'all_cat_img_ids_dict_{}cls.json'.format(args.class_num))))
    img_ids_names_maps = json.load(
        open(os.path.join(args.txt_save_dir, 'all_image_ids_names_dict_{}cls.json'.format(args.class_num))))
    imgids = [k for k in img_ids_names_maps.keys()]
    imgnames = [v for v in img_ids_names_maps.values()]

    val_files = pd.read_csv(os.path.join(args.data_save_dir, 'xview{}_img.txt'.format(typestr)), header=None)
    val_img_names = [f.split('/')[-1] for f in val_files[0]]
    val_img_ids = [imgids[imgnames.index(n)] for n in val_img_names]
    cat_val_imgs = {}

    all_cats_val_imgs = []
    for rc in cat_ids:
        cat_img_ids = cat_img_ids_maps[rc]
        cat_img_names = [val_img_names[val_img_ids.index(str(i))] for i in cat_img_ids if str(i) in val_img_ids]
        all_cats_val_imgs.extend(cat_img_names)
        cat_val_imgs[rc] = cat_img_names
    cat_val_imgs_files = os.path.join(args.txt_save_dir, 'cat_ids_2_{}_img_names.json'.format(typestr))
    json.dump(cat_val_imgs, open(cat_val_imgs_files, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)


def plot_val_image_with_gt_bbx_by_image_name_from_patches(image_name=None, typstr='val', comments=''):
    # img_ids_names_file = args.txt_save_dir + 'all_image_ids_names_dict_{}cls.json'.format(args.class_num)
    # img_ids_names_map = json.load(open(img_ids_names_file))
    # img_ids = [int(k) for k in img_ids_names_map.keys()]  ## important
    # img_names = [v for v in img_ids_names_map.values()]

    df_cat_color = pd.read_csv(
        args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num, args.class_num),
        delimiter='\t')
    cat_ids = df_cat_color['category_id'].tolist()
    cat_colors = df_cat_color['color'].tolist()
    if comments:
        df_val_img = pd.read_csv(os.path.join(args.data_save_dir, comments[1:], 'xview{}_img{}.txt'.format(typstr, comments)), header=None)
        df_val_gt = pd.read_csv(os.path.join(args.data_save_dir, comments[1:], 'xview{}_lbl{}.txt'.format(typstr, comments)), header=None)
    else:
        df_val_img = pd.read_csv(args.data_save_dir + 'xview{}_img{}.txt'.format(typstr, comments), header=None)
        df_val_gt = pd.read_csv(args.data_save_dir + 'xview{}_lbl{}.txt'.format(typstr, comments), header=None)
    val_img_list = df_val_img[0].tolist()
    # img_size = 608

    val_img_names = [os.path.basename(f) for f in val_img_list]
    for ix, image_name in enumerate(val_img_names):
        if not pps.is_non_zero_file(df_val_gt.iloc[ix, 0]):
            continue
        # img_index = val_img_names.index(image_name)
        gt = pd.read_csv(df_val_gt.iloc[ix, 0], header=None, delimiter=' ')
        gt.iloc[:, 1:] = gt.iloc[:, 1:] * args.input_size
        gt.iloc[:, 1] = gt.iloc[:, 1] - gt.iloc[:, 3] / 2
        gt.iloc[:, 2] = gt.iloc[:, 2] - gt.iloc[:, 4] / 2

        img_bbx_fig_dir = args.cat_sample_dir + 'img_name_2_gt_bbx_figures/' + typestr + '/'
        if not os.path.exists(img_bbx_fig_dir):
            os.makedirs(img_bbx_fig_dir)

        img = cv2.imread(val_img_list[ix])
        for ix in range(len(gt)):
            cat_id = gt.iloc[ix, 0]
            color = literal_eval(cat_colors[cat_ids.index(cat_id)])
            gt_bbx = gt.iloc[ix, 1:].to_numpy()
            gt_bbx = gt_bbx.astype(np.int64)
            img = cv2.rectangle(img, (gt_bbx[0], gt_bbx[1]), (gt_bbx[0] + gt_bbx[2], gt_bbx[1] + gt_bbx[3]), color, 2)
            cv2.putText(img, text=str(ix), org=(gt_bbx[0] + 10, gt_bbx[1] + 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))
        cv2.imwrite(img_bbx_fig_dir + image_name, img)


def plot_img_with_prd_bbx_by_image_name(image_name, cat_ids, score_thres=0.3, whr_thres=3):
    img_id = get_img_id_by_image_name(image_name)

    results = json.load(open(args.results_dir + 'results.json'))
    results_img_id_cat_id = [rs for rs in results if rs['category_id'] in cat_ids and rs['image_id'] == img_id]
    save_dir = args.cat_sample_dir + 'img_name_2_prd_bbx_figures/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    img = cv2.imread(args.images_save_dir + image_name)
    for rs in results_img_id_cat_id:
        cat_id = rs['category_id']
        w, h = rs['bbox'][2:]
        whr = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        rs['bbox'][2] = rs['bbox'][2] + rs['bbox'][0]
        rs['bbox'][3] = rs['bbox'][3] + rs['bbox'][1]
        # print('whr', whr)
        if whr <= whr_thres and rs['score'] >= score_thres:
            bbx = [int(x) for x in rs['bbox']]
            cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[2], bbx[3]), (0, 255, 255), 2)
            cv2.putText(img, text='{} {:.3f}'.format(cat_id, rs['score']), org=(bbx[0] - 5, bbx[1] - 5),
                        # [pr_bx[0], pr[-1]]
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 0))
    cv2.imwrite(save_dir + 'cat{}_'.format(cat_ids) + image_name, img)


def get_img_id_by_image_name(image_name):
    img_ids_names_file = args.txt_save_dir + 'all_image_ids_names_dict_{}cls.json'.format(args.class_num)
    img_ids_names_map = json.load(open(img_ids_names_file))
    img_ids = [int(k) for k in img_ids_names_map.keys()]  ## important
    img_names = [v for v in img_ids_names_map.values()]
    return img_ids[img_names.index(image_name)]


def cnt_ground_truth_overlap_from_pathces(cat_ids, iou_thres=0.5, px_thres=None, whr_thres=None, syn = False):
    if syn:
        args = pps.get_syn_args()
        json_name = '{}_gt_iou_overlap_cnt_each_cat.json'.format(args.syn_display_type)
    else:
        args = get_args(px_thres, whr_thres)
        json_name = 'xview_gt_iou_overlap_cnt_each_cat.json'

    pxwhr_dir = args.annos_save_dir
    lbl_files = np.sort(glob.glob(os.path.join(pxwhr_dir, '*.txt')))
    backup_path = args.annos_save_dir[:-1] + '_backup/'

    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    os.mkdir(backup_path)
    for f in lbl_files:
        shutil.copy(f, backup_path)

    img_names = [os.path.basename(x) for x in lbl_files]

    f_save_dir = args.txt_save_dir + 'gt_iou_overlap/'
    if not os.path.exists(f_save_dir):
        os.mkdir(f_save_dir)

    cnt_cat_overlap = {}
    for c in cat_ids:
        cnt_cat_overlap[c] = 0

    for ix, name in enumerate(img_names):
        if not pps.is_non_zero_file(lbl_files[ix]):
            continue
        df_txt = pd.read_csv(lbl_files[ix], header=None, sep=' ')
        df_lbl_back = df_txt.to_numpy()
        df_lbl_back[:, 1:] = df_lbl_back[:,1:]*args.input_size
        df_lbl_back[:, 3] = df_lbl_back[:,1] + df_lbl_back[:, 3]
        df_lbl_back[:, 4] = df_lbl_back[:,2] + df_lbl_back[:, 4]
        #fixme
        # for new 6 categories
        # only remove duplication of the bbox with the same cat id
        cat_bbox_overlap = {}
        for c in cat_ids:
            cat_bbox_overlap[c] = []
        for i in range(df_txt.shape[0]):
            ci = df_txt.iloc[i, 0]

            for j in range(i+1, df_txt.shape[0]):
                cj = df_txt.iloc[j, 0]
                if ci != cj:
                    continue
                iou = coord_iou(df_lbl_back[i, 1:], df_lbl_back[j, 1:])
                # if name == '2160_1.txt':
                #     print(i, j, iou)
                if iou > iou_thres:
                    cnt_cat_overlap[ci] += 1
                    cat_bbox_overlap[ci].append([ci, i, j])
        cc = [c for c in cat_bbox_overlap.values() if c]
        if cc:
            json_cat_bbox = os.path.join(f_save_dir, name.split('.')[0] + '.json')
            # print(json_cat_bbox)
            json.dump(cat_bbox_overlap, open(json_cat_bbox, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)
    json_file = os.path.join(f_save_dir, json_name)  # topleft
    json.dump(cnt_cat_overlap, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)


def check_duplicate_gt_bbx_for_60_classes(cat_id, iou_thres=0.5, whr_thres=3):
    df_cats_children = pd.read_csv('../../data_xview/6_cls/categories_id_new_to_old_maps.txt', delimiter='\t')
    cat_children_ids = df_cats_children['old_category_id'][df_cats_children['category_id'] == cat_id].to_numpy()
    cat_color = [literal_eval(x) for x in df_cats_children['color'].to_numpy()]
    cat_old_ids = df_cats_children['old_category_id'].to_list()

    save_dir = args.cat_sample_dir + 'duplicate_check_gt_bbx/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    b_str = 'cat_{}_'.format(cat_id)

    dup_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/6_cls/gt_iou_overlap/'
    dup_files = glob.glob(dup_dir + '*.txt')
    img_names = [os.path.basename(x) for x in dup_files]
    img_names = [x.split(b_str)[-1] for x in img_names if x.startswith(b_str)]

    for ix, name in enumerate(img_names):
        img_name = name.replace('.txt', '.jpg')
        img = cv2.imread(args.images_save_dir + img_name)
        show = False
        df_lbl_old = pd.read_csv(args.annos_save_dir + name, header=None, delimiter=' ')

        df_lbl_old.iloc[:, 1:] = df_lbl_old.iloc[:, 1:] * args.input_size
        w = df_lbl_old.iloc[:, 3]
        df_lbl_old = df_lbl_old[w > 4]
        h = df_lbl_old.iloc[:, 4]
        df_lbl_old = df_lbl_old[h > 4]
        whr = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        df_lbl_old = df_lbl_old[whr <= whr_thres]
        df_lbl_old.iloc[:, 1] = df_lbl_old.iloc[:, 1] - df_lbl_old.iloc[:, 3] / 2
        df_lbl_old.iloc[:, 2] = df_lbl_old.iloc[:, 2] - df_lbl_old.iloc[:, 4] / 2
        df_lbl_old.iloc[:, 3] = df_lbl_old.iloc[:, 1] + df_lbl_old.iloc[:, 3]
        df_lbl_old.iloc[:, 4] = df_lbl_old.iloc[:, 2] + df_lbl_old.iloc[:, 4]
        df_lbl_old = df_lbl_old.to_numpy().astype(np.int)
        indices = []
        for ix, lbl in enumerate(df_lbl_old):
            if lbl[0] in cat_children_ids:
                indices.append(ix)
        df_lbl_old = df_lbl_old[indices]
        dup_indices = []
        for i in range(df_lbl_old.shape[0]):
            for j in range(i+1, df_lbl_old.shape[0]):
                # print('i, j', i, j)
                iou = coord_iou(df_lbl_old[i, 1:], df_lbl_old[j, 1:])
                if iou > iou_thres:
                    dup_indices.append(i)
                    dup_indices.append(j)
        dup_indices = list(set(dup_indices))
        for ix in dup_indices:
            lbl = df_lbl_old[ix]
            clr = cat_color[cat_old_ids.index(lbl[0])]
            cv2.rectangle(img, (lbl[1], lbl[2]), (lbl[3], lbl[4]), clr, 2)
            cv2.putText(img, text='{}'.format(lbl[0]), org=(lbl[3] - 5, lbl[4] - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=[0, 255, 255])
            show = True
        if show:
            cv2.imwrite(save_dir + b_str + img_name, img)


def remove_duplicate_gt_bbx(cat_id, syn = False):
    dup_dir = args.txt_save_dir + 'gt_iou_overlap/'
    duplicate_files = np.sort(glob.glob(dup_dir + '*.json'))

    save_dir = args.cat_sample_dir + 'remove_overlap_bbx/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    lbl_path = args.annos_save_dir
    duplicate_names = [os.path.basename(d) for d in duplicate_files if 'xview' not in os.path.basename(d)]
    # b_str = 'cat_{}_'.format(cat_id)
    # duplicate_names = [x.split(b_str)[-1] for x in duplicate_names if x.startswith(b_str)]
    for dx, dn in enumerate(duplicate_names):
        img = cv2.imread(args.images_save_dir + dn.replace('.json', '.jpg'))

        df_lbl = pd.read_csv(lbl_path + dn.replace('.json', '.txt'), header=None, delimiter=' ')
        json_dup = json.load(open(dup_dir + dn))
        # dup_keys = [k for k in json_dup.keys() if json_dup.get(k)]
        for k in json_dup.keys():
            if json_dup.get(k):
                df_dup = np.array(json_dup[k])
                # df_dup = pd.read_csv(dup_dir + b_str + dn, header=None, delimiter=' ').to_numpy()
                drop_list = []
                # print('-----', df_lbl.shape, df_lbl.index)
                for i in range(df_dup.shape[0]):
                    #fixme
                    # .loc with the index; .iloc with the sequence number
                    fst = df_lbl.iloc[df_dup[i, 1]]
                    fst_area = fst[3] * fst[4]
                    # print('df dup %d, 2--' % i, df_dup[i, 2])
                    # print(df_lbl.iloc[13])
                    sec = df_lbl.iloc[df_dup[i, 2]]
                    sec_area = sec[3] * sec[4]

                    if fst_area > sec_area and df_dup[i, 1] not in drop_list:
                        drop_list.append(df_dup[i, 1])
                    elif fst_area <= sec_area and df_dup[i, 2] not in drop_list:
                        drop_list.append(df_dup[i, 2])

                if drop_list:
                    df_lbl = df_lbl.drop(drop_list)
                    df_lbl_back = df_lbl.copy()
                    df_lbl_back.to_csv(args.annos_new_dir + dn.replace('.json', '.txt'), sep=' ', header=None, index=None) # save the new annotation without overlap
                    df_lbl.iloc[:, 1:] = df_lbl_back.iloc[:, 1:]*args.input_size

                    df_lbl.iloc[:, 1] = df_lbl.iloc[:, 1] - df_lbl.iloc[:, 3] / 2
                    df_lbl.iloc[:, 2] = df_lbl.iloc[:, 2] - df_lbl.iloc[:, 4] / 2
                    df_lbl.iloc[:, 3] = df_lbl.iloc[:, 1] + df_lbl.iloc[:, 3]
                    df_lbl.iloc[:, 4] = df_lbl.iloc[:, 2] + df_lbl.iloc[:, 4]
                    df_lbl = df_lbl.to_numpy().astype(np.int)
                    df_lbl = df_lbl[df_lbl[:, 0] == cat_id]
                    for lbl in df_lbl:
                        # if lbl[2] == 465:
                        #     print(lbl[2], lbl[2]/args.input_size)
                        cv2.rectangle(img, (lbl[1], lbl[2]), (lbl[3], lbl[4]), (0, 255, 255), 2)
                        cv2.putText(img, text='{}'.format(lbl[2]), org=(lbl[3] - 10, lbl[4] - 5),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=[0, 255, 255])
                    cv2.imwrite(save_dir + 'cat_{}_'.format(cat_id) + dn.replace('.json', '.jpg'), img)


# def discard_bad_deviation_bbx_of_images():
#     false_bx_files = pd.read_csv('/media/lab/Yang/data/xView/airplane_to_be_reomved_false_bbox.txt', header=None)
#     deviation_files = pd.read_csv('/media/lab/Yang/data/xView/airplane_to_be_adjust_jpg_bbox.txt', header=None)
#     missed_files = pd.read_csv('/media/lab/Yang/data/xView/airplane_to_add_missed_bbox.txt', header=None)
#     src_img_bbx_path = args.cat_sample_dir + 'image_with_bbox/'
#     des_img_bbx_path = args.cat_sample_dir + 'image_with_bbox/discarded/'
#     if not os.path.exists(des_img_bbx_path):
#         os.mkdir(des_img_bbx_path)
#
#     des_dir = args.images_save_dir[:-1] + '_discarded/'
#     if not os.path.exists(des_dir):
#         os.mkdir(des_dir)
#     images_reduced_save_dir = args.images_save_dir[:-1] + '_reduced/'
#     if not os.path.join(images_reduced_save_dir):
#         os.mkdir(images_reduced_save_dir)
#     shutil.copy(args.images_save_dir, images_reduced_save_dir)
#
#     for f in false_bx_files.loc[:, 0]:
#         img_file = os.path.join(images_reduced_save_dir, f)
#         if os.path.exists(img_file):
#             shutil.move(img_file, os.path.join(des_dir, f))
#         img_bbx_file = os.path.join(src_img_bbx_path, f)
#         if os.path.exists(img_bbx_file):
#             shutil.move(img_bbx_file, os.path.join(des_img_bbx_path, f))
#
#     for f in deviation_files.loc[:, 0]:
#         img_file = os.path.join(images_reduced_save_dir, f)
#         if os.path.exists(img_file):
#             shutil.move(img_file, os.path.join(des_dir, f))
#         img_bbx_file = os.path.join(src_img_bbx_path, f)
#         if os.path.exists(img_bbx_file):
#             shutil.move(img_bbx_file, os.path.join(des_img_bbx_path, f))
#
#     for f in missed_files.loc[:, 0]:
#         # only move the missed_bbox_image to the discard folder
#         img_bbx_file = os.path.join(src_img_bbx_path, f)
#         if os.path.exists(img_bbx_file):
#             shutil.move(img_bbx_file, os.path.join(des_img_bbx_path, f))

def cats_split_crop_bbx_from_patches(typestr='val', whr_thres=3, N=200):
    img_files = pd.read_csv(os.path.join(args.data_save_dir, 'xview{}_img.txt'.format(typestr)), header=None)
    lbl_files = pd.read_csv(os.path.join(args.data_save_dir, 'xview{}_lbl.txt'.format(typestr)), header=None)
    lbl_files = lbl_files[0].to_list()
    img_names = [f.split('/')[-1] for f in img_files[0].tolist()]
    # img_ids = [all_img_ids[all_imgs.index(v)] for v in img_names]
    df_cats = pd.read_csv(args.data_save_dir + 'xview.names'.format(args.class_num), header=None)
    cat_names = df_cats.iloc[:, 0]
    for cx, c in enumerate(cat_names):
        cat_dir = args.cat_bbx_patches_dir + 'cat_{}_{}_bbxes/'.format(cx, c)
        if not os.path.exists(os.path.join(cat_dir)):
            os.mkdir(cat_dir)

    for cx, c in enumerate(cat_names):
        cntN = 0
        cat_dir = args.cat_bbx_patches_dir + 'cat_{}_{}_bbxes/'.format(cx, c)
        for ix, name in enumerate(img_names):
            # fixme
            # save first N samples
            cntN += 1
            if cntN > N:
                break
            df_lbl = pd.read_csv(lbl_files[ix], header=None, delimiter=' ')
            df_lbl.iloc[:, 1:] = df_lbl.iloc[:, 1:] * args.input_size
            df_lbl.iloc[:, 1] = df_lbl.iloc[:, 1] - df_lbl.iloc[:, 3] / 2
            df_lbl.iloc[:, 2] = df_lbl.iloc[:, 2] - df_lbl.iloc[:, 4] / 2
            df_lbl.iloc[:, 3] = df_lbl.iloc[:, 1] + df_lbl.iloc[:, 3]
            df_lbl.iloc[:, 4] = df_lbl.iloc[:, 2] + df_lbl.iloc[:, 4]
            df_lbl = df_lbl.to_numpy().astype(np.int)

            cat_bxs = df_lbl[df_lbl[:, 0] == cx]
            img = cv2.imread(args.images_save_dir + name)
            for bx, cb in enumerate(cat_bxs):
                bbx = cb[1:]
                w = bbx[2] - bbx[0]
                h = bbx[3] - bbx[1]
                whr = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
                if whr > whr_thres or w < 4 or h < 4:
                    continue
                bbx_img = img[bbx[1]:bbx[3], bbx[0]:bbx[2], :]  # h w c
                cv2.imwrite(cat_dir + name.split('.')[0] + '_cat{}_{}.jpg'.format(cx, bx), bbx_img)


def cats_split_crop_bbx_from_origins(px_thres=23, whr_thres=3, imgN=200, N=10):
    if args.rare:
        df_category_id = pd.read_csv(args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num),
                                     sep="\t")
    else:
        df_category_id = pd.read_csv(args.data_save_dir + 'categories_id_color_{}_new_group.txt'.format(args.class_num),
                                     sep="\t")
    category_ids = df_category_id['category_id'].tolist()
    category_labels = df_category_id['category_label'].tolist()
    category_names = df_category_id['category'].tolist()
    cat_names = np.unique(category_names).tolist()

    data = json.load(open(args.json_filepath))
    feas = data['features']
    img_names = []
    fea_indices = []
    cat_imgs_dict = {}
    cat_wh_dict = {}

    #fixme
    for cx in range(len(cat_names)):
    # for cx in range(len(cat_names[:1])):
        cat_imgs_dict[cx] = []
        cat_wh_dict[cx] = []
        # if not os.path.exists(args.cat_bbx_origins_dir + 'cat_{}_{}_bbxes/'.format(cx, cat_names[cx])):
        #     os.mkdir(args.cat_bbx_origins_dir + 'cat_{}_{}_bbxes/'.format(cx, cat_names[cx]))

    for i in range(len(feas)):
        clbl = feas[i]['properties']['type_id']
        if clbl in category_labels:
            img_name = feas[i]['properties']['image_id']
            cid = category_ids[category_labels.index(clbl)]
            #fixme
            if img_name not in cat_imgs_dict[cid]:
            # if cid == 0 and img_name not in cat_imgs_dict[cid]:
                cat_imgs_dict[cid].append(img_name)
            img_names.append(img_name)
            fea_indices.append(i)
    #fixme
    # json_file = os.path.join(args.cat_sample_dir, 'xView_cid_2_imgs_maps.json')  # topleft
    json_file = os.path.join(args.cat_sample_dir, 'xView_cid_2_imgs_maps_cat0.json')  # topleft
    json.dump(cat_imgs_dict, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)

    img_names = np.unique(img_names).tolist()
    img_cid_bbxs_dict = {}
    for n in img_names:
        img_cid_bbxs_dict[n] = []

    for fx in fea_indices:
        clbl = feas[fx]['properties']['type_id']
        cid = category_ids[category_labels.index(clbl)]
        bbx = list(literal_eval(feas[fx]['properties']['bounds_imcoords']))  # (w1, h1, w2, h2)
        img_cid_bbxs_dict[feas[fx]['properties']['image_id']].append([cid] + bbx)  # [cid, w1,h1, w2, h2]
    #fixme
    # json_file = os.path.join(args.cat_sample_dir, 'xView_img_2_cid_bbx_maps.json')  # topleft
    json_file = os.path.join(args.cat_sample_dir, 'xView_img_2_cid_bbx_maps_cat0.json')  # topleft
    json.dump(img_cid_bbxs_dict, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)

    del data, feas

    #fixme
    for cx in range(len(cat_names)):
    # for cx in range(len(cat_names[:1])):
        print('category_id', cx)
        img_list = cat_imgs_dict[cx]
        # fixme show imgN images
        for ix, name in enumerate(img_list):
            cntN = 0
            # img = cv2.imread(args.image_folder + name)  # 2355.tif
            cbbx_list = img_cid_bbxs_dict[name]  # [[18, 2726, 2512, 2740, 2518], [18, 2729, 2494, 2737, 2504]]
            # print(name, cbbx_list)
            for i in range(len(cbbx_list)):
                bbx = cbbx_list[i][1:]
                w = bbx[2] - bbx[0]
                h = bbx[3] - bbx[1]
                whr = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
                if cbbx_list[i][0] == cx and whr <= whr_thres and w >= px_thres and h >= px_thres:
                    cat_wh_dict[cx].append([w, h])
                    # fixme
                    # each image select N samples
                    # if ix < imgN and cntN < N:
                    #     cat_dir = args.cat_sample_dir + 'cat_{}_{}_bbxes/'.format(cx, cat_names[cx])
                    #     bbx_img = img[bbx[1]:bbx[3], bbx[0]:bbx[2], :]  # h w c
                    #     cv2.imwrite(cat_dir + name.split('.')[0] + '_cat{}_{}.jpg'.format(cx, i), bbx_img)
                    cntN += 1
    #fixme
    # json_file = os.path.join(args.cat_sample_dir, 'xView_cid_2_wh_maps.json')  #
    json_file = os.path.join(args.cat_sample_dir, 'xView_cid_2_wh_maps_cat0.json')
    json.dump(cat_wh_dict, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)


def plot_image_with_gt_bbx_by_image_name_from_origin(img_name):
    df_cat_color = pd.read_csv(args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num),
                               delimiter='\t')
    cat_ids = df_cat_color['category_id'].tolist()
    cat_colors = df_cat_color['color'].tolist()

    img_name_cid_maps = json.load(open(args.cat_sample_dir + 'xView_img_2_cid_bbx_maps_cat0.json'))
    # img_name_cid_maps = json.load(open(args.cat_sample_dir + 'xView_img_2_cid_bbx_maps.json'))
    img_bbx_list = img_name_cid_maps[img_name]

    img_bbx_fig_dir = args.cat_sample_dir + 'img_name_2_gt_bbx_figures/'
    if not os.path.exists(img_bbx_fig_dir):
        os.makedirs(img_bbx_fig_dir)

    img = cv2.imread(args.image_folder + img_name)
    h, w, _ = img.shape
    for lbl in img_bbx_list:
        print(lbl)
        cat_id = lbl[0]
        cat_color = literal_eval(cat_colors[cat_ids.index(cat_id)])
        lbl[1] = max(0, lbl[1])  # w1
        lbl[2] = max(0, lbl[2])  # h1
        lbl[3] = min(w, lbl[3])  # w2
        lbl[4] = min(h, lbl[4])  # h2

        img = cv2.rectangle(img, (lbl[1], lbl[2]), (lbl[3], lbl[4]), cat_color, 2)
    cv2.imwrite(img_bbx_fig_dir + img_name, img)


def draw_wh_scatter_for_cats(cid_wh_maps, cat_ids):
    df_category_id = pd.read_csv(args.data_save_dir + 'categories_id_color_{}_new_group.txt'.format(args.class_num),
                                 sep="\t")
    category_ids = df_category_id['category_id'].tolist()
    categories = df_category_id['category'].tolist()

    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    for cx, cid in enumerate(cat_ids):
        wh_arr = np.array(cid_wh_maps[str(cid)])
        plt.scatter(wh_arr[:, 0], wh_arr[:, 1], label='{}'.format(categories[category_ids.index(cid)]))

    plt.legend(prop=literal_eval(args.font3), loc='upper right')

    xlabel = 'Width'
    ylabel = "Height"

    plt.title('Categories W H Distribution', literal_eval(args.font2))
    plt.ylabel(ylabel, literal_eval(args.font2))
    plt.xlabel(xlabel, literal_eval(args.font2))
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(args.cat_bbx_origins_dir, 'cats_wh_distribution.jpg'))
    plt.show()


def check_prd_gt_iou(image_name, score_thres=0.3, iou_thres=0.5, rare=False):
    img_id = get_img_id_by_image_name(image_name)
    img = cv2.imread(args.images_save_dir + image_name)
    img_size = img.shape[0]
    gt_rare_cat = pd.read_csv(args.annos_save_dir + image_name.replace('.jpg', '.txt'), header=None, delimiter=' ')
    gt_rare_cat = gt_rare_cat.to_numpy()
    gt_rare_cat[:, 1:] = gt_rare_cat[:, 1:] * img_size
    gt_rare_cat[:, 1] = gt_rare_cat[:, 1] - gt_rare_cat[:, 3] / 2
    gt_rare_cat[:, 2] = gt_rare_cat[:, 2] - gt_rare_cat[:, 4] / 2
    gt_rare_cat[:, 3] = gt_rare_cat[:, 1] + gt_rare_cat[:, 3]
    gt_rare_cat[:, 4] = gt_rare_cat[:, 2] + gt_rare_cat[:, 4]
    if rare:
        prd_lbl_rare = json.load(open(args.results_dir + 'results_rare.json'.format(args.class_num)))  # xtlytlwh
    else:
        prd_lbl_rare = json.load(open(args.results_dir + 'results.json'.format(args.class_num)))  # xtlytlwh

    for g in gt_rare_cat:
        g_bbx = [int(x) for x in g[1:]]
        img = cv2.rectangle(img, (g_bbx[0], g_bbx[1]), (g_bbx[2], g_bbx[3]), (255, 255, 0), 2)  # cyan
        # g_cat_id = g[0]
        for px, p in enumerate(prd_lbl_rare):
            if p['image_id'] == img_id and p['score'] >= score_thres:
                p_bbx = p['bbox']  # xtlytlwh
                p_bbx[2] = p_bbx[0] + p_bbx[2]
                p_bbx[3] = p_bbx[3] + p_bbx[1]
                p_bbx = [int(x) for x in p_bbx]
                p_cat_id = p['category_id']
                iou = coord_iou(p_bbx, g[1:])
                # print('iou', iou)
                if iou >= iou_thres:
                    img = cv2.rectangle(img, (p_bbx[0], p_bbx[1]), (p_bbx[2], p_bbx[3]), (0, 0, 255), 2)
                    cv2.putText(img, text='[{}, {:.3f}]'.format(p_cat_id, iou), org=(p_bbx[0] - 10, p_bbx[1] - 10),
                                # [pr_bx[0], pr[-1]]
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 0))
                    img = cv2.rectangle(img, (g_bbx[0], g_bbx[1]), (g_bbx[2], g_bbx[3]), (255, 255, 0), 2)  # cyan
    if rare:
        rare_result_iout_check_dir = args.results_dir + 'rare_result_iou_check/'
    else:
        rare_result_iout_check_dir = args.results_dir + 'result_iou_check/'
    if not os.path.exists(rare_result_iout_check_dir):
        os.mkdir(rare_result_iout_check_dir)
    cv2.imwrite(rare_result_iout_check_dir + image_name, img)


def get_fp_fn_separtely_by_cat_ids(cat_ids, iou_thres=0.5, score_thres=0.3, whr_thres=3):
    '''
    https://github.com/svpino/tf_object_detection_cm/blob/83cb8a1cf3a5abd24b18a5fc79b5ce99e8a9b317/confusion_matrix.py#L37
    '''
    img_ids_names_file = args.txt_save_dir + 'all_image_ids_names_dict_{}cls.json'.format(args.class_num)
    img_ids_names_map = json.load(open(img_ids_names_file))
    img_ids = [int(k) for k in img_ids_names_map.keys()]  ## important
    img_names = [v for v in img_ids_names_map.values()]

    cat_id_2_img_names_file = args.txt_save_dir + 'cat_ids_2_val_img_names.json'
    cat_id_2_img_names_json = json.load(open(cat_id_2_img_names_file))
    img_name_list = []
    for rc in cat_ids:
        img_name_list.extend(cat_id_2_img_names_json[str(rc)])
    img_name_list = np.unique(img_name_list).tolist()
    print('len images', len(img_name_list))
    img_id_list = [img_ids[img_names.index(n)] for n in img_name_list]

    result_json_file = args.results_dir + 'results.json'
    result_allcat_list = json.load(open(result_json_file))
    result_list = []
    for ri in result_allcat_list:
        if ri['category_id'] in cat_ids and ri['score'] >= score_thres:  # ri['image_id'] in rare_img_id_list and
            result_list.append(ri)
    print('len result_list', len(result_list))
    del result_allcat_list

    fn_color = (255, 0, 0)  # Blue
    fp_color = (0, 0, 255)  # Red
    for c in cat_ids:
        fp_save_dir = args.results_dir + 'result_fp_cats/cat_{}/'.format(c)
        if not os.path.exists(fp_save_dir):
            os.makedirs(fp_save_dir)
        fn_save_dir = args.results_dir + 'result_fn_cats/cat_{}/'.format(c)
        if not os.path.exists(fn_save_dir):
            os.makedirs(fn_save_dir)

    for ix in range(len(img_id_list)):
        img_id = img_id_list[ix]
        img_name = img_name_list[ix]

        # fixme
        fp_list, fn_list = get_fp_fn_list(cat_ids, img_name, img_id, result_list, None, iou_thres, score_thres,
                                          whr_thres)

        img = cv2.imread(args.images_save_dir + img_name)
        h, w = img.shape[0:2]
        for gx, gr in enumerate(fn_list):
            gr_bx = [int(x) for x in gr[1:-1]]
            # g_img = cv2.rectangle(img, (gr_bx[0], gr_bx[1]), (gr_bx[2], gr_bx[3]), fn_color, 2)  # w1, h1, w2, h2
            g_img = drawrect(img, (gr_bx[0], gr_bx[1]), (gr_bx[2], gr_bx[3]), fp_color, thickness=2, style='dotted')
            cv2.putText(g_img, text=str(int(gr[0])), org=(gr_bx[2] - 8, gr_bx[3] - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))
            fn_img = g_img[max(0, gr_bx[1] - 10):min(h, gr_bx[3] + 10), max(0, gr_bx[0] - 10):min(w, gr_bx[2] + 10), :]
            fn_save_dir = args.results_dir + 'result_fn_cats/cat_{}/'.format(int(gr[0]))

            cv2.imwrite(fn_save_dir + 'cat_{}_sample{}_'.format(int(gr[0]), gx) + img_name, fn_img)

        img = cv2.imread(args.images_save_dir + img_name)
        h, w = img.shape[0:2]
        for px, pr in enumerate(fp_list):
            pr_bx = [int(x) for x in pr[1:-1]]
            # p_img = cv2.rectangle(img, (pr_bx[0], pr_bx[1]), (pr_bx[2], pr_bx[3]), fp_color, 2)
            p_img = drawrect(img, (pr_bx[0], pr_bx[1]), (pr_bx[2], pr_bx[3]), fp_color, thickness=1, style='dotted')
            cv2.putText(p_img, text='{}'.format(int(pr[0])), org=(pr_bx[0] + 5, pr_bx[1] + 8),  # [pr_bx[0], pr[-1]]
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 0))
            fp_img = p_img[max(0, pr_bx[1] - 10):min(h, pr_bx[3] + 10), max(0, pr_bx[0] - 10):min(w, pr_bx[2] + 10), :]
            fp_save_dir = args.results_dir + 'result_fp_cats/cat_{}/'.format(int(pr[0]))
            cv2.imwrite(fp_save_dir + 'cat_{}_sample{}_'.format(int(pr[0]), px) + img_name, fp_img)


def get_fp_fn_list(cat_ids, img_name, img_id, result_list, confusion_matrix=None, iou_thres=0.5, score_thres=0.3,
                   px_thres=6, whr_thres=4):
    ''' ground truth '''
    args = get_args()
    good_gt_list = []
    df_lbl = pd.read_csv(args.annos_save_dir + img_name.replace('.jpg', '.txt'), delimiter=' ', header=None)
    df_lbl.iloc[:, 1:] = df_lbl.iloc[:, 1:] * args.input_size
    df_lbl.iloc[:, 1] = df_lbl.iloc[:, 1] - df_lbl.iloc[:, 3] / 2
    df_lbl.iloc[:, 2] = df_lbl.iloc[:, 2] - df_lbl.iloc[:, 4] / 2
    df_lbl.iloc[:, 3] = df_lbl.iloc[:, 1] + df_lbl.iloc[:, 3]
    df_lbl.iloc[:, 4] = df_lbl.iloc[:, 2] + df_lbl.iloc[:, 4]
    df_lbl = df_lbl.to_numpy()
    for cat_id in cat_ids:
        df_lbl_rare = df_lbl[df_lbl[:, 0] == cat_id, :]
        for dx in range(df_lbl_rare.shape[0]):
            w, h = df_lbl_rare[dx, 3] - df_lbl_rare[dx, 1], df_lbl_rare[dx, 4] - df_lbl_rare[dx, 2]
            whr = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            if whr <= whr_thres and w >= px_thres and h >= px_thres:
                good_gt_list.append(df_lbl_rare[dx, :])
    gt_boxes = []
    if good_gt_list:
        good_gt_arr = np.array(good_gt_list)
        gt_boxes = good_gt_arr[:, 1:]
        gt_classes = good_gt_arr[:, 0]

    prd_list = [rx for rx in result_list if rx['image_id'] == img_id]
    prd_lbl_list = []
    for rx in prd_list:
        w, h = rx['bbox'][2:]
        whr = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        rx['bbox'][2] = rx['bbox'][2] + rx['bbox'][0]
        rx['bbox'][3] = rx['bbox'][3] + rx['bbox'][1]
        # print('whr', whr)
        if whr <= whr_thres and rx['score'] >= score_thres:  #
            prd_lbl = [rx['category_id']]
            prd_lbl.extend([int(b) for b in rx['bbox']])  # xywh
            prd_lbl.extend([rx['score']])
            prd_lbl_list.append(prd_lbl)

    matches = []
    dt_boxes = []
    if prd_lbl_list:
        prd_lbl_arr = np.array(prd_lbl_list)
        # print(prd_lbl_arr.shape)
        dt_scores = prd_lbl_arr[:, -1]  # [prd_lbl_arr[:, -1] >= score_thres]
        dt_boxes = prd_lbl_arr[dt_scores >= score_thres][:, 1:-1]
        dt_classes = prd_lbl_arr[dt_scores >= score_thres][:, 0]

    for i in range(len(gt_boxes)):
        for j in range(len(dt_boxes)):
            iou = coord_iou(gt_boxes[i], dt_boxes[j])

            if iou >= iou_thres:
                matches.append([i, j, iou])

    matches = np.array(matches)
    fn_list = []
    fp_list = []

    if matches.shape[0] > 0:
        # Sort list of matches by descending IOU so we can remove duplicate detections
        # while keeping the highest IOU entry.
        matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

        # Remove duplicate detections from the list.
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

        # Sort the list again by descending IOU. Removing duplicates doesn't preserve
        # our previous sort.
        matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

        # Remove duplicate ground truths from the list.
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

    for i in range(len(gt_boxes)):
        if matches.shape[0] > 0 and matches[matches[:, 0] == i].shape[0] == 1 and confusion_matrix:
            mt_i = matches[matches[:, 0] == i]
            # print('mt_i', mt_i.shape)
            confusion_matrix[
                cat_ids.index(gt_classes[int(mt_i[0, 0])]), cat_ids.index(dt_classes[int(mt_i[0, 1])])] += 1
        else:
            # fixme
            # unique matches at most has one match for each ground truth
            # 1. ground truth id deleted due to duplicate detections  --> FN
            # 2. matches.shape[0] == 0 --> no matches --> FN
            if confusion_matrix:
                confusion_matrix[cat_ids.index(gt_classes[i]), -1] += 1
            c_box_iou = [gt_classes[i]]
            c_box_iou.extend(gt_boxes[i])
            c_box_iou.append(0)  # [cat_id, box[0:4], iou]
            fn_list.append(c_box_iou)

    for j in range(len(dt_boxes)):
        # fixme
        # detected object not in the matches --> FP
        # 1. deleted due to duplicate ground truth (background-->Y_prd)
        # 2. lower than iou_thresh (maybe iou=0)  (background-->Y_prd)
        if matches.shape[0] > 0 and matches[matches[:, 1] == j].shape[0] == 0:
            if confusion_matrix:
                confusion_matrix[-1, cat_ids.index(dt_classes[j])] += 1
            c_box_iou = [dt_classes[j]]
            c_box_iou.extend(dt_boxes[j])
            c_box_iou.append(0)  # [cat_id, box[0:4], iou]
            # print(c_box_iou)
            fp_list.append(c_box_iou)
        elif matches.shape[0] == 0:  # fixme
            if confusion_matrix:
                confusion_matrix[-1, cat_ids.index(dt_classes[j])] += 1
            c_box_iou = [dt_classes[j]]
            c_box_iou.extend(dt_boxes[j])
            c_box_iou.append(0)  # [cat_id, box[0:4], iou]
            fp_list.append(c_box_iou)
    return fp_list, fn_list


def get_confusion_matrix(cat_ids, iou_thres=0.5, score_thres=0.3, px_thres=4, whr_thres=3):
    '''
    https://github.com/svpino/tf_object_detection_cm/blob/83cb8a1cf3a5abd24b18a5fc79b5ce99e8a9b317/confusion_matrix.py#L37
    '''
    img_ids_names_file = args.txt_save_dir + 'all_image_ids_names_dict_{}cls.json'.format(args.class_num)
    img_ids_names_map = json.load(open(img_ids_names_file))
    img_ids = [int(k) for k in img_ids_names_map.keys()]  ## important
    img_names = [v for v in img_ids_names_map.values()]

    cat_id_2_img_names_file = args.txt_save_dir + 'cat_ids_2_val_img_names.json'
    cat_id_2_img_names_json = json.load(open(cat_id_2_img_names_file))
    img_name_list = []
    for rc in cat_ids:
        img_name_list.extend(cat_id_2_img_names_json[str(rc)])
    img_name_list = np.unique(img_name_list).tolist()
    print('len images', len(img_name_list))
    img_id_list = [img_ids[img_names.index(n)] for n in img_name_list]

    result_json_file = args.results_dir + 'results.json'
    result_allcat_list = json.load(open(result_json_file))
    result_list = []
    # #fixme filter, and rare_result_allcat_list contains rare_cat_ids, rare_img_id_list and object score larger than score_thres
    for ri in result_allcat_list:
        if ri['category_id'] in cat_ids and ri['score'] >= score_thres:  # ri['image_id'] in rare_img_id_list and
            result_list.append(ri)
    print('len result_list', len(result_list))
    del result_allcat_list

    confusion_matrix = np.zeros(shape=(len(cat_ids) + 1, len(cat_ids) + 1))

    fn_color = (255, 0, 0)  # Blue
    fp_color = (0, 0, 255)  # Red
    # save_dir = args.results_dir + 'result_figures_fp_fn_nms_iou{}/'.format(iou_thres)
    save_dir = args.results_dir + 'result_figures_fp_fn_nms_iou{}_cats_{}/'.format(iou_thres, cat_ids)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    confu_mat_dir = args.results_dir + 'result_confusion_matrix_iou{}/'.format(iou_thres)
    if not os.path.exists(confu_mat_dir):
        os.mkdir(confu_mat_dir)
    for ix in range(len(img_id_list)):
        img_id = img_id_list[ix]
        # for img_id in [3160]: # 6121 2609 1238 489 6245
        #     ix = rare_img_id_list.index(img_id)
        img_name = img_name_list[ix]

        # fixme
        fp_list, fn_list = get_fp_fn_list(cat_ids, img_name, img_id, result_list, confusion_matrix, iou_thres,
                                          score_thres, px_thres, whr_thres)

        rcids = []
        img = cv2.imread(args.images_save_dir + img_name)
        for gr in fn_list:
            gr_bx = [int(x) for x in gr[1:-1]]
            img = cv2.rectangle(img, (gr_bx[0], gr_bx[1]), (gr_bx[2], gr_bx[3]), fn_color, 2)
            rcids.append(int(gr[0]))
            cv2.putText(img, text=str(int(gr[0])), org=(gr_bx[2] - 20, gr_bx[3] - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))

        for pr in fp_list:
            pr_bx = [int(x) for x in pr[:-1]]
            rcids.append(pr_bx[0])
            img = cv2.rectangle(img, (pr_bx[1], pr_bx[2]), (pr_bx[3], pr_bx[4]), fp_color, 2)
            cv2.putText(img, text='{} {:.3f}'.format(pr_bx[0], pr[-1]), org=(pr_bx[1] + 5, pr_bx[2] + 10),
                        # [pr_bx[0], pr[-1]]
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 0))
        rcids = list(set(rcids))
        cv2.imwrite(save_dir + 'cat{}_'.format(rcids) + img_name_list[ix], img)
    np.save(confu_mat_dir + 'confusion_matrix.npy', confusion_matrix)

    return confusion_matrix


def plot_img_with_prd_bbx_by_cat_id(cat_id, typestr, score_thres=0.3, whr_thres=3):
    cat_ids_2_img_name_maps = json.load(open(args.txt_save_dir + 'cat_ids_2_{}_img_names.json'.format(typestr)))
    img_names_of_cat_id = cat_ids_2_img_name_maps[cat_id]

    img_ids_2_names = json.load(open(args.txt_save_dir + 'all_image_ids_names_dict_{}cls.json'.format(args.class_num)))
    img_ids = [int(k) for k in img_ids_2_names.keys()]
    img_names = [v for v in img_ids_2_names.values()]

    img_ids_of_cat_id = []
    for n in img_names_of_cat_id:
        ix = img_names.index(n)
        img_ids_of_cat_id.append(img_ids[ix])

    results = json.load(open(args.results_dir + 'results.json'))
    cid = int(cat_id)
    results_cat_id = [rs for rs in results if rs['category_id'] == cid]
    save_dir = args.results_dir + 'results_prd_bbx_cat_id_{}/'.format(cat_id)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for ix, id in enumerate(img_ids_of_cat_id):
        name = img_names_of_cat_id[ix]
        img = cv2.imread(args.images_save_dir + name)
        results_cat_id_img_id = [rs for rs in results_cat_id if rs['image_id'] == id]
        for rs in results_cat_id_img_id:
            if rs['image_id'] == id:
                w, h = rs['bbox'][2:]
                whr = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
                rs['bbox'][2] = rs['bbox'][2] + rs['bbox'][0]
                rs['bbox'][3] = rs['bbox'][3] + rs['bbox'][1]
                print('whr', whr)
                if whr <= whr_thres and rs['score'] >= score_thres:
                    bbx = [int(x) for x in rs['bbox']]
                    cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[2], bbx[3]), (0, 255, 255), 2)
                    cv2.putText(img, text=cat_id, org=(bbx[0] - 5, bbx[1] - 5),  # [pr_bx[0], pr[-1]]
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 0))
        cv2.imwrite(save_dir + 'cat{}_'.format(cat_id) + name, img)


def plot_prd_gt_bbox_by_cat_ids(cat_ids, score_thres=0.3, whr_thres=4):
    '''
    all: all prd & all gt
    '''
    img_ids_names_file = args.txt_save_dir + 'all_image_ids_names_dict_{}cls.json'.format(args.class_num)
    img_ids_names_map = json.load(open(img_ids_names_file))
    img_ids = [int(k) for k in img_ids_names_map.keys()]  ## important
    img_names = [v for v in img_ids_names_map.values()]

    df_val_img = pd.read_csv(args.data_save_dir + 'xviewval_img.txt'.format(args.class_num), header=None)
    df_val_gt = pd.read_csv(args.data_save_dir + 'xviewval_lbl.txt'.format(args.class_num), header=None)
    val_img_list = df_val_img[0].tolist()
    val_id_list = df_val_gt[0].tolist()
    val_img_names = [f.split('/')[-1] for f in val_img_list]
    val_img_ids = [img_ids[img_names.index(x)] for x in val_img_names]
    print('gt len', len(val_img_names))

    result_json_file = args.results_dir + 'results.json'
    result_allcat_list = json.load(open(result_json_file))
    result_list = []
    for ri in result_allcat_list:
        if ri['category_id'] in cat_ids and ri['score'] >= score_thres:
            result_list.append(ri)
    del result_allcat_list

    prd_color = (255, 255, 0)
    gt_color = (0, 255, 255)  # yellow
    save_dir = args.results_dir + 'result_figures_prd_score_gt_bbox_cat_ids/'.format(cat_ids)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for ix in range(len(val_img_list)):
        img = cv2.imread(val_img_list[ix])
        name = val_img_names[ix]
        img_size = img.shape[0]
        gt_cat = pd.read_csv(val_id_list[ix], delimiter=' ', header=None).to_numpy()
        # gt_cat = pd.read_csv(args.annos_save_dir + name.replace('.jpg', 'txt'), delimiter=' ', header=None)
        gt_cat = np.array([g for g in gt_cat if g[0] in cat_ids])
        if name == '1139_4.jpg':
            print('len gt_cat', len(gt_cat))
        gt_lbl_list = []

        for gx in range(gt_cat.shape[0]):
            gt_cat[gx, 1:] = gt_cat[gx, 1:] * img_size
            gt_cat[gx, 1] = gt_cat[gx, 1] - gt_cat[gx, 3] / 2
            gt_cat[gx, 2] = gt_cat[gx, 2] - gt_cat[gx, 4] / 2
            w = gt_cat[gx, 3]
            h = gt_cat[gx, 4]
            whr = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            if whr > whr_thres or w < 4 or h < 4:
                continue
            gt_cat[gx, 3] = gt_cat[gx, 3] + gt_cat[gx, 1]
            gt_cat[gx, 4] = gt_cat[gx, 4] + gt_cat[gx, 2]
            gt_lbl_list.append(gt_cat[gx, :])
        if name == '1139_4.jpg':
            print('len gt_lbl_list', len(gt_lbl_list))

        img_prd_lbl_list = []
        result_img_list = [rs for rs in result_list if rs['image_id'] == val_img_ids[ix]]
        for jx in result_img_list:
            jx_cat_id = jx['category_id']
            bbx = jx['bbox']  # xtopleft, ytopleft, w, h
            whr = np.maximum(bbx[2] / (bbx[3] + 1e-16), bbx[3] / (bbx[2] + 1e-16))
            if whr > whr_thres or bbx[2] < 4 or bbx[3] < 4 or jx['score'] < score_thres:
                continue
            pd_rare = [jx_cat_id]
            bbx[2] = bbx[0] + bbx[2]
            bbx[3] = bbx[1] + bbx[3]
            bbx = [int(b) for b in bbx]
            pd_rare.extend(bbx)
            pd_rare.append(jx['score'])
            img_prd_lbl_list.append(pd_rare)

        for pr in img_prd_lbl_list:  # pr_list
            # print(pr)
            bbx = pr[1:-1]
            img = cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[2], bbx[3]), prd_color, 2)
            cv2.putText(img, text='{} {:.3f}'.format(pr[0], pr[-1]), org=(bbx[0] - 5, bbx[1] - 10),  # cid, score
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=prd_color)
        for gt in gt_lbl_list:  # gt_list
            gt_bbx = gt[1:]
            gt_bbx = [int(g) for g in gt_bbx]
            img = cv2.rectangle(img, (gt_bbx[0], gt_bbx[1]), (gt_bbx[2], gt_bbx[3]), gt_color, 2)
            cv2.putText(img, text='{}'.format(int(gt[0])), org=(gt_bbx[2] - 20, gt_bbx[3] - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=gt_color)
        # if gt_list or pr_list:
        if gt_lbl_list or img_prd_lbl_list:
            cv2.imwrite(save_dir + 'cat_{}_'.format(cat_ids) + val_img_names[ix], img)


def summary_confusion_matrix(confusion_matrix, rare_cat_ids, iou_thres=0.5, rare=False):
    '''
    https://github.com/svpino/tf_object_detection_cm/blob/83cb8a1cf3a5abd24b18a5fc79b5ce99e8a9b317/confusion_matrix.py#L37
    '''
    print("\nConfusion Matrix:")
    print(confusion_matrix, "\n")
    results = []
    df_category_id = pd.read_csv(args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num),
                                 sep="\t")
    category_names = df_category_id['category'].tolist()
    category_ids = df_category_id['category_id'].tolist()
    rare_cat_names = []
    for ix, id in enumerate(rare_cat_ids):
        name = category_names[category_ids.index(id)]
        rare_cat_names.append(name)

        total_target = np.sum(confusion_matrix[ix, :])
        total_predicted = np.sum(confusion_matrix[:, ix])

        precision = float(confusion_matrix[ix, ix] / total_predicted)
        recall = float(confusion_matrix[ix, ix] / total_target)

        # print('precision_{}@{}IOU: {:.2f}'.format(name, IOU_THRESHOLD, precision))
        # print('recall_{}@{}IOU: {:.2f}'.format(name, IOU_THRESHOLD, recall))

        results.append({'category': name, 'precision_@{}IOU'.format(iou_thres): precision,
                        'recall_@{}IOU'.format(iou_thres): recall})

    df = pd.DataFrame(results)
    print(df)
    if rare:
        save_dir = args.results_dir + 'rare_result_confusion_matrix_iou{}/'.format(iou_thres)
        file_name = 'rare_result_confusion_mtx.csv'
    else:
        save_dir = args.results_dir + 'result_confusion_matrix_iou{}/'.format(iou_thres)
        file_name = 'result_confusion_mtx.csv'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    df.to_csv(save_dir + file_name)


def plot_confusion_matrix(confusion_matrix, rare_cat_ids, iou_thres=0.5, rare=False):
    df_category_id = pd.read_csv(args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num),
                                 sep="\t")
    category_names = df_category_id['category'].tolist()
    category_ids = df_category_id['category_id'].tolist()
    rare_cat_names = []
    for ix, id in enumerate(rare_cat_ids):
        name = category_names[category_ids.index(id)]
        rare_cat_names.append(name)

    rare_cat_names.append('background')
    rare_cat_ids.append(000)
    # print(confusion_matrix.shape)
    df_cm = pd.DataFrame(confusion_matrix[:len(confusion_matrix), :len(confusion_matrix)],
                         index=[i for i in rare_cat_names], columns=[i for i in rare_cat_names])
    plt.figure(figsize=(10, 8))

    p = sn.heatmap(df_cm, annot=True, fmt='g', cmap='YlGnBu')  # center=True, cmap=cmap,
    fig = p.get_figure()
    fig.tight_layout()
    plt.xticks(rotation=325)
    if rare:
        save_dir = args.results_dir + 'rare_result_confusion_matrix_iou{}/'.format(iou_thres)
        file_name = 'rare_confusion_matrix.png'
    else:
        save_dir = args.results_dir + 'result_confusion_matrix_iou{}/'.format(iou_thres)
        file_name = 'confusion_matrix.png'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    fig.savefig(save_dir + file_name, bbox_inches='tight')
    plt.show()


def autolabel(ax, rects, x, labels, ylabel=None, rotation=90, txt_rotation=45):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=txt_rotation)
        if rotation == 0:
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
        else:  # rotation=90, multiline label
            xticks = []
            for i in range(len(labels)):
                xticks.append('{} {}'.format(x[i], labels[i]))
            ax.set_xticklabels(xticks, rotation=rotation)
        # ax.set_ylabel(ylabel)
        ax.grid(True)


def draw_bar_for_each_cat_cnt_with_txt_subplot(x, ylist, labels, title, save_dir, png_name, rotation=0):
    width = 0.35
    fig, (ax0, ax1, ax2) = plt.subplots(3)

    x0 = x[:20]
    x1 = x[20:40]
    x2 = x[40:60]

    y0 = ylist[:20]
    y1 = ylist[20:40]
    y2 = ylist[40:60]

    ylabel = "Num"

    rects0 = ax0.bar(np.array(x0) - width / 2, y0, width)  # , label=labels
    autolabel(ax0, rects0, x0, labels[:20], ylabel, rotation=rotation)

    rects1 = ax1.bar(np.array(x1) - width / 2, y1, width)  # , label=labels
    autolabel(ax1, rects1, x1, labels[20:40], ylabel, rotation=rotation)

    rects2 = ax2.bar(np.array(x2) - width / 2, y2, width)  # , label=labels
    autolabel(ax2, rects2, x2, labels[40:60], ylabel, rotation=rotation)

    xlabel = 'Categories'
    ax0.set_title(title, literal_eval(args.font3))
    ax2.set_xlabel(xlabel, literal_eval(args.font3))
    # plt.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.savefig(os.path.join(save_dir, png_name))
    plt.show()
    # plt.cla()
    # plt.close()


def draw_bar_for_each_cat_cnt_with_txt_rotation(x, ylist, labels, title, save_dir, png_name, rotation=0,
                                                txt_rotation=45, sort=False):
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    fig, ax = plt.subplots(1)
    width = 0.35
    ylabel = "Num"
    if sort:
        rects0 = ax.bar(np.array(range(len(x))) - width / 2, ylist, width)  # , label=labels
        ax.set_xticks(range(len(x)))
    else:
        rects0 = ax.bar(np.array(x) - width / 2, ylist, width)  # , label=labels
        ax.set_xticks(x)
    autolabel(ax, rects0, x, labels, ylabel, rotation=rotation, txt_rotation=txt_rotation)

    xlabel = 'Categories'
    ax.set_title(title, literal_eval(args.font3))
    ax.set_xlabel(xlabel, literal_eval(args.font3))
    plt.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.savefig(os.path.join(save_dir, png_name))
    plt.show()
    # plt.cla()
    # plt.close()


def draw_bar_for_each_cat_cnt(x, ylist, labels, title, save_dir, png_name, rotation=0):
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    if rotation == 0:
        for i in range(len(ylist)):
            bar = plt.bar(x[i], ylist[i])
        plt.xticks(x, x)
    else:  # rotation=90, multiline label
        xticks = []
        for i in range(len(ylist)):
            bar = plt.bar(x[i], ylist[i])
            xticks.append('{} {}'.format(x[i], labels[i]))
        plt.xticks(x, xticks, rotation=rotation)

    xlabel = 'Categories'
    ylabel = "Num"
    # plt.xticks(x, labels, rotation=rotation)
    # plt.xticks(x, x, rotation=rotation)
    plt.title(title, literal_eval(args.font2))
    plt.ylabel(ylabel, literal_eval(args.font2))
    plt.xlabel(xlabel, literal_eval(args.font2))
    plt.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.grid()
    # plt.legend()
    plt.savefig(os.path.join(save_dir, png_name))
    plt.show()
    # plt.cla()


def categories_cnt_for_all():
    df_category_id = pd.read_csv(args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num),
                                 sep="\t")
    categories = {}
    for ci in range(args.class_num):
        categories[ci] = 0

    lbl_txt = glob.glob(args.annos_save_dir + '*.txt')
    for i in range(len(lbl_txt)):
        if not pps.is_non_zero_file(lbl_txt[i]):
            continue
        lbl = pd.read_csv(lbl_txt[i], sep=' ', header=None)
        for j in range(lbl.shape[0]):
            categories[lbl[0].iloc[j]] += 1

    df_cat = pd.DataFrame({"category_id": [], "category_count": []})
    df_cat['category_id'] = [k for k in categories.keys()]
    df_cat['category_count'] = [v for v in categories.values()]

    df_merge_cat_super = pd.merge(df_cat, df_category_id)
    df_merge_cat_super.to_csv(
        os.path.join(args.txt_save_dir, 'all_{}_cat_cnt_{}cls.csv'.format(args.input_size, args.class_num)))


def plot_bar_of_each_cat_cnt_with_txt():
    df_category_id_cnt = pd.read_csv(
        os.path.join(args.txt_save_dir, 'all_{}_cat_cnt_{}cls.csv'.format(args.input_size, args.class_num)),
        index_col=None)
    y = [v for v in df_category_id_cnt['category_count']]
    x = [k for k in df_category_id_cnt['category_id']]
    label = df_category_id_cnt['category'].to_list()
    title = 'All Chips (608)'

    # png_name_txt = 'cat_count_all_{}_with_txt_subplot.png'.format(args.input_size)
    # draw_bar_for_each_cat_cnt_with_txt_subplot(x, y, label, title, args.cat_sample_dir, png_name_txt, rotation=300)
    png_name_txt = 'cat_count_all_{}_with_txt_rotation.png'.format(args.input_size)
    draw_bar_for_each_cat_cnt_with_txt_rotation(x, y, label, title, args.cat_sample_dir, png_name_txt, rotation=300,
                                                txt_rotation=45)
    # png_name = 'cat_count_all_original.png'
    # draw_bar_for_each_cat_cnt(x, y, label, title, args.fig_save_dir, png_name, rotation=0)


def get_cat_names_by_cat_ids(cat_ids):

    img_ids_names_file = args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num,
                                                                                          args.class_num)
    df_cat_id_names = pd.read_csv(img_ids_names_file, delimiter='\t')
    category_names = df_cat_id_names['category'].tolist()
    category_ids = df_cat_id_names['category_id'].tolist()
    result_cat_names = []
    for c in cat_ids:
        result_cat_names.append(category_names[category_ids.index(c)])
    return result_cat_names


def copy_lbl_to_lbl_model(typstr='val'):
    if comments:
        df_val_img = pd.read_csv(os.path.join(args.data_save_dir, comments[1:], 'xview{}_img{}.txt'.format(typstr, comments)), header=None)
        df_val_gt = pd.read_csv(os.path.join(args.data_save_dir, comments[1:], 'xview{}_lbl{}.txt'.format(typstr, comments)), header=None)
    else:
        df_val_img = pd.read_csv(args.data_save_dir + 'xview{}_img{}.txt'.format(typstr, comments), header=None)
        df_val_gt = pd.read_csv(args.data_save_dir + 'xview{}_lbl{}.txt'.format(typstr, comments), header=None)
    val_lbl_list = df_val_gt[0].tolist()

    save_lbl_dir = os.path.join(args.txt_save_dir, 'lbl_with_model_id', typstr)
    if not os.path.exists(save_lbl_dir):
        os.makedirs(save_lbl_dir)
        for i in range(len(val_lbl_list)):
            shutil.copy(val_lbl_list[i], os.path.join(save_lbl_dir, os.path.basename(val_lbl_list[i])))


def get_raw_train_tifs(px_thres=23, whr_thres=3, seed=17):
    tif_list = []
    df_trn = pd.read_csv( os.path.join(args.data_save_dir,
                                       'px{}whr{}_seed{}/xviewtrain_img_px{}whr{}_seed{}.txt'.format(px_thres, whr_thres, seed, px_thres, whr_thres, seed)), header=None)
    for f in df_trn.loc[:, 0]:
        name = os.path.basename(f)
        name = name.split('_')[0] + '.tif'
        tif_list.append(name)
    tif_list = list(set(tif_list))

    src_path = args.image_folder
    dst_path = args.image_folder[:-1] + '_train/'
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    for t in tif_list:
        shutil.copy(os.path.join(src_path, t), os.path.join(dst_path, t))


def get_new_background_images_based_on_cropped_tiff():
    src_path = args.image_folder[:-1] + '_modify/'
    des_path = args.image_folder[:-1] + '_modify_patches/'
    if not os.path.exists(des_path):
        os.mkdir(des_path)
    src_images = glob.glob(os.path.join(src_path, '*.tiff'))
    src_images.sort()
    ''' get the width list and height list'''
    # w_list = []
    # h_list = []
    # for f in src_images:
    #     # print(os.path.basename(f))
    #     img = cv2.imread(f)
    #     # print(img.shape) # height, width, chanel
    #     h_list.append(img.shape[0])
    #     w_list.append(img.shape[1])
    # h_arr = np.array(h_list)
    # w_arr = np.array(w_list)
    # # thres = 2190 # h_cnt  39 w_cnt  48
    # thres = 1824 # 608*0.3*3/0.3 # h_cnt  54 w_cnt  56
    # h_cnt = np.sum(h_arr>=thres)
    # w_cnt = np.sum(w_arr>=thres)
    # print('h_cnt ', h_cnt, 'w_cnt ', w_cnt)

    # num_bins = 30
    # plt.hist(h_list, bins=num_bins, label='height', alpha=0.5)
    # plt.hist(w_list, bins=num_bins, label='width', alpha=0.6)
    # plt.legend()
    # plt.show()
    ''' given the threshold for image cropping '''
    thres = 1824
    for f in src_images:
        name = os.path.basename(f)
        img = cv2.imread(f)
        h = img.shape[0]
        w = img.shape[1]
        w_num = round(w/thres)
        h_num = round(h/thres)
        if w_num==0 or h_num==0:
            continue
        for hix in range(h_num):
            for wix in range(w_num):
                hleft = hix*thres
                hright = (hix+1)*thres
                wleft = wix*thres
                wright = (wix+1)*thres
                if hright > h and wright > w:
                    hright = h
                    hleft = h-thres
                    wright = w
                    wleft = w-thres
                elif wright > w:
                    wright = w
                    wleft = w-thres
                elif hright > h:
                    hright = h
                    hleft = h-thres
                crop_img = img[hleft: hright, wleft:wright, :]
                crop_name = name.split('.tiff')[0] + '_{}_{}.png'.format(hix, wix)
                cv2.imwrite(os.path.join(des_path, crop_name), crop_img)


def get_args(px_thres=None, whr_thres=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image_folder", type=str,
    #                     help="Path to folder containing image chips (ie 'Image_Chips/') ",
    #                     default='/media/lab/Yang/data/xView/train_images/')
    parser.add_argument("--image_folder", type=str,
                        help="Path to folder containing image chips (ie 'Image_Chips/') ",
                        default='/media/lab/Yang/data/xView/airplane_tifs/')

    parser.add_argument("--json_filepath", type=str, help="Filepath to GEOJSON coordinate file",
                        default='/media/lab/Yang/data/xView/xView_train.geojson')

    parser.add_argument("--xview_yolo_dir", type=str, help="dir to xViewYOLO",
                        default='/media/lab/Yang/data/xView_YOLO/')

    parser.add_argument("--images_save_dir", type=str, help="to save chip trn val images files",
                        default='/media/lab/Yang/data/xView_YOLO/images/')

    parser.add_argument("--txt_save_dir", type=str, help="to save  related label files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')
    parser.add_argument("--data_list_save_dir", type=str, help="to save selected trn val images and labels",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_cls/data_list/')

    parser.add_argument("--annos_save_dir", type=str, help="to save txt annotation files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')

    parser.add_argument("--ori_lbl_dir", type=str, help="to save original labels files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/original/')

    parser.add_argument("--fig_save_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/figures/')

    parser.add_argument("--data_save_dir", type=str, help="to save data files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_cls/')

    parser.add_argument("--results_dir", type=str, help="to save category files",
                        default='/media/lab/Yang/code/yolov3/result_output/{}_cls/')

    parser.add_argument("--cat_sample_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/')

    # parser.add_argument("--img_bbx_figures_dir", type=str, help="to save figures",
    #                     default='/media/lab/Yang/data/xView_YOLO/cat_samples/{}/{}_cls/img_name_2_gt_bbx_figures/')

    parser.add_argument("--cat_bbx_patches_dir", type=str, help="to split cats bbx patches",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/{}/{}_cls/cat_split_patches/')
    parser.add_argument("--cat_bbx_origins_dir", type=str, help="to split cats bbx patches",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/original/{}_cls/cat_split_origins/')

    parser.add_argument("--val_percent", type=float, default=0.20,
                        help="Percent to split into validation (ie .25 = val set is 25% total)")
    parser.add_argument("--font3", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 10}")
    parser.add_argument("-ft2", "--font2", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 13}")
    parser.add_argument("--class_num", type=int, default=1, help="Number of Total Categories")  # 60  6 1
    parser.add_argument("--seed", type=int, default=17, help="random seed") #fixme ---- 1024 17
    parser.add_argument("--input_size", type=int, default=608, help="Number of Total Categories")  # 300 416

    args = parser.parse_args()
    # if args.class_num == 1:
    #     args.images_save_dir = args.images_save_dir + '{}/'.format(args.input_size, args.class_num)
    # else:
    #     args.images_save_dir = args.images_save_dir + '{}_{}cls/'.format(args.input_size, args.class_num)
    #fixme ----------==--------change
    args.images_save_dir = args.images_save_dir + '{}_{}cls/'.format(args.input_size, args.class_num)
    # args.images_save_dir = args.images_save_dir + '{}_{}cls_reduced/'.format(args.input_size, args.class_num)
    args.annos_new_dir = args.annos_save_dir + '{}/{}_cls_xcycwh_part_new/'.format(args.input_size, args.class_num)
    if px_thres:
        args.annos_save_dir = args.annos_save_dir + '{}/{}_cls_xcycwh_px{}whr{}/'.format(args.input_size, args.class_num, px_thres, whr_thres)
    else:
        args.annos_save_dir = args.annos_save_dir + '{}/{}_cls_xcycwh/'.format(args.input_size, args.class_num)
    args.txt_save_dir = args.txt_save_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)
    args.cat_sample_dir = args.cat_sample_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)
    args.ori_lbl_dir = args.ori_lbl_dir + '{}_cls/'.format(args.class_num)
    args.data_save_dir = args.data_save_dir.format(args.class_num)
    args.results_dir = args.results_dir.format(args.class_num)
    args.data_list_save_dir = args.data_list_save_dir.format(args.input_size, args.class_num)
    if not os.path.exists(args.txt_save_dir):
        os.makedirs(args.txt_save_dir)

    if not os.path.exists(args.annos_new_dir):
        os.makedirs(args.annos_new_dir)

    if not os.path.exists(args.annos_save_dir):
        os.makedirs(args.annos_save_dir)

    if not os.path.exists(args.images_save_dir):
        os.makedirs(args.images_save_dir)

    if not os.path.exists(args.cat_sample_dir):
        os.makedirs(args.cat_sample_dir)

    if not os.path.exists(args.ori_lbl_dir):
        os.makedirs(args.ori_lbl_dir)

    if not os.path.exists(args.data_save_dir):
        os.makedirs(args.data_save_dir)
    if not os.path.exists(args.data_list_save_dir):
        os.makedirs(args.data_list_save_dir)

    args.cat_bbx_patches_dir = args.cat_bbx_patches_dir.format(args.input_size, args.class_num)
    args.cat_bbx_origins_dir = args.cat_bbx_origins_dir.format(args.class_num)
    if not os.path.exists(args.cat_bbx_patches_dir):
        os.makedirs(args.cat_bbx_patches_dir)
    if not os.path.exists(args.cat_bbx_origins_dir):
        os.makedirs(args.cat_bbx_origins_dir)
    return args

'''
Datasets
https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data
_aug: Augmented dataset
'''

if __name__ == "__main__":

    ''' get raw train tifs '''

    # px_thres=23 #20 #30
    # whr_thres=3
    # seed=17
    # args = get_args(px_thres, whr_thres)
    # get_raw_train_tifs(px_thres, whr_thres, seed)

    '''
    cats_split_crop_bbx_from_origins for val images
    '''
    # args = get_args(px_thres=None, whr_thres=None)
    # whr_thres = 3
    # px_thres = 23
    # # imgN = 1000 # show imgN images 50
    # # N = 1000 # each image select N samples 20
    # imgN = 50
    # N = 20
    # cats_split_crop_bbx_from_origins(px_thres, whr_thres, imgN, N)

    # cat_ids = np.arange(0, 6).tolist()
    # cid_wh_maps = json.load(open(os.path.join(args.ori_lbl_dir, 'xView_cid_2_wh_maps.json')))
    # draw_wh_scatter_for_cats(cid_wh_maps, cat_ids)

    '''
    plot origin tif with bbox
    '''
    # args = get_args(px_thres=None, whr_thres=None)
    # # img_name = '525.tif'
    # # img_name = '1127.tif'
    # img_name = '1164.tif'
    # plot_image_with_gt_bbx_by_image_name_from_origin(img_name)

    '''
    airplane raw images
    # bad images 807.tif  983.tif 996.tif 2503.tif 2514.tif 2524.tif 2523.tif
    # part cloud 1164.tif 2215.tif 2230.tif 2239.tif  2531.tif 345.tif  847.tif  2110.tif 2399.tif 
    half black  1154.tif  2309.tif 2513.tif  2532.tif
    '''
    # get_raw_airplane_tifs()

    '''
     remove bad tifs ***** /media/lab/Yang/data/xView/airplane_bad_raw_tif_names.txt
     '''
    # args = get_args()
    # bad_img_path = '/media/lab/Yang/data/xView/airplane_bad_raw_tif_names.txt'
    # src_dir = args.image_folder
    # remove_bad_image(bad_img_path, src_dir)

    '''
    create chips and label txt and get all images json, convert from *.geojson to *.json
    then manually *****  remove bad images according to  /media/lab/Yang/data/xView/airplane_part_occlusion_raw_tif_names.txt
    '''
    # syn=False
    # if syn:
    #     args = pps.get_syn_args()
    # else:
    #     args = get_args()
    # create_chips_and_txt_geojson_2_json(args)

    '''
    plot images with bbox from patches 
    '''
    # args = get_args(px_thres=None, whr_thres=None)
    # img_name = ['1164_1.jpg', '1164_2.jpg', '1164_3.jpg','1127_2.jpg', '1127_3.jpg'] #
    # for name in img_name:
    #     plot_image_with_bbox_by_image_name_from_patches(name)

    '''
    remove bad cropped .jpg 
    manually get ***** /media/lab/Yang/data/xView/airplane_removed_cropped_jpg_names.txt
    '''
    # args = get_args()
    # bad_img_path = '/media/lab/Yang/data/xView/airplane_removed_cropped_jpg_names.txt'
    # src_dir = args.images_save_dir
    # remove_bad_image(bad_img_path, src_dir)

    '''
    backup ground truth *.txt 
    remove bbox and annotations of bad cropped .jpg 
    manually get ***** /media/lab/Yang/data/xView/airplane_removed_cropped_jpg_names.txt
    '''
    # px_thres = 23
    # whr_thres = 3
    # if px_thres:
    #     args = get_args(px_thres, whr_thres)
    # else:
    #     args = get_args()
    # bad_img_names = '/media/lab/Yang/data/xView/airplane_removed_cropped_jpg_names.txt'
    # remove_txt_and_json_of_bad_image(bad_img_names, args,px_thres, whr_thres)

    '''
    # backup ground truth *.txt 
    delete bbox that doesn't meet the condition (>px_thres, <whr_thres)
    count ground truth self overlap from patches for each cat 
    '''
    # cat_ids = [0]
    # # # # whr_thres = 4
    # # # # px_thres=6
    # # # cat_ids = [0, 1, 2, 3]
    # whr_thres = 3 # 3.5
    # px_thres= 23
    # iou_thres = 0.5
    # clean_backup_xview_plane_with_constraints(px_thres, whr_thres)
    # cnt_ground_truth_overlap_from_pathces(cat_ids, iou_thres, px_thres, whr_thres)

    '''
    check duplicate bbox of images
    '''
    # image_name = '2160_1.jpg'
    # image_name = '525_1.jpg'
    # # image_name = '525_3.jpg'
    # # image_name = '2399_7.jpg'
    # whr_thres = 3 # 3.5
    # px_thres= 23
    # args = get_args(px_thres, whr_thres)
    # image_file = os.path.join(args.images_save_dir, image_name)
    # args = get_args(px_thres, whr_thres)
    # lbl_file = os.path.join(args.annos_save_dir, image_name.replace('.jpg', '.txt'))
    # save_path = args.cat_sample_dir + 'image_with_bbox/'
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # gbc.plot_img_with_bbx(image_file, lbl_file, save_path)


    '''
    remove duplicate ground truth bbox by cat_id
    ** manually replace old overlap bbox with new lbl**** 
    /media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_part_new
    '''
    #fixme Not Done
    # cat_id = 0
    # px_thres = 23
    # whr_thres = 3
    # args = get_args(px_thres, whr_thres)
    # remove_duplicate_gt_bbx(cat_id)

    '''
    plot bbox on images
    deviation
    1127_2.jpg 1127_3.jpg 1154_0.jpg  1164_1.jpg 1164_2.jpg 1164_3.jpg  1568_5.jpg 1568_6.jpg 2399_2.jpg 2399_6.jpg 2399_7.jpg 2513_1.jpg 
    2530_2.jpg 2530_3.jpg 2530_4.jpg 2531_10.jpg 79_3.jpg 80_5.jpg 294_2.jpg 302_1.jpg 310_2.jpg 
    1585_7.jpg 1587_3.jpg 1593_9.jpg 1593_10.jpg 1883_4.jpg 1183_5.jpg 2250_2.jpg  2309_2.jpg 2315_6.jpg  2341_5.jpg 2341_6.jpg 
    2370_1.jpg 2391_4.jpg
   
    342_0.jpg 1076_1.jpg 
    
    discard 1590_5.jpg 1606_2.jpg 1610_1.jpg 1692_3.jpg 1899_1.jpg 1899_2.jpg 2064_6.jpg 88_1.jpg  1076_5.jpg
    2239.tif
    '''
    # whr_thres = 3 # 3.5
    # px_thres= 23
    # args = get_args(px_thres, whr_thres)
    # save_path = args.cat_sample_dir + 'image_with_bbox/'
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # img_list = np.sort(glob.glob(os.path.join(args.images_save_dir, '*.jpg')))
    # for img in img_list:
    #     lbl_name = os.path.basename(img).replace('.jpg', '.txt')
    #     lbl_file = os.path.join(args.annos_save_dir, lbl_name)
    #     gbc.plot_img_with_bbx(img, lbl_file, save_path, label_index=False)

    '''
    re-generate *xtlytlwh.json with empty label files 
    '''
    # px_thres = 23
    # whr_thres = 3
    # adjust_empty_annos_json(px_thres, whr_thres)

    '''
    create xview.names 
    '''
    # file_name = 'xview'
    # create_xview_names(file_name)

    '''
    split train:val randomly split chips
    default split
    '''
    # comments = ''
    # comments = '_px4whr3'
    # comments = '_px6whr4_giou0'
    # px_thres = 23
    # whr_thres = 3
    # comments = '_px23whr3_seed17'
    # data_name = 'xview'
    # seed=17
    # split_trn_val_with_chips(data_name, comments, seed = 17, px_thres=px_thres, whr_thres=whr_thres)

    '''
    create json for val according to all jsons 
    '''
    # # create_json_for_val_according_to_all_json()
    # typestr='val'
    # px_thres = 23
    # whr_thres = 3
    # seed=17
    # comments = '_px{}whr{}_seed{}'.format(px_thres, whr_thres, seed)
    # data_name = 'xview'
    # create_json_for_train_or_val_according_to_all_json(data_name, comments, seed, px_thres, whr_thres, typestr)

    '''
    split train:val  randomly split tifs
    '''
    # split_trn_val_with_tifs()

    '''
    create xview.data includes cls_num, trn, val, ...
    '''
    # create_xview_data()

    '''
    categories count for all label files
    and plot category count
    '''
    # px_thres = 23
    # whr_thres = 3
    # args = get_args(px_thres, whr_thres)
    # categories_cnt_for_all()
    # plot_bar_of_each_cat_cnt_with_txt()

    '''
    cats_split_crop_bbx_from_patches for val images
    '''
    # typestr = 'val'
    # whr_thres = 3
    # N = 200 # each category select N samples
    # cats_split_crop_bbx_from_patches(typestr, whr_thres, N)


    '''
    IoU check by image name
    '''
    # # # # image_name = '310_12.jpg'
    # # # # image_name = '1090_3.jpg'
    # # image_name = '1139_4.jpg'
    # image_name = '2318_1.jpg'

    # score_thres = 0.3
    # iou_thres = 0.5
    # check_prd_gt_iou(image_name, score_thres, iou_thres, args.rare)

    '''
    get cat ids to img names maps
    '''
    # cat_ids = ['0', '1', '2', '3', '4', '5']
    # # # cat_ids = np.arange(0, 6).tolist()
    # # typestr='train'
    # typestr='val'
    # get_train_or_val_imgs_by_cat_id(cat_ids, typestr)

    '''
    val results confusion matrix val results statistic FP FN NMS
    '''
    # cat_ids = np.arange(0, 6).tolist()
    # cat_ids = [0]
    # score_thres = 0.3
    # whr_thres = 3
    # iou_thres = 0.5
    # args.rare = False
    # confusion_matrix = get_confusion_matrix(cat_ids, iou_thres, score_thres, whr_thres)
    # summary_confusion_matrix(confusion_matrix, cat_ids, iou_thres, args.rare)

    # cat_ids = np.arange(0, 6).tolist()
    # iou_thres = 0.5
    # confusion_matrix = np.load(args.results_dir + 'result_confusion_matrix_iou{}/confusion_matrix.npy'.format(iou_thres))
    # plot_confusion_matrix(confusion_matrix, cat_ids, iou_thres, args.rare)

    '''
    save results of fp fn with bbox separately 
    '''
    # cat_ids = np.arange(0, 6).tolist()
    # # cat_ids = [0]
    # score_thres = 0.3
    # whr_thres = 3
    # iou_thres = 0.5
    # get_fp_fn_separtely_by_cat_ids(cat_ids, iou_thres=0.5, score_thres=0.3, whr_thres=3)

    '''
    plot img with prd bbox by cat id
    '''
    # score_thres = 0.3
    # whr_thres = 3
    # typestr='val'
    # cat_id = '0'
    # plot_img_with_prd_bbx_by_cat_id(cat_id, typestr, score_thres, whr_thres)

    '''
    plot prd bbox with object score and gt bbox
    '''
    # cat_ids = np.arange(0, 6).tolist()
    # cat_ids = [0]
    # score_thres = 0.3
    # iou_thres=0.5
    # whr_thres=3
    # plot_prd_gt_bbox_by_cat_ids(cat_ids, score_thres, whr_thres)

    '''
    get prd bbx by image name and cat id
    '''
    # score_thres = 0.3
    # whr_thres = 3
    # cat_ids = [0]
    # image_name = '1139_4.jpg'
    # image_name = '2160_1.jpg'
    # plot_img_with_prd_bbx_by_image_name(image_name, cat_ids, score_thres, whr_thres)

    '''
    plot img with gt bbx by img name from patches
    '''
    # image_name = '1139_4.jpg'
    # image_name = '5_7.jpg'
    # # image_name = '2230_3.jpg'
    # image_name = '525_20.jpg'
    # image_name = '1139_4.jpg'
    # image_name = '525_15.jpg'
    # image_name = '2318_1.jpg'
    # # # typestr = 'val'
    # typestr = 'train'
    # plot_val_image_with_gt_bbx_by_image_name_from_patches(image_name, typestr)

    # typestr = 'val'
    # # typestr = 'train'
    # comments = '_px23whr3_seed17'
    # px_thres = 23
    # whr_thres = 3
    # args = get_args(px_thres, whr_thres)
    # plot_val_image_with_gt_bbx_by_image_name_from_patches(None, typestr, comments)

    '''
    copy_lbl_to_lbl_model
    '''
    # typstr = 'train'
    # typstr = 'val'
    # comments = '_px23whr3_seed17'
    # px_thres = 23
    # whr_thres = 3
    # args = get_args(px_thres, whr_thres)
    # copy_lbl_to_lbl_model(typstr)

    '''
    get img id by image_name
    '''
    # image_name = '2230_3.jpg' # 6121
    # image_name = '145_16.jpg' # 2609
    # image_name = '1154_6.jpg' # 1238
    # image_name = '1076_16.jpg' # 489
    # image_name = '2294_5.jpg' #6245
    # image_name = '1568_0.jpg' # 3160
    # image_name='1139_4.jpg' # 1124
    # image_id = get_img_id_by_image_name(image_name)
    # print(image_id)

    '''
    IoU check by image name
    '''
    # score_thres = 0.5
    # iou_thres = 0.5
    # image_name = '310_12.jpg'
    # args.rare = False
    # check_prd_gt_iou(image_name, score_thres, iou_thres, args.rare)

    '''
    check the the origin 60 classes: see if one object is assigned to  two different classes
    '''
    # iou_thres = 0.5
    # whr_thres = 3
    # px_thres = 4
    # cat_ids = np.arange(0, 6).tolist()
    # for cat_id in cat_ids:
    #     check_duplicate_gt_bbx_for_60_classes(cat_id, iou_thres, px_thres, whr_thres)


    '''
    get class names by cat_id
    '''
    # cat_ids = np.arange(0, 6).tolist()
    # cat_names = get_cat_names_by_cat_ids(cat_ids)
    # print(rare_cat_names)

    '''
    crop patches per bbox from 608x608 patches
    '''
    # cat_ids = np.arange(0, 6).tolist()
    # cat_names = ['Aircraft', 'Motor Vehicles', 'Railway', 'Maritime', 'Construction Vehicles', 'Fixed Facilities']
    # # typestr = 'train'
    # typestr = 'val'
    # cats_split_crop_bbx_from_patches(cat_ids, cat_names, typestr)

    '''
    crop patches per bbox from tiles 
    '''
    # cat_ids = np.arange(0, 6).tolist()
    # rare_cat_names = ['Fixed-Wing Aircraft', 'Straddle Carrier', 'Helicopter', 'Small Aircraft', 'Passenger/Cargo Plane', 'Yacht']
    # rare_cats_split_crop_bbx_from_tiles(rare_cat_ids, rare_cat_names)
    #
    # clabel_wh_maps = json.load(open(os.path.join(args.ori_lbl_dir, 'xView_rare_clabel_2_wh_maps.json')))
    # draw_wh_scatter_for_rare_cats(clabel_wh_maps, rare_cat_ids, rare_cat_names)

    ''' 
    crop tiff images that not contain aircrafts
    get new background images
    '''
    px_thres = 23
    whr_thres = 3
    args = get_args(px_thres, whr_thres)
    get_new_background_images_based_on_cropped_tiff()



