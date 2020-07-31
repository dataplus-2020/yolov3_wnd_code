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
import numpy as np
import argparse
import os
import pandas as pd
from ast import literal_eval
import json
from matplotlib import pyplot as plt
import cv2
import seaborn as sn
import sys
sys.path.append('./')
from utils.utils_xview import coord_iou
from utils.data_process_distribution_vis_util.process_wv_coco_for_yolo_patches_no_trnval import draw_bar_for_each_cat_cnt_with_txt_rotation
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
    dh = 1. / (size[0]) # h--0--y
    dw = 1. / (size[1]) # w--1--x

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

def get_val_imgs_by_rare_cat_id(rare_cat_ids, typestr='all'):
    cat_img_ids_maps = json.load(open(os.path.join(args.txt_save_dir, '{}_cat_img_ids_dict_{}cls.json'.format(typestr, args.class_num))))
    img_ids_names_maps = json.load(open(os.path.join(args.txt_save_dir, '{}_image_ids_names_dict_{}cls.json'.format(typestr, args.class_num))))
    imgids = [k for k in img_ids_names_maps.keys()]
    imgnames = [v for v in img_ids_names_maps.values()]

    val_rare_img_txt = open(os.path.join(args.data_save_dir, 'xviewval_rare_img.txt'), 'w')
    val_rare_lbl_txt = open(os.path.join(args.data_save_dir, 'xviewval_rare_lbl.txt'), 'w')
    img_path = args.images_save_dir
    lbl_path = args.txt_save_dir + '{}_cls_xcycwh/'.format(args.class_num)

    val_files = pd.read_csv(os.path.join(args.data_save_dir, 'xviewval_img.txt'), header=None)
    val_img_names = [f.split('/')[-1] for f in val_files[0]]
    val_img_ids = [imgids[imgnames.index(n)] for n in val_img_names]
    rare_cat_val_imgs = {}
    #fixme
    # for rc in rare_cat_ids:
    #     cat_img_ids = cat_img_ids_maps[rc]
    #     rare_cat_all_imgs_files = []
    #     for c in cat_img_ids:
    #         rare_cat_all_imgs_files.append(img_ids_names_maps[str(c)]) # rare cat id map to all imgs
    #     rare_cat_val_imgs[rc] = [v for v in rare_cat_all_imgs_files if v in val_img_names] # rare cat id map to val imgs
    #     print('cat imgs in val', rare_cat_val_imgs[rc])
    #     print('cat {} total imgs {}, val imgs {}'.format(rc, len(rare_cat_all_imgs_files), len(rare_cat_val_imgs)))
    #     for rimg in rare_cat_val_imgs[rc]:
    #         val_rare_img_txt.write("%s\n" % (img_path + rimg))
    #         val_rare_lbl_txt.write("%s\n" % (lbl_path + rimg.replace('.jpg', '.txt')))
    # rare_cat_val_imgs_files = os.path.join(args.txt_save_dir, 'rare_cat_ids_2_val_img_names.json')
    # json.dump(rare_cat_val_imgs, open(rare_cat_val_imgs_files, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)

    rare_all_cats_val_imgs = []
    for rc in rare_cat_ids:
        cat_img_ids = cat_img_ids_maps[rc]
        cat_img_names = [val_img_names[val_img_ids.index(str(i))] for i in cat_img_ids if str(i) in val_img_ids]
        rare_all_cats_val_imgs.extend(cat_img_names)
        rare_cat_val_imgs[rc] = cat_img_names
    # print(len(rare_all_cats_val_imgs))
    rare_all_cats_val_imgs = list(set(rare_all_cats_val_imgs))
    # print(len(rare_all_cats_val_imgs))
    for rimg in rare_all_cats_val_imgs:
        val_rare_img_txt.write("%s\n" % (img_path + rimg))
        val_rare_lbl_txt.write("%s\n" % (lbl_path + rimg.replace('.jpg', '.txt')))
    rare_cat_val_imgs_files = os.path.join(args.txt_save_dir, 'rare_cat_ids_2_val_img_names.json')
    json.dump(rare_cat_val_imgs, open(rare_cat_val_imgs_files, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)

def create_json_for_val_rare_according_to_all_json():
    json_all_file = os.path.join(args.txt_save_dir, 'xViewall_{}_{}cls_xtlytlwh.json'.format(args.input_size, args.class_num)) # topleft
    all_json = json.load(open(json_all_file))
    all_imgs_info = all_json['images']
    all_annos_info = all_json['annotations']
    all_cats_info = all_json['categories']

    all_img_id_maps = json.load(open(args.txt_save_dir + 'all_image_ids_names_dict_{}cls.json'.format(args.class_num)))
    all_imgs = [i for i in all_img_id_maps.values()]
    all_img_ids = [int(i) for i in all_img_id_maps.keys()]

    val_img_files = pd.read_csv(os.path.join(args.data_save_dir, 'xviewval_rare_img.txt'), header=None)
    val_img_names = [f.split('/')[-1] for f in val_img_files[0].tolist()]
    val_img_ids = [all_img_ids[all_imgs.index(v)] for v in val_img_names]

    image_info_list = []
    anno_info_list = []

    for ix in val_img_ids:
        img_name = all_imgs[ix]
        # for i in all_imgs_info:
        #     if all_imgs_info[i]['id'] == ix:
        #fixme: the index of all_imgs_info is the same as the image_info['id']
        image_info_list.append(all_imgs_info[ix])
        for ai in all_annos_info:
            if ai['image_id'] == ix:
                anno_info_list.append(ai)

    trn_instance = {'info': 'xView val rare chips 600 yx185 created', 'license': ['license'], 'images': image_info_list,
                    'annotations': anno_info_list, 'categories': all_cats_info}
    json_file = os.path.join(args.txt_save_dir, 'xViewval_rare_{}_{}cls_xtlytlwh.json'.format(args.input_size, args.class_num)) # topleft
    json.dump(trn_instance, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)


def plot_val_image_with_bbx_by_image_name(image_name):
    img_ids_names_file = '/media/lab/Yang/data/xView_YOLO/labels/608/all_image_ids_names_dict_60cls.json'
    img_ids_names_map = json.load(open(img_ids_names_file))
    img_ids = [int(k) for k in img_ids_names_map.keys()] ## important
    img_names = [v for v in img_ids_names_map.values()]

    df_cat_color = pd.read_csv('../../data_xview/60_cls/categories_id_color_diverse_60.txt', delimiter='\t')
    cat_ids = df_cat_color['category_id'].tolist()
    cat_colors = df_cat_color['color'].tolist()

    df_val_img = pd.read_csv('../../data_xview/60_cls/xviewval_img.txt', header=None)
    df_val_gt = pd.read_csv('../../data_xview/60_cls/xviewval_lbl.txt', header=None)
    val_img_list = df_val_img[0].tolist()
    img_size = 608

    val_img_names = [f.split('/')[-1] for f in val_img_list]
    img_index = val_img_names.index(image_name)
    gt = pd.read_csv(df_val_gt[0].iloc[img_index], header=None, delimiter=' ')
    gt.iloc[:, 1:] = gt.iloc[:, 1:]*img_size
    gt.iloc[:, 1] = gt.iloc[:, 1] - gt.iloc[:, 3]/2
    gt.iloc[:, 2] = gt.iloc[:, 2] - gt.iloc[:, 4]/2

    img_bbx_fig_dir = args.cat_sample_dir + 'img_name_2_bbx_figures/'
    if not os.path.exists(img_bbx_fig_dir):
        os.mkdir(img_bbx_fig_dir)

    fig = plt.figure(figsize=(10, 10))
    img = cv2.imread(val_img_list[img_index])
    for ix in range(len(gt)):
        cat_id = gt.iloc[ix, 0]
        color = literal_eval(cat_colors[cat_ids.index(cat_id)])
        gt_bbx = gt.iloc[ix, 1:].to_numpy()
        gt_bbx = gt_bbx.astype(np.int64)
        img = cv2.rectangle(img, (gt_bbx[0], gt_bbx[1]), (gt_bbx[0] + gt_bbx[2], gt_bbx[1] + gt_bbx[3]), color, 2)
        # cv2.putText(img, text=str(cat_id), org=(gt_bbx[0] + 10, gt_bbx[1] + 10),
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))
    cv2.imwrite(img_bbx_fig_dir + image_name, img)


def get_img_id_by_image_name(image_name):
    img_ids_names_file = '/media/lab/Yang/data/xView_YOLO/labels/608/all_image_ids_names_dict_60cls.json'
    img_ids_names_map = json.load(open(img_ids_names_file))
    img_ids = [int(k) for k in img_ids_names_map.keys()] ## important
    img_names = [v for v in img_ids_names_map.values()]
    return img_ids[img_names.index(image_name)]

#fixme
# def plot_rare_results_by_rare_cat_ids(rare_cat_ids, score_thres=0.001):
#     '''
#     all: TP + FP + FN
#     '''
#     img_ids_names_file = '/media/lab/Yang/data/xView_YOLO/labels/{}/all_image_ids_names_dict_{}cls.json'.format(args.input_size, args.class_num)
#     img_ids_names_map = json.load(open(img_ids_names_file))
#     img_ids = [int(k) for k in img_ids_names_map.keys()] ## important
#     img_names = [v for v in img_ids_names_map.values()]
#
#     df_rare_val_img = pd.read_csv('../data_xview/{}_cls/xviewval_rare_img.txt'.format(args.class_num), header=None)
#     df_rare_val_gt = pd.read_csv('../data_xview/{}_cls/xviewval_rare_lbl.txt'.format(args.class_num), header=None)
#     rare_val_img_list = df_rare_val_img[0].tolist()
#     rare_val_id_list = df_rare_val_gt[0].tolist()
#     rare_val_img_names = [f.split('/')[-1] for f in rare_val_img_list]
#     rare_val_img_ids = [img_ids[img_names.index(x)] for x in rare_val_img_names]
#
#     rare_result_json_file = '../result_output/{}_cls/results_rare.json'.format(args.class_num)
#     rare_result_allcat_list = json.load(open(rare_result_json_file))
#     rare_result_list = []
#     for ri in rare_result_allcat_list:
#         if ri['category_id'] in rare_cat_ids and ri['score'] >= score_thres:
#             rare_result_list.append(ri)
#     del rare_result_allcat_list
#
#     prd_color = (255, 255, 0)
#     gt_color = (0, 255, 255) # yellow
#     args.txt_save_dir = args.txt_save_dir + '{}_cls_xcycwh/'.format(args.class_num)
#     save_dir = '../result_output/rare_result_figures/'
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
#
#     for ix in range(len(rare_val_img_list)):
#         img = cv2.imread(rare_val_img_list[ix])
#         img_size = img.shape[0]
#         gt_rare_cat = pd.read_csv(rare_val_id_list[ix], delimiter=' ').to_numpy()
#         gt_rare_list = []
#         # for cat_id in rare_cat_ids:
#         #     for gx in range(gt_rare_cat.shape[0]):
#         #         if gt_rare_cat[gx, 0] == cat_id:
#         #             gt_rare_cat[gx, 1:] = gt_rare_cat[gx, 1:] * img_size
#         #             gt_rare_cat[gx, 1] = gt_rare_cat[gx, 1] - gt_rare_cat[gx, 3]/2
#         #             gt_rare_cat[gx, 2] = gt_rare_cat[gx, 2] - gt_rare_cat[gx, 4]/2
#         #             gt_bbx = gt_rare_cat[gx, 1:]
#         #             gt_bbx = gt_bbx.astype(np.int64)
#         #             img = cv2.rectangle(img, (gt_bbx[0], gt_bbx[1]), (gt_bbx[0] + gt_bbx[2], gt_bbx[1] + gt_bbx[3]), gt_color, 2)
#         #             cv2.putText(img, text=str(cat_id), org=(gt_bbx[0] - 10, gt_bbx[1] - 10),
#         #                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#         #                         fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=gt_color)
#         #             gt_rare_list.append(gx)
#         #     img_prd_lbl_list = []
#         #     for jx in rare_result_list:
#         #         if jx['category_id'] == cat_id and jx['image_id'] == rare_val_img_ids[ix]:
#         #             img_prd_lbl_list.append(jx) # rare_result_list.pop()
#         #             bbx = jx['bbox'] # xtopleft, ytopleft, w, h
#         #             bbx = [int(x) for x in bbx]
#         #             img = cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[0] + bbx[2], bbx[1] + bbx[3]), prd_color, 2)
#         #             cv2.putText(img, text=str(jx['category_id']), org=(bbx[0] + 10, bbx[1] + 20),
#         #                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#         #                         fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=prd_color)
#         #     if gt_rare_list or img_prd_lbl_list:
#         #         cv2.imwrite(save_dir + 'cat{}_'.format(cat_id) + rare_val_img_names[ix], img)
#         for gx in range(gt_rare_cat.shape[0]):
#             cat_id = int(gt_rare_cat[gx, 0])
#             if cat_id in rare_cat_ids:
#                 gt_rare_cat[gx, 1:] = gt_rare_cat[gx, 1:] * img_size
#                 gt_rare_cat[gx, 1] = gt_rare_cat[gx, 1] - gt_rare_cat[gx, 3]/2
#                 gt_rare_cat[gx, 2] = gt_rare_cat[gx, 2] - gt_rare_cat[gx, 4]/2
#                 gt_bbx = gt_rare_cat[gx, 1:]
#                 gt_bbx = gt_bbx.astype(np.int64)
#                 img = cv2.rectangle(img, (gt_bbx[0], gt_bbx[1]), (gt_bbx[0] + gt_bbx[2], gt_bbx[1] + gt_bbx[3]), gt_color, 2)
#                 cv2.putText(img, text=str(cat_id), org=(gt_bbx[0] - 10, gt_bbx[1] - 10),
#                             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                             fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=gt_color)
#                 gt_rare_list.append(gx)
#         img_prd_lbl_list = []
#         for jx in rare_result_list:
#             jx_cat_id = jx['category_id']
#             if jx['image_id'] == rare_val_img_ids[ix] and jx_cat_id in rare_cat_ids:
#                 img_prd_lbl_list.append(jx) # rare_result_list.pop()
#                 bbx = jx['bbox'] # xtopleft, ytopleft, w, h
#                 bbx = [int(x) for x in bbx]
#                 img = cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[0] + bbx[2], bbx[1] + bbx[3]), prd_color, 2)
#                 cv2.putText(img, text=str(jx_cat_id), org=(bbx[0] + 10, bbx[1] + 20),
#                             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                             fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=prd_color)
#
#         if gt_rare_list or img_prd_lbl_list:
#             cv2.imwrite(save_dir + 'rare_cat_' + rare_val_img_names[ix], img)


def plot_rare_results_by_rare_cat_ids(rare_cat_ids, iou_thres=0.5, score_thres=0.3, whr_thres=3):
    '''
    all: TP + FP + FN
    '''
    img_ids_names_file = '/media/lab/Yang/data/xView_YOLO/labels/{}/all_image_ids_names_dict_{}cls.json'.format(args.input_size, args.class_num)
    img_ids_names_map = json.load(open(img_ids_names_file))
    img_ids = [int(k) for k in img_ids_names_map.keys()] ## important
    img_names = [v for v in img_ids_names_map.values()]

    df_rare_val_img = pd.read_csv(args.data_save_dir + 'xviewval_rare_img.txt'.format(args.class_num), header=None)
    df_rare_val_gt = pd.read_csv(args.data_save_dir + 'xviewval_rare_lbl.txt'.format(args.class_num), header=None)
    rare_val_img_list = df_rare_val_img[0].tolist()
    rare_val_id_list = df_rare_val_gt[0].tolist()
    rare_val_img_names = [f.split('/')[-1] for f in rare_val_img_list]
    rare_val_img_ids = [img_ids[img_names.index(x)] for x in rare_val_img_names]
    print('gt len', len(rare_val_img_names))

    rare_result_json_file = args.results_dir + '{}_cls/results_rare.json'.format(args.class_num)
    rare_result_allcat_list = json.load(open(rare_result_json_file))
    rare_result_list = []
    for ri in rare_result_allcat_list:
        if ri['category_id'] in rare_cat_ids and ri['score'] >= score_thres:
            rare_result_list.append(ri)
    print('len rare_result_list', len(rare_result_list))
    del rare_result_allcat_list

    prd_color = (255, 255, 0)
    gt_color = (0, 255, 255) # yellow
    args.txt_save_dir = args.txt_save_dir + '{}_cls_xcycwh/'.format(args.class_num)
    save_dir = args.results_dir + 'rare_result_figures/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for ix in range(len(rare_val_img_list)):
        img = cv2.imread(rare_val_img_list[ix])
        img_size = img.shape[0]
        gt_rare_cat = pd.read_csv(rare_val_id_list[ix], delimiter=' ').to_numpy()
        gt_rare_list = []

        for gx in range(gt_rare_cat.shape[0]):
            cat_id = int(gt_rare_cat[gx, 0])
            if cat_id in rare_cat_ids:
                gt_rare_cat[gx, 1:] = gt_rare_cat[gx, 1:] * img_size
                gt_rare_cat[gx, 1] = gt_rare_cat[gx, 1] - gt_rare_cat[gx, 3]/2
                gt_rare_cat[gx, 2] = gt_rare_cat[gx, 2] - gt_rare_cat[gx, 4]/2
                w = gt_rare_cat[gx, 3]
                h = gt_rare_cat[gx, 4]
                whr = np.maximum(w/(h+1e-16), h/(w+1e-16))
                if whr > whr_thres or w < 4 or h < 4:
                    continue
                gt_rare_cat[gx, 3] = gt_rare_cat[gx, 3] + gt_rare_cat[gx, 1]
                gt_rare_cat[gx, 4] = gt_rare_cat[gx, 4] + gt_rare_cat[gx, 2]
                gt_rare_list.append(gt_rare_cat[gx, :])

        img_prd_lbl_list = []
        for jx in rare_result_list:
            jx_cat_id = jx['category_id']
            if jx['image_id'] == rare_val_img_ids[ix] and jx_cat_id in rare_cat_ids:
                bbx = jx['bbox'] # xtopleft, ytopleft, w, h
                whr = np.maximum(bbx[2]/(bbx[3]+1e-16), bbx[3]/(bbx[2]+1e-16))
                if whr > whr_thres or bbx[2] < 4 or bbx[3] < 4 or jx['score'] < score_thres:
                    continue
                pd_rare = [jx_cat_id]
                bbx[2] = bbx[0] + bbx[2]
                bbx[3] = bbx[1] + bbx[3]
                bbx = [int(b) for b in bbx]
                pd_rare.extend(bbx)
                pd_rare.append(jx['score'])
                img_prd_lbl_list.append(pd_rare)

        matches = []
        gt_list = []
        pr_list = []
        for gx, gt in enumerate(gt_rare_list):
            gt_bx = gt[1:]
            match = False
            for px, pr in enumerate(img_prd_lbl_list):
                pr_bx = pr[1:-1]
                iou = coord_iou(gt_bx, pr_bx)
                if iou > iou_thres:
                    matches.append([gx, px, iou])
                    match = True

            if not match:
                gt_list.append(gt_rare_list[gx])

        matches = np.array(matches)
        if matches.shape[0] > 0:
            matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            pr_list.extend([img_prd_lbl_list[int(p)] for p in matches[:, 1]]) # note extend not append list
            gt_list.extend([gt_rare_list[int(g)] for g in matches[:, 0]])

        for pr in pr_list:
            # print(pr)
            bbx = pr[1:-1]
            img = cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[2], bbx[3]), prd_color, 2)
            cv2.putText(img, text=str(int(pr[0])), org=(bbx[0] - 5, bbx[1] + 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=prd_color)
        for gt in gt_list:
            gt_bbx = gt[1:]
            gt_bbx = [int(g) for g in gt_bbx]
            img = cv2.rectangle(img, (gt_bbx[0], gt_bbx[1]), (gt_bbx[2], gt_bbx[3]), gt_color, 2)
            cv2.putText(img, text=str(int(gt[0])), org=(gt_bbx[2] - 20, gt_bbx[3] - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=gt_color)
        # if gt_list or pr_list:
        cv2.imwrite(save_dir + 'rare_cat_' + rare_val_img_names[ix], img)


def check_prd_gt_iou(image_name, score_thres=0.3, iou_thres=0.5):
    img_id = get_img_id_by_image_name(image_name)
    img = cv2.imread(args.images_save_dir + image_name)
    img_size = img.shape[0]
    gt_rare_cat = pd.read_csv(args.txt_save_dir + '{}_cls_xcycwh/'.format(args.class_num) + image_name.replace('.jpg', '.txt'), header=None, delimiter=' ')
    gt_rare_cat = gt_rare_cat.to_numpy()
    gt_rare_cat[:, 1:] = gt_rare_cat[:, 1:] * img_size
    gt_rare_cat[:, 1] = gt_rare_cat[:, 1] - gt_rare_cat[:, 3]/2
    gt_rare_cat[:, 2] = gt_rare_cat[:, 2] - gt_rare_cat[:, 4]/2
    gt_rare_cat[:, 3] = gt_rare_cat[:, 1] + gt_rare_cat[:, 3]
    gt_rare_cat[:, 4] = gt_rare_cat[:, 2] + gt_rare_cat[:, 4]

    prd_lbl_rare = json.load(open('../result_output/{}_cls/results_rare.json'.format(args.class_num))) # xtlytlwh
    for px, p in enumerate(prd_lbl_rare):
        if p['image_id'] == img_id and p['score'] > score_thres:
            p_bbx = p['bbox']
            p_bbx[2] = p_bbx[0] + p_bbx[2]
            p_bbx[3] = p_bbx[3] + p_bbx[1]
            p_bbx = [int(x) for x in p_bbx]
            p_cat_id = p['category_id']
            g_lbl_part = gt_rare_cat[gt_rare_cat[:, 0] == p_cat_id, :]
            for g in g_lbl_part:
                g_bbx = [int(x) for x in g[1:]]
                iou = coord_iou(p_bbx, g[1:])
                # print('iou', iou)
                if iou >= iou_thres:
                    print(iou)
                    img = cv2.rectangle(img, (p_bbx[0], p_bbx[1]), (p_bbx[2], p_bbx[3]), (0, 255, 255), 2)
                    cv2.putText(img, text=str([p_cat_id, iou]), org=(p_bbx[0] + 10, p_bbx[1] + 10), # [pr_bx[0], pr[-1]]
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 0))
                    img = cv2.rectangle(img, (g_bbx[0], g_bbx[1]), (g_bbx[2], g_bbx[3]), (255, 255, 0), 2)
    rare_result_iout_check_dir = args.results_dir + 'rare_result_iou_check/'
    if not os.path.exists(rare_result_iout_check_dir):
        os.mkdir(rare_result_iout_check_dir)
    cv2.imwrite(rare_result_iout_check_dir + image_name, img)

def get_confusion_matrix(rare_cat_ids, iou_thres=0.5, score_thres=0.3, whr_thres=3):
    '''
    https://github.com/svpino/tf_object_detection_cm/blob/83cb8a1cf3a5abd24b18a5fc79b5ce99e8a9b317/confusion_matrix.py#L37
    '''
    img_ids_names_file = '/media/lab/Yang/data/xView_YOLO/labels/{}/all_image_ids_names_dict_{}cls.json'.format(args.input_size, args.class_num)
    img_ids_names_map = json.load(open(img_ids_names_file))
    img_ids = [int(k) for k in img_ids_names_map.keys()] ## important
    img_names = [v for v in img_ids_names_map.values()]

    rare_cat_id_2_img_names_file = args.txt_save_dir + 'rare_cat_ids_2_val_img_names.json'
    rare_cat_id_2_img_names_json = json.load(open(rare_cat_id_2_img_names_file))
    rare_img_name_list = []
    for rc in rare_cat_ids:
        rare_img_name_list.extend(rare_cat_id_2_img_names_json[str(rc)])
    rare_img_name_list = list(set(rare_img_name_list))
    print('len rare images', len(rare_img_name_list))
    rare_img_id_list = [img_ids[img_names.index(n)] for n in rare_img_name_list]

    rare_result_json_file = '../result_output/{}_cls/results_rare.json'.format(args.class_num)
    rare_result_allcat_list = json.load(open(rare_result_json_file))
    rare_result_list = []
    # #fixme filter, and rare_result_allcat_list contains rare_cat_ids, rare_img_id_list and object score larger than score_thres
    for ri in rare_result_allcat_list:
        if ri['category_id'] in rare_cat_ids and ri['score'] >= score_thres: # ri['image_id'] in rare_img_id_list and
            rare_result_list.append(ri)
    print('len rare_result_list', len(rare_result_list))
    del rare_result_allcat_list

    # confusion_matrix = np.zeros(shape=(args.class_num + 1, args.class_num + 1))
    confusion_matrix = np.zeros(shape=(len(rare_cat_ids) + 1, len(rare_cat_ids) + 1))

    fn_color = (255, 0, 0) # Blue
    fp_color = (0, 0, 255) # Red
    save_dir = args.results_dir + 'rare_result_figures_fp_fn_nms_iou{}/'.format(iou_thres)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    confu_mat_dir = '../result_output/result_confusion_matrix_iou{}/'.format(iou_thres)
    if not os.path.exists(confu_mat_dir):
        os.mkdir(confu_mat_dir)
    label_save_dir = args.txt_save_dir + '{}_cls_xcycwh/'.format(args.class_num)

    for ix in range(len(rare_img_id_list)):
        img_id = rare_img_id_list[ix]
    # for img_id in [3160]: # 6121 2609 1238 489 6245
    #     ix = rare_img_id_list.index(img_id)
        img_name = rare_img_name_list[ix]
        # print(img_name)

        ''' ground truth '''
        good_rare_gt_list = []
        df_lbl = pd.read_csv(label_save_dir + img_name.replace('.jpg', '.txt'), delimiter=' ', header=None)
        df_lbl.iloc[:, 1:] = df_lbl.iloc[:, 1:] * args.input_size
        df_lbl.iloc[:, 1] = df_lbl.iloc[:, 1] - df_lbl.iloc[:, 3]/2
        df_lbl.iloc[:, 2] = df_lbl.iloc[:, 2] - df_lbl.iloc[:, 4]/2
        df_lbl.iloc[:, 3] = df_lbl.iloc[:, 1] + df_lbl.iloc[:, 3]
        df_lbl.iloc[:, 4] = df_lbl.iloc[:, 2] + df_lbl.iloc[:, 4]
        df_lbl = df_lbl.to_numpy()
        for cat_id in rare_cat_ids:
            df_lbl_rare = df_lbl[df_lbl[:, 0] == cat_id, :]
            for dx in range(df_lbl_rare.shape[0]):
                w, h = df_lbl_rare[dx, 3] - df_lbl_rare[dx, 1], df_lbl_rare[dx, 4] - df_lbl_rare[dx, 2]
                whr = np.maximum(w/(h+1e-16), h/(w+1e-16))
                if whr < whr_thres and w > 4 and h > 4:
                    good_rare_gt_list.append(df_lbl_rare[dx, :])
        gt_boxes = []
        if good_rare_gt_list:
            good_rare_gt_arr = np.array(good_rare_gt_list)
            gt_boxes = good_rare_gt_arr[:, 1:]
            gt_classes = good_rare_gt_arr[:, 0]

        rare_prd_list = [rx for rx in rare_result_list if rx['image_id'] == img_id]
        prd_lbl_list = []
        for rx in rare_prd_list:
            w, h = rx['bbox'][2:]
            whr = np.maximum(w/(h+1e-16), h/(w+1e-16))
            rx['bbox'][2] = rx['bbox'][2] + rx['bbox'][0]
            rx['bbox'][3] = rx['bbox'][3] + rx['bbox'][1]
            # print('whr', whr)
            if whr < whr_thres and rx['score'] > score_thres:
                prd_lbl = [rx['category_id']]
                prd_lbl.extend([int(b) for b in rx['bbox']]) # xywh
                prd_lbl.extend([rx['score']])
                prd_lbl_list.append(prd_lbl)

        matches = []
        dt_boxes = []
        if prd_lbl_list:
            prd_lbl_arr = np.array(prd_lbl_list)
            # print(prd_lbl_arr.shape)
            dt_scores = prd_lbl_arr[:, -1] # [prd_lbl_arr[:, -1] > score_thres]
            dt_boxes = prd_lbl_arr[:, 1:-1][dt_scores > score_thres]
            dt_classes = prd_lbl_arr[:, 0][dt_scores > score_thres]

            for i in range(len(gt_boxes)):
                for j in range(dt_boxes.shape[0]):
                    iou = coord_iou(gt_boxes[i], dt_boxes[j])
                    if iou > iou_thres:
                        matches.append([i, j, iou])

        matches = np.array(matches)

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
        fn_list = []
        for i in range(len(gt_boxes)):
            if matches.shape[0] > 0 and matches[matches[:, 0] == i].shape[0] == 1:
                confusion_matrix[rare_cat_ids.index(gt_classes[i])][rare_cat_ids.index(dt_classes[int(matches[matches[:, 0] == i][0, 1])])] += 1
            elif matches.shape[0] > 0 and matches[matches[:, 0] == i].shape[0] > 1:
                confusion_matrix[rare_cat_ids.index(gt_classes[i]), -1] += 1
                drop_match_list = matches[matches[:, 0] == i][1:] # drop from the second matches
                for j in range(len(drop_match_list)):
                    c_box_iou = [gt_classes[drop_match_list[j, 0]]]
                    c_box_iou.extend(gt_boxes[drop_match_list[j, 0]])
                    c_box_iou.append(drop_match_list[j, 2]) # [cat_id, box[0:4], iou]
                    # print(c_box_iou)
                    fn_list.append(c_box_iou)
            else:
                confusion_matrix[rare_cat_ids.index(gt_classes[i]), -1] += 1
                c_box_iou = [gt_classes[i]]
                c_box_iou.extend(gt_boxes[i])
                c_box_iou.append(0)
                fn_list.append(c_box_iou)

        fp_list = []
        for j in range(len(dt_boxes)):
            if matches.shape[0] > 0 and matches[matches[:, 1] == j].shape[0] == 1:
                confusion_matrix[rare_cat_ids.index(gt_classes[int(matches[matches[:, 1] == j][0, 0])])][rare_cat_ids.index(dt_classes[j])] += 1
            elif matches.shape[0] > 0 and matches[matches[:, 1] == j].shape[0] > 1:
                confusion_matrix[-1][rare_cat_ids.index(dt_classes[j])] += 1
                drop_match_list = matches[matches[:, 1] == j][1:]
                for i in range(len(drop_match_list)):
                    c_box_iou = [dt_classes[drop_match_list[i, 1]]]
                    c_box_iou.extend(gt_boxes[drop_match_list[i, 1]])
                    c_box_iou.append(drop_match_list[i, 2]) # [cat_id, box[0:4], iou]
                    fp_list.append(c_box_iou)
            else:
                confusion_matrix[-1][rare_cat_ids.index(dt_classes[j])] += 1
                c_box_iou = [dt_classes[j]]
                c_box_iou.extend(dt_boxes[j])
                c_box_iou.append(0) # [cat_id, box[0:4], iou]
                fp_list.append(c_box_iou)

        rcids = []
        img = cv2.imread(args.images_save_dir + img_name)
        for gr in fn_list:
            gr_bx = [int(x) for x in gr[1:]]
            # print('gr', gr)
            img = cv2.rectangle(img, (gr_bx[0], gr_bx[1]), (gr_bx[2], gr_bx[3]), fn_color, 2)
            rcids.append(int(gr[0]))
            cv2.putText(img, text=str(int(gr[0])), org=(gr_bx[2] - 20, gr_bx[3] - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))

            for pr in fp_list:
                # print('pr', pr)
                pr_bx = [int(x) for x in pr[:-1]]
                # print('pr', pr_bx)
                rcids.append(pr_bx[0])
                img = cv2.rectangle(img, (pr_bx[1], pr_bx[2]), (pr_bx[3], pr_bx[4]), fp_color, 2)
                cv2.putText(img, text=str(pr_bx[0]), org=(pr_bx[1] + 5, pr_bx[2] + 10), # [pr_bx[0], pr[-1]]
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 0))
            rcids = list(set(rcids))
        cv2.imwrite(save_dir + 'cat{}_'.format(rcids) + rare_img_name_list[ix], img)

    np.save(confu_mat_dir + 'rare_confusion_matrix.npy', confusion_matrix)

    return confusion_matrix


def summary_confusion_matrix(confusion_matrix, rare_cat_ids, iou_thres=0.5):
    '''
    https://github.com/svpino/tf_object_detection_cm/blob/83cb8a1cf3a5abd24b18a5fc79b5ce99e8a9b317/confusion_matrix.py#L37
    '''
    print("\nConfusion Matrix:")
    print(confusion_matrix, "\n")
    results = []
    df_category_id = pd.read_csv(args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num), sep="\t")
    category_names = df_category_id['category'].tolist()
    category_ids = df_category_id['category_id'].tolist()
    rare_cat_names = []
    for i in range(len(rare_cat_ids)):
        id = rare_cat_ids.index(rare_cat_ids[i])
        name = category_names[category_ids.index(rare_cat_ids[i])]
        rare_cat_names.append(name)

        total_target = np.sum(confusion_matrix[id, :])
        total_predicted = np.sum(confusion_matrix[:, id])

        precision = float(confusion_matrix[id, id] / total_predicted)
        recall = float(confusion_matrix[id, id] / total_target)

        #print('precision_{}@{}IOU: {:.2f}'.format(name, IOU_THRESHOLD, precision))
        #print('recall_{}@{}IOU: {:.2f}'.format(name, IOU_THRESHOLD, recall))

        results.append({'category' : name, 'precision_@{}IOU'.format(iou_thres) : precision, 'recall_@{}IOU'.format(iou_thres) : recall})

    df = pd.DataFrame(results)
    print(df)
    save_dir = '../result_output/result_confusion_matrix_iou{}/'.format(iou_thres)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file_name = 'result_confusion_mtx.csv'
    df.to_csv(save_dir+ file_name)


def plot_confusion_matrix(confusion_matrix, rare_cat_ids, iou_thres=0.5):
    df_category_id = pd.read_csv(args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num), sep="\t")
    category_names = df_category_id['category'].tolist()
    category_ids = df_category_id['category_id'].tolist()
    rare_cat_names = []
    for i in range(len(rare_cat_ids)):
        id = rare_cat_ids.index(rare_cat_ids[i])
        name = category_names[id]
        rare_cat_names.append(name)

    rare_cat_names.append('background')
    rare_cat_ids.append(000)
    # print(confusion_matrix.shape)
    df_cm = pd.DataFrame(confusion_matrix[:len(confusion_matrix), :len(confusion_matrix)], index=[i for i in rare_cat_names], columns=[i for i in rare_cat_names])
    # sn.set(font_scale=1) # for label size
    # sn.heatmap(df_cm, annot=True, cmap="YlGnBu") # font size
    plt.figure(figsize=(10, 8))

    # cmap = sn.cubehelix_palette(start=5, rot=10,  gamma=0.8, as_cmap=True)
    p = sn.heatmap(df_cm, annot=True, fmt='g', cmap='YlGnBu')# center=True, cmap=cmap,
    fig = p.get_figure()
    fig.tight_layout()
    plt.xticks(rotation=325)
    save_dir = '../result_output/result_confusion_matrix_iou{}/'.format(iou_thres)
    fig.savefig(save_dir + 'rare_confusion_matrix.png', bbox_inches='tight')
    plt.show()


def autolabel(ax, rects, x, labels, ylabel, rotation=90, txt_rotation=45):
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
        else: # rotation=90, multiline label
            xticks = []
            for i in range(len(labels)):
                xticks.append('{} {}'.format(x[i], labels[i]))

            ax.set_xticklabels(xticks, rotation=rotation)
        ax.set_ylabel(ylabel)
        ax.grid(True)


def draw_rare_cat(N):

    title = 'Categories Count Least Rare {}'.format(N)
    save_dir = args.fig_save_dir + '{}_{}_cls_xcycwh/'.format(args.input_size, args.class_num)
    png_name = 'all_{}_cat_cnt_with_top{}_least_rare.png'.format(args.input_size, N)
    df_cat = pd.read_csv(os.path.join(args.txt_save_dir, 'all_{}_cat_cnt_{}cls.csv'.format(args.input_size, args.class_num)))
    x_ids = df_cat['category_id']
    x_cats = df_cat['category']
    y = df_cat['category_count']

    y_sort_inx = np.argsort(y)[:N]
    labels = x_cats[y_sort_inx].to_list()
    print(y_sort_inx)
    # x = [x_ids[i] for i in y_sort_inx]
    x = x_ids[y_sort_inx].to_list()
    y = y[y_sort_inx].to_list()
    print('category ids', x)
    print('category', labels)
    print('num', y)

    draw_bar_for_each_cat_cnt_with_txt_rotation(x, y, labels, title, save_dir, png_name, rotation=80, sort=True)

def get_cat_names_by_cat_ids(rare_cat_ids):
    img_ids_names_file = '../data_xview/{}_cls/categories_id_color_diverse_{}.txt'.format(args.class_num, args.class_num)
    df_cat_id_names = pd.read_csv(img_ids_names_file, delimiter='\t')
    cat_names = df_cat_id_names['category'].tolist()
    cat_ids = df_cat_id_names['category_id'].tolist()
    rare_cat_names = []
    for c in rare_cat_ids:
        rare_cat_names.append(cat_names[cat_ids.index(c)])
    return rare_cat_names



'''
Datasets
https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data
_aug: Augmented dataset
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str,
                        help="Path to folder containing image chips (ie 'Image_Chips/') ",
                        default='/media/lab/Yang/data/xView/train_images/')

    parser.add_argument("--json_filepath", type=str, help="Filepath to GEOJSON coordinate file",
                        default='/media/lab/Yang/data/xView/xView_train.geojson')

    parser.add_argument("--xview_yolo_dir", type=str, help="dir to xViewYOLO",
                        default='/media/lab/Yang/data/xView_YOLO/')

    parser.add_argument("--images_save_dir", type=str, help="to save chip trn val images files",
                        default='/media/lab/Yang/data/xView_YOLO/images/')

    parser.add_argument("--txt_save_dir", type=str, help="to save txt labels files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')

    parser.add_argument("--fig_save_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/figures/')

    parser.add_argument("--data_save_dir", type=str, help="to save data files",
                        default='../data_xview/')

    parser.add_argument("--results_dir", type=str, help="to save category files",
                        default='../result_output/')

    parser.add_argument("--cat_sample_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/')

    parser.add_argument("--img_bbx_figures_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/{}/img_name_2_bbx_figures/')

    parser.add_argument("--rare_cat_bbx_patches_dir", type=str, help="to split cats bbx patches",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/{}/rare_cat_split_patches')

    parser.add_argument("--val_percent", type=float, default=0.20,
                        help="Percent to split into validation (ie .25 = val set is 25% total)")
    parser.add_argument("--font3", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 10}")

    parser.add_argument("--class_num", type=int, default=60, help="Number of Total Categories")  # 60  6
    parser.add_argument("--seed", type=int, default=1024, help="random seed")
    parser.add_argument("--input_size", type=int, default=608, help="Number of Total Categories")  # 300 416

    args = parser.parse_args()

    args.txt_save_dir = args.txt_save_dir + '{}/'.format(args.input_size)
    args.images_save_dir = args.images_save_dir + '{}/'.format(args.input_size)
    args.cat_sample_dir = args.cat_sample_dir + '{}/'.format(args.input_size)
    args.data_save_dir = args.data_save_dir + '{}_cls/'.format(args.class_num)
    args.img_bbx_figures_dir = args.img_bbx_figures_dir.format(args.input_size)
    args.rare_cat_bbx_patches_dir = args.rare_cat_bbx_patches_dir.format(args.input_size)

    if not os.path.exists(args.txt_save_dir):
        os.mkdir(args.txt_save_dir)

    if not os.path.exists(args.images_save_dir):
        os.mkdir(args.images_save_dir)

    if not os.path.exists(args.cat_sample_dir):
        os.mkdir(args.cat_sample_dir)

    if not os.path.exists(args.data_save_dir):
        os.mkdir(args.data_save_dir)

    if not os.path.exists(args.img_bbx_figures_dir):
        os.mkdir(args.img_bbx_figures_dir)

    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger(__name__)


    '''
    draw cat cnt least N rare classes
    '''
    # N = 20
    # draw_rare_cat(N)

    '''
    get rare class names by cat_id
    '''
    # rare_cat_ids = [18, 46, 23, 43, 33, 59]
    # rare_cat_names = get_cat_names_by_cat_ids(rare_cat_ids)
    # print(rare_cat_names)



    '''
    plot img with bbx by img name
    '''
    # image_name = '1139_4.jpg'
    # image_name = '5_7.jpg'
    # image_name = '2230_3.jpg'
    # plot_val_image_with_bbx_by_image_name(image_name)


    '''
    get img id by image_name
    '''
    # image_name = '2230_3.jpg' # 6121
    # image_name = '145_16.jpg' # 2609
    # image_name = '1154_6.jpg' # 1238
    # image_name = '1076_16.jpg' # 489
    # image_name = '2294_5.jpg' #6245
    # image_name = '1568_0.jpg' # 3160
    # image_id = get_img_id_by_image_name(image_name)
    # print(image_id)

    '''
    plot rare results 
    '''
    # rare_cat_ids = [18, 46, 23, 43, 33, 59]
    # # rare_cat_ids = [18]
    # score_thres = 0.3
    # iou_thres=0.5
    # whr_thres=3
    # plot_rare_results_by_rare_cat_ids(rare_cat_ids, iou_thres, score_thres, whr_thres)

    '''
    IoU check by image name
    '''
    # score_thres = 0.5
    # iou_thres = 0.5
    # image_name = '310_12.jpg'
    # check_prd_gt_iou(image_name, score_thres, iou_thresh)


    '''
    rare results confusion matrix  rare results statistic FP FN NMS
    '''
    # rare_cat_ids = [18, 46, 23, 43, 33, 59]
    # score_thres = 0.3
    # whr_thres = 3
    # iou_thres = 0.5
    # confusion_matrix = get_confusion_matrix(rare_cat_ids, iou_thres, score_thres, whr_thres)
    # summary_confusion_matrix(confusion_matrix, rare_cat_ids)
    #
    # rare_cat_ids = [18, 46, 23, 43, 33, 59]
    # iou_thres = 0.5
    # confusion_matrix = np.load('../result_output/result_confusion_matrix_iou0.5/rare_confusion_matrix.npy')
    # plot_confusion_matrix(confusion_matrix, rare_cat_ids, iou_thres)





