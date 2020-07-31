import glob
import numpy as np
import argparse
import os
import pandas as pd
from ast import literal_eval
import json
import shutil

IMG_FORMAT = '.jpg'
TXT_FORMAT = '.txt'


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def get_xview_background_double_data(seed=1024, src_cmts='_px6whr4_ng0'):
    if seed == 1024:
        comments = '_xview_background_double'
    else:
        comments = '_xview_background_double_seed{}'.format(seed)

    args = get_part_args(comments, seed)

    src_txt_dir = args.annos_save_dir.replace(comments, src_cmts)
    src_txt_files = np.sort(glob.glob(os.path.join(src_txt_dir, '*' + TXT_FORMAT)))
    src_img_names = [os.path.basename(f).replace(TXT_FORMAT, IMG_FORMAT) for f in src_txt_files]
    empty_txt_names = [os.path.basename(f) for f in src_txt_files if not is_non_zero_file(f)]

    all_img_files = np.sort(glob.glob(os.path.join(args.images_src_dir, '*' + IMG_FORMAT)))
    all_img_names = [os.path.basename(f) for f in all_img_files]
    rest_img_names = [r for r in all_img_names if r not in src_img_names]
    # src_txt_files contain empty label files, so minus the number of original empty files
    num_trn_background = len(src_img_names) - 2*len(empty_txt_names)

    np.random.seed(args.seed)
    indices = np.random.permutation(len(rest_img_names))

    for i in range(num_trn_background): # select the same number of images from rest as src_img_names
        img_name = rest_img_names[indices[i]]
        rest_txt_file = open(os.path.join(args.annos_save_dir, img_name.replace(IMG_FORMAT, TXT_FORMAT)), 'w')
        rest_txt_file.close()

    for j in range(len(src_txt_files)): # copy src txt files to *_xview_background_double folders
        txt_name = os.path.basename(src_txt_files[j])
        shutil.copy(src_txt_files[j], os.path.join(args.annos_save_dir, txt_name))


def collect_xview_data(seed=1024, data_name='xview_background_double', src_cmts='px6whr4_ng0'):
    comments = '_' + data_name
    xview_args = get_part_args(comments, seed)
    df_src_val = pd.read_csv(os.path.join(xview_args.data_save_dir.replace(data_name, src_cmts), 'xviewval_lbl_{}.txt'.format(src_cmts)), header=None)
    src_val_names = [os.path.basename(f) for f in df_src_val.loc[:, 0]]
    df_src_trn = pd.read_csv(os.path.join(xview_args.data_save_dir.replace(data_name, src_cmts), 'xviewtrain_lbl_{}.txt'.format(src_cmts)), header=None)
    src_trn_names = [os.path.basename(f) for f in df_src_trn.loc[:, 0]]

    # src_txt_files = np.sort(glob.glob(os.path.join(xview_args.annos_save_dir.replace(data_name, src_cmts), '*' + TXT_FORMAT)))
    # empty_txt_names = [os.path.basename(f) for f in src_txt_files if not is_non_zero_file(f)]

    all_files = np.sort(glob.glob(os.path.join(xview_args.annos_save_dir, '*' + TXT_FORMAT)))
    all_txt_names = [os.path.basename(f) for f in all_files]

    rest_names = [f for f in all_txt_names if f not in src_val_names and f not in src_trn_names]
    rest_val_num = int(len(rest_names) * xview_args.val_percent)
    rest_trn_num = len(rest_names) - rest_val_num

    np.random.seed(seed)
    rest_indices = np.random.permutation(len(rest_names))

    trn_img_txt = open(os.path.join(xview_args.data_save_dir, '{}_train_img.txt'.format(data_name)), 'w')
    trn_lbl_txt = open(os.path.join(xview_args.data_save_dir, '{}_train_lbl.txt'.format(data_name)), 'w')

    for t in df_src_trn.loc[:, 0]:
        trn_lbl_txt.write("%s\n" % t)
        img_name = os.path.basename(t).replace(TXT_FORMAT, IMG_FORMAT)
        trn_img_txt.write("%s\n" % (os.path.join(xview_args.images_src_dir, img_name)))

    for i in range(rest_trn_num):
        trn_lbl_txt.write("%s\n" % os.path.join(xview_args.annos_save_dir, rest_names[rest_indices[i]]))
        img_name = rest_names[rest_indices[i]].replace(TXT_FORMAT, IMG_FORMAT)
        trn_img_txt.write("%s\n" % (os.path.join(xview_args.images_src_dir, img_name)))

    trn_img_txt.close()
    trn_lbl_txt.close()

    val_img_txt = open(os.path.join(xview_args.data_save_dir, '{}_val_img.txt'.format(data_name)), 'w')
    val_lbl_txt = open(os.path.join(xview_args.data_save_dir, '{}_val_lbl.txt'.format(data_name)), 'w')
    for v in df_src_val.loc[:, 0]:
        val_lbl_txt.write("%s\n" % v)
        img_name = os.path.basename(v).replace(TXT_FORMAT, IMG_FORMAT)
        val_img_txt.write("%s\n" % (os.path.join(xview_args.images_src_dir, img_name)))

    for j in range(rest_val_num):
        val_lbl_txt.write("%s\n" % os.path.join(xview_args.annos_save_dir, rest_names[rest_indices[j]]))
        img_name = rest_names[rest_indices[j]].replace(TXT_FORMAT, IMG_FORMAT)
        val_img_txt.write("%s\n" % (os.path.join(xview_args.images_src_dir, img_name)))

    val_img_txt.close()
    val_lbl_txt.close()

    shutil.copyfile(os.path.join(xview_args.data_save_dir, '{}_train_img.txt'.format(data_name)),
                    os.path.join(xview_args.data_list_save_dir, '{}_train_img.txt'.format(data_name)))
    shutil.copyfile(os.path.join(xview_args.data_save_dir, '{}_train_lbl.txt'.format(data_name)),
                    os.path.join(xview_args.data_list_save_dir, '{}_train_lbl.txt'.format(data_name)))
    shutil.copyfile(os.path.join(xview_args.data_save_dir, '{}_val_img.txt'.format(data_name)),
                    os.path.join(xview_args.data_list_save_dir, '{}_val_img.txt'.format(data_name)))
    shutil.copyfile(os.path.join(xview_args.data_save_dir, '{}_val_lbl.txt'.format(data_name)),
                    os.path.join(xview_args.data_list_save_dir, '{}_val_lbl.txt'.format(data_name)))


def create_xview_data(seed=1024, src_cmts='px6whr4_ng0', dst_cmts='px6whr4_ng0', comments='xview_background_double'):
    xview_args = get_part_args('_' + comments, seed)
    data_txt = open(os.path.join(xview_args.data_save_dir, '{}_{}.data'.format(comments, dst_cmts)), 'w')
    data_txt.write('train=./data_xview/{}_cls/{}/{}_train_img.txt\n'.format(xview_args.class_num, comments, comments))
    data_txt.write('train_label=./data_xview/{}_cls/{}/{}_train_lbl.txt\n'.format(xview_args.class_num, comments, comments))

    df = pd.read_csv(os.path.join(xview_args.data_save_dir.replace(comments, src_cmts), 'xviewtrain_img_{}.txt'.format(src_cmts)), header=None) # **********
    data_txt.write('syn_0_xview_number=%d\n' % df.shape[0]) # fixme **********
    data_txt.write('classes=%s\n' % str(xview_args.class_num))
    data_txt.write('valid=./data_xview/{}_cls/{}/{}_val_img.txt\n'.format(xview_args.class_num, comments, comments))
    data_txt.write('valid_label=./data_xview/{}_cls/{}/{}_val_lbl.txt\n'.format(xview_args.class_num, comments, comments))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(xview_args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval={}'.format(comments))
    data_txt.close()

    shutil.copy(os.path.join(xview_args.data_save_dir, '{}_{}.data'.format(comments, dst_cmts)),
                os.path.join(xview_args.data_save_dir, '{}_{}_mosaic_rect.data'.format(comments, dst_cmts)))


def get_part_args(comments='', seed=1024):
    parser = argparse.ArgumentParser()
    parser.add_argument("--xview_yolo_dir", type=str, help="dir to xViewYOLO",
                        default='/media/lab/Yang/data/xView_YOLO/')

    parser.add_argument("--images_src_dir", type=str, help="to get chip trn val images files",
                        default='/media/lab/Yang/data/xView_YOLO/images/{}/')

    parser.add_argument("--txt_save_dir", type=str, help="to save  related label files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_cls/')
    parser.add_argument("--data_list_save_dir", type=str, help="to save selected trn val images and labels",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_cls/data_list/{}/')

    parser.add_argument("--annos_save_dir", type=str, help="to save txt annotation files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_cls_xcycwh{}/')

    parser.add_argument("--data_save_dir", type=str, help="to save data files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_cls/{}/')

    parser.add_argument("--val_percent", type=float, default=0.20,
                        help="Percent to split into validation (ie .25 = val set is 25% total)")
    parser.add_argument("--font3", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 10}")
    parser.add_argument("-ft2", "--font2", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 13}")
    parser.add_argument("--class_num", type=int, default=1, help="Number of Total Categories")  # 60  6 1
    parser.add_argument("--seed", type=int, default=seed, help="random seed") #fixme ---- 1024 17
    parser.add_argument("--input_size", type=int, default=608, help="Number of Total Categories")  # 300 416

    args = parser.parse_args()

    args.images_src_dir = args.images_src_dir.format(args.input_size)
    args.annos_save_dir = args.annos_save_dir.format(args.input_size, args.class_num, comments)
    args.txt_save_dir = args.txt_save_dir.format(args.input_size, args.class_num)
    args.data_save_dir = args.data_save_dir.format(args.class_num, comments[1:])
    args.data_list_save_dir = args.data_list_save_dir.format(args.input_size, args.class_num, comments[1:])

    if not os.path.exists(args.txt_save_dir):
        os.makedirs(args.txt_save_dir)

    if not os.path.exists(args.annos_save_dir):
        os.makedirs(args.annos_save_dir)

    if not os.path.exists(args.images_src_dir):
        os.makedirs(args.images_src_dir)

    if not os.path.exists(args.data_save_dir):
        os.makedirs(args.data_save_dir)

    if not os.path.exists(args.data_list_save_dir):
        os.makedirs(args.data_list_save_dir)

    return args


if __name__ == '__main__':
    '''
    xview only: 1. xview with airplanes 2. xview without airplanes
    '''
    # seed = 1024
    # seed = 17
    # get_xview_background_double_data(seed)

    seeds = [3, 5, 9]
    for seed in seeds:
        get_xview_background_double_data(seed)

    '''
    collect_xview_data
    '''
    # src_cmts = 'px6whr4_ng0'
    # seed = 1024
    # data_name = 'xview_background_double'
    # collect_xview_data(seed=seed, data_name, src_cmts)

    # seed = 17
    # src_cmts = 'px6whr4_ng0_seed{}'.format(seed)
    # data_name = 'xview_background_double_seed{}'.format(seed)
    # collect_xview_data(seed=seed, data_name, src_cmts)

    # seeds = [3, 5, 9]
    # for seed in seeds:
    #     src_cmts = 'px6whr4_ng0'
    #     data_name = 'xview_background_double_seed{}'.format(seed)
    #     collect_xview_data(seed, data_name, src_cmts)

    '''
    create xview_background_double.data
    '''
    # dst_cmts = src_cmts = 'px6whr4_ng0'
    # comments = 'xview_background_double'
    # seed = 1024
    # create_xview_data(seed, src_cmts, comments)

    # seed = 17
    # dst_cmts = src_cmts = 'px6whr4_ng0_seed{}'.format(seed)
    # comments = 'xview_background_double_seed{}'.format(seed)
    # create_xview_data(seed, src_cmts, comments)

    seeds = [3, 5, 9]
    for seed in seeds:
        src_cmts = 'px6whr4_ng0'
        dst_cmts = 'px6whr4_hgiou1'
        comments = 'xview_background_double_seed{}'.format(seed)
        create_xview_data(seed, src_cmts, dst_cmts, comments)



