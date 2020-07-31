import glob
import numpy as np
import argparse
import os
import pandas as pd
from ast import literal_eval
from matplotlib import pyplot as plt
import cv2
import json
import shutil
from utils.data_process_distribution_vis_util import process_wv_coco_for_yolo_patches_no_trnval as pwv
from utils.xview_synthetic_util import preprocess_xview_syn_data_distribution as pps
from utils.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc
from utils.xview_synthetic_util import anaylze_xview_syn_results as axs

IMG_SUFFIX = '.png'
TXT_SUFFIX = '.txt'


def check_difference_of_first_second_dataset():
    new_val = pd.read_csv('/data_xview/1_cls/second_dataset_backup/xviewval_img.txt', header=None)
    new_val = list(new_val.loc[:, 0])
    new_vname = [os.path.basename(f) for f in new_val]
    new_val_lbl = pd.read_csv('/data_xview/1_cls/second_dataset_backup/xviewval_lbl.txt', header=None)
    new_val_lbl = list(new_val_lbl.loc[:, 0])

    new_trn = pd.read_csv('/data_xview/1_cls/second_dataset_backup/xviewtrain_img.txt', header=None)
    new_trn = list(new_trn.loc[:, 0])
    new_tname = [os.path.basename(f) for f in new_trn]
    new_trn_lbl = pd.read_csv('/data_xview/1_cls/second_dataset_backup/xviewtrain_lbl.txt', header=None)
    new_trn_lbl = list(new_trn_lbl.loc[:, 0])

    old_val = pd.read_csv('/media/lab/Yang/code/yolov3/data_xview/1_cls/first_data_set_backup/xview_val_img.txt', header=None)
    old_val = list(old_val.loc[:, 0])
    old_vname = [os.path.basename(f) for f in old_val]
    old_val_lbl = pd.read_csv('/media/lab/Yang/code/yolov3/data_xview/1_cls/first_data_set_backup/xview_val_lbl.txt', header=None)
    old_val_lbl = list(old_val_lbl.loc[:, 0])

    old_trn = pd.read_csv('/media/lab/Yang/code/yolov3/data_xview/1_cls/first_data_set_backup/xview_train_img.txt', header=None)
    old_trn = list(old_trn.loc[:, 0])
    old_tname = [os.path.basename(f) for f in old_trn]
    old_trn_lbl = pd.read_csv('/media/lab/Yang/code/yolov3/data_xview/1_cls/first_data_set_backup/xview_train_lbl.txt', header=None)
    old_trn_lbl = list(old_trn_lbl.loc[:, 0])

    only_old_v = [v for v in old_vname if v not in new_vname]
    only_old_t = [t for t in old_tname if t not in new_tname]
    only_new_v = [v for v in new_vname if v not in old_vname]
    only_new_t = [t for t in new_tname if t not in old_tname]
    print('only_old_v', len(only_old_v))
    print('only_old_t', len(only_old_t))
    print('only_new_v', len(only_new_v))
    print('only_new_t', len(only_new_t))

    data_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_drop/'
    only_old_t_dir = os.path.join(data_dir, 'only_in_old_trn')
    if not os.path.exists(only_old_t_dir):
        os.mkdir(only_old_t_dir)
    only_old_v_dir = os.path.join(data_dir, 'only_in_old_val')
    if not os.path.exists(only_old_v_dir):
        os.mkdir(only_old_v_dir)
    only_new_t_dir = os.path.join(data_dir, 'only_in_new_trn')
    if not os.path.exists(only_new_t_dir):
        os.mkdir(only_new_t_dir)
    only_new_v_dir = os.path.join(data_dir, 'only_in_new_val')
    if not os.path.exists(only_new_v_dir):
        os.mkdir(only_new_v_dir)

    for v in only_old_t:
        # shutil.copy(old_trn[old_tname.index(v)], os.path.join(only_old_t_dir, v))
        gbc.plot_img_with_bbx(old_trn[old_tname.index(v)], old_trn_lbl[old_tname.index(v)], only_old_t_dir)
    for v in only_old_v:
        # shutil.copy(old_val[old_vname.index(v)], os.path.join(only_old_v_dir, v))
        gbc.plot_img_with_bbx(old_val[old_vname.index(v)], old_val_lbl[old_vname.index(v)], only_old_v_dir)
    for v in only_new_t:
        # shutil.copy(new_trn[new_tname.index(v)], os.path.join(only_new_t_dir, v))
        gbc.plot_img_with_bbx(new_trn[new_tname.index(v)], new_trn_lbl[new_tname.index(v)], only_new_t_dir)
    for v in only_new_v:
        # shutil.copy(new_val[new_vname.index(v)], os.path.join(only_new_v_dir, v))
        gbc.plot_img_with_bbx(new_val[new_vname.index(v)], new_val_lbl[new_vname.index(v)], only_new_v_dir)


def combine_xview_syn_by_ratio(dt, syn_ratio, comments='', seed=1024):
    syn_args = get_part_syn_args(dt, syn_ratio)
    if comments:
        comments = comments.format(seed)
        data_xview_dir = os.path.join(syn_args.data_xview_dir, comments[1:])
    else:
        data_xview_dir = syn_args.data_xview_dir
    # print(os.path.join(syn_args.data_xview_dir, '{}_{}_img.txt'.format(syn_args.syn_display_type, syn_args.class_num)))
    syn_img_txt = pd.read_csv(open(os.path.join(syn_args.syn_data_list_dir, '{}_{}_img.txt'.format(dt, syn_args.class_num))), header=None).to_numpy()
    syn_lbl_txt = pd.read_csv(open(os.path.join(syn_args.syn_data_list_dir, '{}_{}_lbl.txt'.format(dt, syn_args.class_num))), header=None).to_numpy()
    syn_total_num = syn_img_txt.shape[0]
    np.random.seed(seed)
    perm_indices = np.random.permutation(syn_total_num)
    syn_ratio_num = np.int(syn_total_num*syn_ratio)

    xview_img_txt = pd.read_csv(open(os.path.join(data_xview_dir, 'xviewtrain_img{}.txt'.format(comments))), header=None).to_numpy()
    xview_lbl_txt = pd.read_csv(open(os.path.join(data_xview_dir, 'xviewtrain_lbl{}.txt'.format(comments))), header=None).to_numpy()
    xview_trn_num = xview_img_txt.shape[0]

    xview_syn_trn_txt_list_dir = os.path.join(syn_args.data_xview_dir, 'xview_{}_{}'.format(dt, syn_ratio))
    if not os.path.exists(xview_syn_trn_txt_list_dir):
        os.mkdir(xview_syn_trn_txt_list_dir)

    f_img = open(os.path.join(xview_syn_trn_txt_list_dir, 'xview_{}_{}_train_img{}.txt'.format(dt, syn_ratio, comments)), 'w')
    f_lbl = open(os.path.join(xview_syn_trn_txt_list_dir, 'xview_{}_{}_train_lbl{}.txt'.format(dt, syn_ratio, comments)), 'w')

    for ix in range(xview_trn_num):
        f_img.write("%s\n" % xview_img_txt[ix, 0])
        f_lbl.write("%s\n" % xview_lbl_txt[ix, 0])

    for i in range(syn_ratio_num):
        f_img.write("%s\n" % syn_img_txt[perm_indices[i], 0])
        f_lbl.write("%s\n" % syn_lbl_txt[perm_indices[i], 0])
    f_img.close()
    f_lbl.close()

    xview_syn_data_list_dir = os.path.join(syn_args.data_list_save_dir, 'xview_{}_{}'.format(dt, syn_ratio))
    if not os.path.exists(xview_syn_data_list_dir):
        os.mkdir(xview_syn_data_list_dir)
    shutil.copy(os.path.join(xview_syn_trn_txt_list_dir, 'xview_{}_{}_train_img{}.txt'.format(dt, syn_ratio, comments)),
                os.path.join(xview_syn_data_list_dir, 'xview_{}_{}_train_img{}.txt'.format(dt, syn_ratio, comments)))
    shutil.copy(os.path.join(xview_syn_trn_txt_list_dir, 'xview_{}_{}_train_lbl{}.txt'.format(dt, syn_ratio, comments)),
                os.path.join(xview_syn_data_list_dir, 'xview_{}_{}_train_lbl{}.txt'.format(dt, syn_ratio, comments)))


def mv_list():
    '''
    move the *.txt to des dir if necessary
    :return:
    '''
    syn_args = get_part_syn_args()
    args = pwv.get_args()
    srs = [0.25, 0.5, 0.75]
    for syn_ratio in srs:
        xview_syn_trn_txt_list_dir = os.path.join(syn_args.data_xview_dir, 'xview_{}_{}'.format(syn_args.syn_display_type, syn_ratio))
        if not os.path.exists(xview_syn_trn_txt_list_dir):
            os.mkdir(xview_syn_trn_txt_list_dir)
        xview_syn_data_list_dir = os.path.join(syn_args.data_list_save_dir, 'xview_{}_{}'.format(syn_args.syn_display_type, syn_ratio))
        if not os.path.exists(xview_syn_data_list_dir):
            os.mkdir(xview_syn_data_list_dir)
        f_img = os.path.join(syn_args.data_xview_dir, 'xview_{}_{}_train_img.txt'.format(syn_args.syn_display_type, syn_ratio))
        f_lbl = os.path.join(syn_args.data_xview_dir, 'xview_{}_{}_train_lbl.txt'.format(syn_args.syn_display_type, syn_ratio))
        f_data = os.path.join(syn_args.data_xview_dir, 'xview_{}_{}.data'.format(syn_args.syn_display_type, syn_ratio))
        t_img = os.path.join(xview_syn_trn_txt_list_dir, 'xview_{}_{}_train_img.txt'.format(syn_args.syn_display_type, syn_ratio))
        t_lbl = os.path.join(xview_syn_trn_txt_list_dir, 'xview_{}_{}_train_lbl.txt'.format(syn_args.syn_display_type, syn_ratio))
        t_data = os.path.join(xview_syn_trn_txt_list_dir, 'xview_{}_{}.data'.format(syn_args.syn_display_type, syn_ratio))
        if os.path.exists(f_img):
            shutil.move(f_img, t_img)
        if os.path.exists(f_lbl):
            shutil.move(f_lbl, t_lbl)
        if os.path.exists(f_data):
            shutil.move(f_data, t_data)

        xview_syn_data_list_dir = os.path.join(syn_args.data_list_save_dir, 'xview_{}_{}'.format(syn_args.syn_display_type, syn_ratio))
        if not os.path.exists(xview_syn_data_list_dir):
            os.mkdir(xview_syn_data_list_dir)

        src_img_txt = os.path.join(syn_args.data_list_save_dir, 'xview_{}_{}_train_img.txt'.format(syn_args.syn_display_type, syn_ratio))
        src_lbl_txt = os.path.join(syn_args.data_list_save_dir, 'xview_{}_{}_train_lbl.txt'.format(syn_args.syn_display_type, syn_ratio))
        dst_img_txt = os.path.join(syn_args.data_list_save_dir, 'xview_{}_{}'.format(syn_args.syn_display_type, syn_ratio), 'xview_{}_{}_train_img.txt'.format(syn_args.syn_display_type, syn_ratio))
        dst_lbl_txt = os.path.join(syn_args.data_list_save_dir, 'xview_{}_{}'.format(syn_args.syn_display_type, syn_ratio), 'xview_{}_{}_train_lbl.txt'.format(syn_args.syn_display_type, syn_ratio))
        if os.path.exists(src_img_txt):
            shutil.move(src_img_txt, dst_img_txt)
        if os.path.exists(src_lbl_txt):
            shutil.move(src_lbl_txt, dst_lbl_txt)


def collect_syn_data(data_name, comments=''):
    syn_args = pps.get_syn_args()
    images_save_dir = syn_args.syn_images_save_dir
    txt_save_dir = syn_args.syn_txt_save_dir
    data_save_dir = syn_args.syn_data_list_dir
    lbl_path = syn_args.syn_annos_save_dir

    if '0' in syn_args.syn_display_type:
        suffix = '.jpg'
    else:
        suffix = IMG_SUFFIX
    all_files = np.sort(glob.glob(images_save_dir + '*' + suffix))
    num_files = len(all_files)

    img_txt = open(os.path.join(txt_save_dir, '{}_img{}.txt'.format(data_name, comments)), 'w')
    lbl_txt = open(os.path.join(txt_save_dir, '{}_lbl{}.txt'.format(data_name, comments)), 'w')

    for i in range(num_files):
        img_txt.write("%s\n" % all_files[i])
        img_name = all_files[i].split('/')[-1]
        lbl_name = img_name.replace(suffix, TXT_SUFFIX)
        lbl_txt.write("%s\n" % (lbl_path + lbl_name))

    img_txt.close()
    lbl_txt.close()

    shutil.copyfile(os.path.join(txt_save_dir, '{}_img{}.txt'.format(data_name, comments)),
                    os.path.join(data_save_dir, '{}_img{}.txt'.format(data_name, comments)))
    shutil.copyfile(os.path.join(txt_save_dir, '{}_lbl{}.txt'.format(data_name, comments)),
                    os.path.join(data_save_dir, '{}_lbl{}.txt'.format(data_name, comments)))


def split_syn_trn_val(display_type='syn_texture'):
    syn_args = get_part_syn_args(display_type, sr)
    data_dir = syn_args.syn_data_list_dir.format(display_type, syn_args.class_num)
    syn_all_img_files = pd.read_csv(os.path.join(data_dir, '{}_{}_img.txt'.format(display_type, syn_args.class_num)), header=None).to_numpy()
    syn_all_lbl_files = pd.read_csv(os.path.join(data_dir, '{}_{}_lbl.txt'.format(display_type, syn_args.class_num)), header=None).to_numpy()

    val_num = np.int(syn_all_img_files.shape[0] * syn_args.val_percent)
    np.random.seed(syn_args.seed)
    all_indices = np.random.permutation(syn_all_img_files.shape[0])

    trn_img_txt = open(os.path.join(data_dir, '{}_{}_train_img.txt'.format(display_type, syn_args.class_num )), 'w')
    trn_lbl_txt = open(os.path.join(data_dir, '{}_{}_train_lbl.txt'.format(display_type, syn_args.class_num)), 'w')

    val_img_txt = open(os.path.join(data_dir, '{}_{}_val_img.txt'.format(display_type, syn_args.class_num)), 'w')
    val_lbl_txt = open(os.path.join(data_dir, '{}_{}_val_lbl.txt'.format(display_type, syn_args.class_num)), 'w')

    for i in all_indices[:val_num]:
        val_img_txt.write('%s\n' % syn_all_img_files[i, 0])
        val_lbl_txt.write('%s\n' % syn_all_lbl_files[i, 0])
    val_img_txt.close()
    val_lbl_txt.close()
    for j in all_indices[val_num:]:
        trn_img_txt.write('%s\n' % syn_all_img_files[j, 0])
        trn_lbl_txt.write('%s\n' % syn_all_lbl_files[j, 0])
    trn_img_txt.close()
    trn_lbl_txt.close()


def create_syn_data(dt, comments='syn_only'):
    syn_args = get_part_syn_args()
    data_dir = syn_args.syn_data_list_dir.format(dt, syn_args.class_num)
    data_txt = open(os.path.join(syn_args.data_xview_dir, '{}_{}.data'.format(dt, comments)), 'w')
    data_txt.write('train=./data_xview/{}_{}_cls/{}_{}_train_img.txt\n'.format(dt, syn_args.class_num, dt, syn_args.class_num))
    data_txt.write('train_label=./data_xview/{}_{}_cls/{}_{}_train_lbl.txt\n'.format(dt, syn_args.class_num, dt, syn_args.class_num))

    df = pd.read_csv(os.path.join(data_dir, '{}_{}_train_img.txt'.format(dt, syn_args.class_num)), header=None) # **********
    data_txt.write('syn_0_xview_number=%d\n' % df.shape[0]) #fixme **********
    data_txt.write('classes=%s\n' % str(syn_args.class_num))
    data_txt.write('valid=./data_xview/{}_{}_cls/{}_{}_val_img.txt\n'.format(dt, syn_args.class_num, dt, syn_args.class_num))
    data_txt.write('valid_label=./data_xview/{}_{}_cls/{}_{}_val_lbl.txt\n'.format(dt, syn_args.class_num, dt, syn_args.class_num))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(syn_args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval={}'.format(dt))
    data_txt.close()


def create_mismatch_syn_labels(mis_ratio, ratio=0.25, trial=3):
    cat_distribution_map_syn = json.load(open(os.path.join(syn_args.syn_plane_txt_dir,
                                                           '{}_number_of_cat_0_to_imagenumber_map_inputsize{}.json'.format(
                                                               syn_args.syn_display_type, syn_args.tile_size))))
    plane_number = np.sum([v for v in cat_distribution_map_syn.values()])
    display_type = ['syn_texture', 'syn_color', 'syn_mixed']
    xview_num = 374
    for mr in mis_ratio:
        for dt in display_type:
            df_lbl = pd.read_csv(os.path.join(syn_args.data_xview_dir, 'xview_{}_{}'.format(dt, ratio), 'xview_{}_{}_train_lbl.txt'.format(dt, ratio)), header=None)
            syn_num = df_lbl.shape[0] - xview_num
            mis_num = int(plane_number*mr)
            np.random.seed(syn_args.seed)
            syn_rdn_indices = np.random.permutation(syn_num)
            syn_files = list(df_lbl.loc[xview_num:, 0])
            for t in range(trial):
                save_txt_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/syn_1_cls_xcycwh_mismatch/{}_{}_mismatch{}_{}/'.format(dt, ratio, mr, t)
                if not os.path.exists(save_txt_dir):
                    os.makedirs(save_txt_dir)
                for i in range(mis_num):
                    syn_f = syn_files[syn_rdn_indices[i]]
                    df_txt = pd.read_csv(syn_f, header=None, sep=' ')
                    file_name = os.path.basename(syn_files[syn_rdn_indices[i]])
                    df_txt.drop(df_txt.index[np.random.randint(0, df_txt.shape[0])])
                    df_txt.to_csv(os.path.join(save_txt_dir, file_name), header=None, index=None, sep=' ')

                    df_lbl.loc[df_lbl.loc[df_lbl[0] == syn_f].index, 0] = os.path.join(save_txt_dir, file_name)
                df_lbl.to_csv(os.path.join(syn_args.data_xview_dir, 'xview_{}_{}'.format(dt, ratio), 'xview_{}_{}_train_lbl_mismatch{}_{}.txt'.format(dt, ratio, mr, t)),
                              header=None, index=None)


def create_xview_syn_data(dt=None, sr=None, comments='',seed=1024):
    syn_args = get_part_syn_args(dt, sr)

    if sr:
        data_txt = open(os.path.join(syn_args.data_xview_dir, 'xview_{}_{}'.format(dt, sr), 'xview_{}_{}{}.data'.format(dt, sr, comments)), 'w')
        data_txt.write('train=./data_xview/{}_cls/xview_{}_{}/xview_{}_{}_train_img{}.txt\n'.format(syn_args.class_num, dt, sr, dt, sr, comments))
        data_txt.write('train_label=./data_xview/{}_cls/xview_{}_{}/xview_{}_{}_train_lbl{}.txt\n'.format(syn_args.class_num, dt, sr, dt, sr, comments))
        data_txt.write('valid=./data_xview/{}_cls/{}/xviewval_img{}.txt\n'.format(syn_args.class_num, comments[1:], comments))
        data_txt.write('valid_label=./data_xview/{}_cls/{}/xviewval_lbl{}.txt\n'.format(syn_args.class_num, comments[1:], comments))
    elif sr is None and dt is None:
        data_txt = open(os.path.join(syn_args.data_xview_dir, comments[1:], 'xview{}.data'.format(comments)), 'w')
        data_txt.write('train_label=./data_xview/{}_cls/{}/xviewtrain_lbl{}.txt\n'.format(syn_args.class_num, comments[1:], comments))
        data_txt.write('train=./data_xview/{}_cls/{}/xviewtrain_img{}.txt\n'.format(syn_args.class_num, comments[1:], comments))
        data_txt.write('valid=./data_xview/{}_cls/{}/xviewval_img{}.txt\n'.format(syn_args.class_num, comments[1:], comments))
        data_txt.write('valid_label=./data_xview/{}_cls/{}/xviewval_lbl{}.txt\n'.format(syn_args.class_num, comments[1:], comments))
    else:
        data_txt = open(os.path.join(syn_args.data_xview_dir, 'xview_{}_{}{}.data'.format(dt, sr, comments)), 'w')
        data_txt.write('train_label=./data_xview/{}_cls/xviewtrain_lbl{}.txt\n'.format(syn_args.class_num, comments))
        data_txt.write('train=./data_xview/{}_cls/xviewtrain_img{}.txt\n'.format(syn_args.class_num, comments))
        data_txt.write('valid=./data_xview/{}_cls/xviewval_img{}.txt\n'.format(syn_args.class_num, comments))
        data_txt.write('valid_label=./data_xview/{}_cls/xviewval_lbl{}.txt\n'.format(syn_args.class_num, comments))

    df = pd.read_csv(os.path.join(syn_args.data_xview_dir, comments[1:], 'xviewtrain_img{}.txt'.format(comments)), header=None)
    data_txt.write('syn_0_xview_number={}\n'.format(df.shape[0]))
    data_txt.write('classes=%s\n' % str(syn_args.class_num))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(syn_args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval=xview')
    data_txt.close()


def get_part_syn_args(dt, sr):
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_filepath", type=str, help="Filepath to GEOJSON coordinate file",
                        default='/media/lab/Yang/data/xView/xView_train.geojson')

    parser.add_argument("--syn_plane_img_anno_dir", type=str, help="images and annotations of synthetic airplanes",
                        default='/media/lab/Yang/data/synthetic_data/Airplanes/{}/')

    parser.add_argument("--syn_plane_txt_dir", type=str, help="txt labels of synthetic airplanes",
                        default='/media/lab/Yang/data/synthetic_data/Airplanes_txt_xcycwh/{}/')

    parser.add_argument("--syn_plane_gt_bbox_dir", type=str, help="gt images with bbox of synthetic airplanes",
                        default='/media/lab/Yang/data/synthetic_data/Airplanes_gt_bbox/{}/')

    parser.add_argument("--syn_images_save_dir", type=str, help="rgb images of synthetic airplanes",
                        default='/media/lab/Yang/data/xView_YOLO/images/{}_{}/')
    parser.add_argument("--syn_annos_save_dir", type=str, help="gt of synthetic airplanes",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_{}_cls_xcycwh/')
    parser.add_argument("--syn_txt_save_dir", type=str, help="gt related files of synthetic airplanes",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_{}_cls/')

    parser.add_argument("--data_txt_dir", type=str, help="to save txt files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_cls/')

    parser.add_argument("--data_xview_dir", type=str, help="to save data files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_cls/')

    parser.add_argument("--syn_data_list_dir", type=str, help="to syn data list files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_{}_cls/')

    parser.add_argument("--data_list_save_dir", type=str, help="to save selected trn val images and labels",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_cls/data_list/')

    parser.add_argument("--results_dir", type=str, help="to save category files",
                        default='/media/lab/Yang/code/yolov3/result_output/{}_cls/{}_{}/')

    parser.add_argument("--cat_sample_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/{}/{}_cls/')

    parser.add_argument("--val_percent", type=float, default=0.20,
                        help="Percent to split into validation (ie .25 = val set is 25% total)")
    parser.add_argument("--font3", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 10}")
    parser.add_argument("-ft2", "--font2", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 13}")

    parser.add_argument("--class_num", type=int, default=1, help="Number of Total Categories")  # 60  6 1
    parser.add_argument("--seed", type=int, default=17, help="random seed") #fixme -- 1024 17
    parser.add_argument("--tile_size", type=int, default=608, help="image size")  # 300 416

    parser.add_argument("--syn_display_type", type=str, default=dt,
                        help="syn_texture, syn_color, syn_mixed, syn_color0, syn_texture0, syn (match 0)")  # ######*********************change
    parser.add_argument("--syn_ratio", type=float, default=sr,
                        help="ratio of synthetic data: 0.25, 0.5, 0.75, 1.0  0")  # ######*********************change

    parser.add_argument("--min_region", type=int, default=100, help="the smallest #pixels (area) to form an object")
    parser.add_argument("--link_r", type=int, default=15,
                        help="the #pixels between two connected components to be grouped")

    parser.add_argument("--resolution", type=float, default=0.3, help="resolution of synthetic data")

    parser.add_argument("--cities", type=str,
                        default="['barcelona', 'berlin', 'francisco', 'hexagon', 'radial', 'siena', 'spiral']",
                        help="the synthetic data of cities")
    parser.add_argument("--streets", type=str, default="[200, 200, 200, 200, 200, 250, 130]",
                        help="the  #streets of synthetic  cities ")
    syn_args = parser.parse_args()
    syn_args.data_xview_dir = syn_args.data_xview_dir.format(syn_args.class_num)
    syn_args.data_txt_dir = syn_args.data_txt_dir.format(syn_args.tile_size, syn_args.class_num)
    syn_args.cat_sample_dir = syn_args.cat_sample_dir.format(syn_args.tile_size, syn_args.class_num)
    syn_args.data_list_save_dir = syn_args.data_list_save_dir.format(syn_args.tile_size, syn_args.class_num)

    syn_args.syn_images_save_dir = syn_args.syn_images_save_dir.format(syn_args.tile_size, dt)
    syn_args.syn_txt_save_dir = syn_args.syn_txt_save_dir.format(syn_args.tile_size, dt, syn_args.class_num)
    syn_args.syn_annos_save_dir = syn_args.syn_annos_save_dir.format(syn_args.tile_size, dt, syn_args.class_num)

    syn_args.syn_data_list_dir = syn_args.syn_data_list_dir.format(dt, syn_args.class_num)
    syn_args.results_dir = syn_args.results_dir.format(syn_args.tile_size, dt, sr)

    return syn_args


if __name__ == "__main__":

    # syn_args = get_part_syn_args()

    '''
    create chips and label txt and get all images json, convert from *.geojson to *.json
    
    get all images json, convert from *.geojson to *.json
    # convert_geojson_to_json()
    create label txt files
    # create_label_txt()
    
    '''
    # pwv.create_chips_and_txt_geojson_2_json(syn = True)

    '''
    cnt and get overlap bbox & remove duplicate bbox
    '''

    # cat_ids = [0]
    # pwv.cnt_ground_truth_overlap_from_pathces(cat_ids, syn = True)

    '''
    remove duplicate ground truth bbox by cat_id
    then manually replace the old duplicate annotation file with the new annotation file ************
    '''
    # # #
    # cat_id = 0
    # pwv.remove_duplicate_gt_bbx(cat_id, syn = True)

    '''
    backup xview-plane dataset ***********
    clean xview-plane dataset with certain constraints
    '''
    # whr_thres = 4  # 3
    # px_thres = 6  # 4
    # whr_thres = 4  # 3
    # px_thres = 20  # 4
    # whr_thres = 4  # 3
    # px_thres = 23  # 4
    # pwv.clean_backup_xview_plane_with_constraints(px_thres, whr_thres)

    # cat_ids = [1, 2, 3, 4]
    # whr_thres = 3 # 3.5
    # px_thres= 23
    # iou_thres = 0.5
    # # pwv.clean_backup_xview_plane_with_constraints(px_thres, whr_thres)
    # pwv.cnt_ground_truth_overlap_from_pathces(cat_ids, iou_thres, px_thres, whr_thres)

    ''' 
    remove duplicate bbox of xview aireplanes
    '''
    # cat_id = 4
    # px_thres = 23
    # whr_thres = 3
    # args = pwv.get_args(px_thres, whr_thres)
    # pwv.remove_duplicate_gt_bbx(cat_id, px_thres, whr_thres)

    ''' if necessary
    check xview aireplane dropped images
    '''
    # pwv.check_xview_plane_drops()

    ''' if necessary
    recover xview val list with constriants of px_theres=4, whr_thres=3
    regenerate_xview_val_list
    '''
    # pwv.recover_xview_val_list()

    ''' if necessary
    check difference of first and second dataset
    '''
    # check_difference_of_first_second_dataset()

    '''
    create xview.names 
    '''
    # file_name = 'xview'
    # pwv.create_xview_names(file_name)

    '''
    xview
    split train:val randomly split chips
    default split 
    '''
    # comments = ''
    # comments = '_px4whr3'
    # seed = 1024
    # comments = '_px6whr4_ng0'
    # seed = 17
    # comments = '_px6whr4_ng0_seed{}'
    # data_name = 'xview'

    # seed = 17
    # seeds = [17, 1024, 3, 5, 9]
    # comments = '_px20whr4_seed{}'
    # seeds = [17, 5, 9]
    # comments = '_px23whr4_seed{}'

    # seeds = [17]
    # data_name = 'xview'
    # px_thres= 23
    # whr_thres = 3
    # for sd in seeds:
    #     comments = '_px{}whr{}_seed{}'.format(px_thres, whr_thres, sd)
    #     pwv.split_trn_val_with_chips(data_name=data_name, comments=comments, seed=sd, px_thres=px_thres, whr_thres=whr_thres)


    '''
    xview
    create xview_* .data
    default split 
    '''
    # seeds = [17, 1024, 3, 5, 9]
    # comments = '_px20whr4_seed{}'
    # seeds = [17, 5, 9]
    # comments = '_px23whr4_seed{}'
    # seeds = [17]
    # for sd in seeds:
    #     comments = '_px23whr3_seed{}'.format(sd)
    #     create_xview_syn_data(comments=comments, seed=sd)

    '''
    xview
    create xview_* .data
    default split 
    '''
    # typestr='val'
    # px_thres = 23
    # whr_thres = 3
    # seed=17
    # comments = '_px{}whr{}_seed{}'.format(px_thres, whr_thres, seed)
    # data_name = 'xview'
    # pwv.create_json_for_train_or_val_according_to_all_json(data_name, comments, seed, px_thres, whr_thres, typestr)

    '''
    collect all syn images and txt into one file
    '''
    # data_name = '{}_{}'.format(args.syn_display_type, args.class_num)
    # collect_syn_data(data_name)

    '''
    combine xview & synthetic dataset [0.25, 0.5, 0.75]
    change syn_display_type manually ***
    '''
    # comments=''
    # comments = '_with_model'
    # comments = '_px4whr3'
    # comments = '_px6whr4_ng0'
    # syn_ratio = [0.25, 0.5, 0.75]
    # seed=1024
    # display_type = ['syn_texture', 'syn_color', 'syn_mixed']
    # for dt in display_type:
    #     for sr in syn_ratio:
    #         combine_xview_syn_by_ratio(dt, sr, comments, seed)

    '''
    create mismatch syn labels 
    '''
    # trial = 3
    # mis_ratio = [0.025, 0.05] # mismatch_ratio
    # ratio = 0.25 # syn ratio
    # create_mismatch_syn_labels(mis_ratio, ratio, trial)

    '''
    move the *.txt to des dir if necessary
    '''
    # mv_list()

    ''''
    create xview_syn_texture_0.25.data
    '''
    # comments = ''
    # display_type = ['syn_texture', 'syn_color', 'syn_mixed']
    # # comments = '_px4whr3'
    # comments = '_px6whr4'
    # comments = '_px6whr4_ng0'
    # seed = 1024
    # display_type = ['syn_texture0', 'syn_color0']
    # syn_ratio = [0.25, 0.5, 0.75]
    # for dt in display_type:
    #     for sr in syn_ratio:
    #         create_xview_syn_data(dt, sr, comments, trn_comments=False, seed=seed)

    # seed = 1024
    # create_xview_syn_data('syn', 0, comments, seed=seed)

    ''''
    create xview_syn_texture_0.25_mismatches*.data
    '''
    # syn_ratio = 0.25
    # display_type = ['syn_texture', 'syn_color', 'syn_mixed']
    # mis_ratio = [0.025, 0.05]
    # trial = 3
    # seed = 1024
    # for i in range(trial):
    #     for dt in display_type:
    #         for mr in mis_ratio:
    #             comments = '_mismatch{}_{}'.format(mr, i)
    #             create_xview_syn_data(dt, syn_ratio, comments)

    ''''
    create xview_syn_texture_*_with_model.data
    '''
    # display_type = ['syn_texture', 'syn_color', 'syn_mixed']
    # syn_ratio = [0.25, 0.5, 0.75]
    # comments = '_with_model'
    # seed = 1024
    # for dt in display_type:
    #     for sr in syn_ratio:
    #         create_xview_syn_data(dt, sr, comments)
    # seed = 1024
    # create_xview_syn_data('syn', 0)

    '''
    combine xview & syn_background dataset [0.1, 0.2, 0.3]
    change syn_display_type manually ***
    '''
    # comments = '_px6whr4_ng0_seed{}'
    # syn_ratio = [0.1, 0.2, 0.3]
    # dt = 'syn_background'
    # seed = 1024
    # seed = 17
    # for sr in syn_ratio:
    #     combine_xview_syn_by_ratio(dt, sr, comments, seed)

    ''''
    create xview_syn_background_*_px6whr4_ng0_seed{}.data
    xview_syn_background_*_px6whr4_ng0.data
    '''
    # syn_ratio = [0.1, 0.2, 0.3]
    # # comments = '_px6whr4_ng0'
    # seed = 1024
    # seed = 17
    # comments = '_px6whr4_ng0_seed{}'.format(17)
    # for sr in syn_ratio:
    #     create_xview_syn_data('syn_background', sr, comments, trn_comments=True, seed)

    # comments = '_px6whr4_ng0'
    # seed = 1024
    # comments = '_px6whr4_ng0_seed{}'
    # seed = 17
    # sr = 0
    # create_xview_syn_data('syn_background', sr, comments, trn_comments=True, seed=seed)

    '''
    split train and validation for syn_*_1_cls syn_only
    '''
    # display_types = ['syn_texture', 'syn_color', 'syn_mixed']
    # for dt in display_types:
    #     split_syn_trn_val(dt)

    '''
    create *.data syn_only
    '''
    # comments = 'syn_only'
    # display_types = ['syn_texture', 'syn_color', 'syn_mixed']
    # for dt in display_types:
    #     create_syn_data(dt, comments)








