'''
creater xuyang_ustb@163.com
xview process
in order to generate xivew 2-d background with synthetic airplances
'''
import glob
import numpy as np
import argparse
import os
import pandas as pd
import shutil
from utils.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc

IMG_FORMAT = '.png'
TXT_FORMAT = '.txt'


def merge_clean_origin_syn_image_files(ct, st, dt):
    '''
    merge all the origin synthetic data into one folder
    then remove rgb images those contain more than white_thresh*100% white pixels
    and remove gt images that are all white pixels
    :return:
    '''
    step = syn_args.tile_size * syn_args.resolution
    image_folder_name = 'syn_{}_{}_{}_images_step{}'.format(dt, ct, st, step)
    label_folder_name = 'syn_{}_{}_{}_annos_step{}'.format(dt, ct, st, step)
    file_path = syn_args.syn_data_dir

    new_img_folder = '{}_all_images_step{}'.format(dt, step)
    new_lbl_folder = '{}_all_annos_step{}'.format(dt, step)
    des_img_path = os.path.join(file_path, new_img_folder)
    des_lbl_path = os.path.join(file_path, new_lbl_folder)
    if not os.path.exists(des_img_path):
        os.mkdir(des_img_path)
    else:
        shutil.rmtree(des_img_path)
        os.mkdir(des_img_path)
    if not os.path.exists(des_lbl_path):
        os.mkdir(des_lbl_path)
    else:
        shutil.rmtree(des_lbl_path)
        os.mkdir(des_lbl_path)

    image_path = os.path.join(file_path, image_folder_name)
    image_files = np.sort(glob.glob(os.path.join(image_path, '*{}'.format(IMG_FORMAT))))
    for img in image_files:
        shutil.copy(img, des_img_path)

    lbl_path = os.path.join(file_path, label_folder_name)
    lbl_files = np.sort(glob.glob(os.path.join(lbl_path, '*{}'.format(IMG_FORMAT))))

    for lbl in lbl_files:
        shutil.copy(lbl, des_lbl_path)


def group_object_annotation_and_draw_bbox(dt, px_thresh=20, whr_thres=4):
    '''
    px_thres: threshold for the length of edge lenght of b-box (at the margin)
    whr_thres: threshold for width/height or height/width
    group annotation files, generate bbox for each object,

    and draw bbox for each ground truth files
    '''
    step = syn_args.tile_size * syn_args.resolution
    folder_name = '{}_all_annos_step{}'.format(dt, step)
    lbl_path = os.path.join(syn_args.syn_data_dir, folder_name)
    txt_folder_name = 'minr{}_linkr{}_px{}whr{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, px_thresh, whr_thres,
                                                                      dt, step)
    save_txt_path = os.path.join(syn_args.syn_annos_dir, txt_folder_name)
    if not os.path.exists(save_txt_path):
        os.makedirs(save_txt_path)
    else:
        shutil.rmtree(save_txt_path)
        os.makedirs(save_txt_path)

    gbc.get_object_bbox_after_group(lbl_path, save_txt_path, class_label=0, min_region=syn_args.min_region,
                                    link_r=syn_args.link_r, px_thresh=px_thresh, whr_thres=whr_thres)
    gt_files = np.sort(glob.glob(os.path.join(lbl_path, '*{}'.format(IMG_FORMAT))))
    bbox_folder_name = 'minr{}_linkr{}_px{}whr{}_{}_all_annos_with_bbox_step{}'.format(syn_args.min_region, syn_args.link_r,
                                                                                       px_thresh, whr_thres, dt, step)
    save_bbx_path = os.path.join(syn_args.syn_txt_dir, bbox_folder_name)
    if not os.path.exists(save_bbx_path):
        os.makedirs(save_bbx_path)
    else:
        shutil.rmtree(save_bbx_path)
        os.makedirs(save_bbx_path)
    for g in gt_files:
        gt_name = g.split('/')[-1]
        txt_name = gt_name.replace(IMG_FORMAT, TXT_FORMAT)
        txt_file = os.path.join(save_txt_path, txt_name)
        gbc.plot_img_with_bbx(g, txt_file, save_bbx_path)


def draw_bbx_on_rgb_images(dt, px_thresh=20, whr_thres=4):
    step = syn_args.tile_size * syn_args.resolution
    img_folder_name = '{}_all_images_step{}'.format(dt, step)
    img_path = os.path.join(syn_args.syn_data_dir, img_folder_name)
    img_files = np.sort(glob.glob(os.path.join(img_path, '*{}'.format(IMG_FORMAT))))
    img_names = [os.path.basename(f) for f in img_files]

    txt_folder_name = 'minr{}_linkr{}_px{}whr{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, px_thresh, whr_thres,
                                                                                dt, step)
    annos_path = os.path.join(syn_args.syn_annos_dir, txt_folder_name)

    bbox_folder_name = 'minr{}_linkr{}_px{}whr{}_{}_all_images_with_bbox_step{}'.format(syn_args.min_region, syn_args.link_r, px_thresh, whr_thres,
                                                                              dt, step)
    save_bbx_path = os.path.join(syn_args.syn_txt_dir, bbox_folder_name)
    if not os.path.exists(save_bbx_path):
        os.makedirs(save_bbx_path)
    else:
        shutil.rmtree(save_bbx_path)
        os.makedirs(save_bbx_path)

    for ix, f in enumerate(img_files):
        txt_file = os.path.join(annos_path, img_names[ix].replace(IMG_FORMAT, TXT_FORMAT))
        gbc.plot_img_with_bbx(f, txt_file, save_bbx_path, label_index=False)


def split_syn_xview_background_trn_val(seed=17, comment='syn_color', pxwhr='px23whr3'):

    display_type = comment.split('_')[-1]
    step = syn_args.tile_size * syn_args.resolution
    all_files = np.sort(glob.glob(os.path.join(syn_args.syn_data_dir, '{}_all_images_step{}'.format(display_type, step), '*' + IMG_FORMAT)))
    num_files = len(all_files)

    np.random.seed(seed)
    all_indices = np.random.permutation(num_files)
    data_txt_dir = syn_args.syn_txt_dir

    trn_img_txt = open(os.path.join(data_txt_dir, '{}_train_img_seed{}.txt'.format(comment, seed)), 'w')
    trn_lbl_txt = open(os.path.join(data_txt_dir, '{}_train_lbl_seed{}.txt'.format(comment, seed)), 'w')

    val_img_txt = open(os.path.join(data_txt_dir, '{}_val_img_seed{}.txt'.format(comment, seed)), 'w')
    val_lbl_txt = open(os.path.join(data_txt_dir, '{}_val_lbl_seed{}.txt'.format(comment, seed)), 'w')

    num_val = int(num_files*syn_args.val_percent)
    lbl_dir = os.path.join(syn_args.syn_annos_dir, 'minr{}_linkr{}_{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, pxwhr, display_type, step))
    for i in all_indices[:num_val]:
        val_img_txt.write('%s\n' % all_files[i])
        val_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(all_files[i]).replace(IMG_FORMAT, TXT_FORMAT)))
    val_img_txt.close()
    val_lbl_txt.close()
    for j in all_indices[num_val:]:
        trn_img_txt.write('%s\n' % all_files[j])
        trn_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(all_files[j]).replace(IMG_FORMAT, TXT_FORMAT)))
    trn_img_txt.close()
    trn_lbl_txt.close()


def create_syn_data(comment='syn_texture', seed=17):
    data_txt_dir = syn_args.syn_txt_dir
    dt = comment.split('_')[-1]

    data_txt = open(os.path.join(data_txt_dir, '{}_seed{}.data'.format(comment, seed)), 'w')
    data_txt.write('train={}/{}_train_img_seed{}.txt\n'.format(data_txt_dir, comment, seed))
    data_txt.write('train_label={}/{}_train_lbl_seed{}.txt\n'.format(data_txt_dir, comment, seed))

     #********** syn_0_xview_number corresponds to train*.py the number of train files
    df = pd.read_csv(os.path.join(syn_args.syn_txt_dir, '{}_train_img_seed{}.txt'.format(comment, seed)), header=None)
    data_txt.write('syn_0_xview_number={}\n'.format(df.shape[0]))
    data_txt.write('classes=%s\n' % str(syn_args.class_num))

    data_txt.write('valid={}/{}_val_img_seed{}.txt\n'.format(data_txt_dir, comment, seed))
    data_txt.write('valid_label={}/{}_val_lbl_seed{}.txt\n'.format(data_txt_dir, comment, seed))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(syn_args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval={}'.format(dt))
    data_txt.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--syn_data_dir", type=str,
                        help="Path to folder containing synthetic images and annos ",
                        default='/home/jovyan/work/data/')
    parser.add_argument("--syn_annos_dir", type=str, default='/home/jovyan/work/data/syn_background_txt_xcycwh/',
                        help="syn label.txt")
    parser.add_argument("--syn_txt_dir", type=str, default='/home/jovyan/work/data/syn_background_gt_bbox/',
                        help="syn related txt files")

    parser.add_argument("--syn_display_type", type=str, default='color',
                        help="texture, color, mixed")  # syn_color0, syn_texture0,
    #fixme ---***** min_region ***** change
    parser.add_argument("--min_region", type=int, default=100, help="300 100 the smallest #pixels (area) to form an object")
    parser.add_argument("--link_r", type=int, default=15,
                        help="the #pixels between two connected components to be grouped")
    parser.add_argument("--resolution", type=float, default=0.3, help="resolution of synthetic data")
    parser.add_argument("--tile_size", type=int, default=608, help="image size")
    parser.add_argument("--class_num", type=int, default=1, help="class number")
    parser.add_argument("--val_percent", type=float, default=0.25, help="train:val=0.75:0.25")


    args = parser.parse_args()
    if not os.path.exists(args.syn_annos_dir):
        os.makedirs(args.syn_annos_dir)
    if not os.path.exists(args.syn_txt_dir):
        os.makedirs(args.syn_txt_dir)

    return args


if __name__ == '__main__':

    '''
    merge all syn_xveiw_background data
    *****---change syn_data_dir first----******
    '''
    cities = ['siena']
    streets = [500]
    display_types = ['mixed']
    syn_args = get_args()
    for dt in display_types:
        for cx, ct in enumerate(cities):
            st = streets[cx]
            merge_clean_origin_syn_image_files(ct, st, dt)


    '''
    generate txt and bbox for syn_background data
    bbox annotation meet certain conditions: px_thres, whr_thres
    '''

    px_thres=23
    whr_thres=3
    display_types = [ 'mixed']
    syn_args = get_args()
    for dt in display_types:
        group_object_annotation_and_draw_bbox(dt, px_thres, whr_thres)

    '''
    draw bbox on rgb images for syn_background data
    '''
    px_thres=23 #20 #30
    whr_thres=3
    display_types = ['mixed']
    syn_args = get_args()
    for dt in display_types:
        draw_bbx_on_rgb_images(dt, px_thres, whr_thres)

    '''
    split train val
    '''
    comments = ['syn_mixed']
    pxwhr = 'px23whr3'

    base_pxwhrs = 'px23whr3_seed{}'
    syn_args = get_args()
    seed = 17
    for cmt in comments:
        base_pxwhrs = base_pxwhrs.format(seed)
        split_syn_xview_background_trn_val(seed, cmt, pxwhr)

    '''
    create *.data
    '''
    comments = ['syn_mixed']
    for cmt in comments:
        create_syn_data(cmt, seed=17)


