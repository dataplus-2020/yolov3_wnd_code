'''
xview process
in order to generate xivew 2-d background with synthetic airplances
'''
import glob
import numpy as np
import argparse
import os
from PIL import Image
import pandas as pd
import shutil
from utils.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc
from skimage import io
import matplotlib.pyplot as plt
import cv2

IMG_FORMAT = '.png'
TXT_FORMAT = '.txt'


def merge_clean_origin_syn_image_files(seedians, dt):
    '''
    merge all the origin synthetic data into one folder
    then remove rgb images those contain more than white_thresh*100% white pixels
    and remove gt images that are all white pixels
    :return:
    '''
    step = syn_args.tile_size * syn_args.resolution
    image_folder_name = 'syn_xview_background_images_{}_step{}_sd{}'
    label_folder_name = 'syn_xview_background_annos_{}_step{}_sd{}'
    file_path = syn_args.syn_data_dir.format(dt)

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

    for sd in seedians:
        image_path = os.path.join(file_path, image_folder_name.format(dt, step, sd))
        image_files = np.sort(glob.glob(os.path.join(image_path, '*{}'.format(IMG_FORMAT))))
        for img in image_files:
            shutil.copy(img, des_img_path)

        lbl_path = os.path.join(file_path, label_folder_name.format(dt, step, sd))
        lbl_files = np.sort(glob.glob(os.path.join(lbl_path, '*{}'.format(IMG_FORMAT))))

        for lbl in lbl_files:
            shutil.copy(lbl, des_lbl_path)


def group_object_annotation_and_draw_bbox(dt, px_thresh=20, whr_thres=4):
    '''
    group annotation files, generate bbox for each object,

    and draw bbox for each ground truth files
    '''
    step = syn_args.tile_size * syn_args.resolution
    folder_name = '{}_all_annos_step{}'.format(dt, step)
    lbl_path = os.path.join(syn_args.syn_data_dir.format(dt), folder_name)
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
    img_path = os.path.join(syn_args.syn_data_dir.format(dt), img_folder_name)
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


def plot_rgb_histogram(img_path, syn=False):
    save_dir = '/media/lab/Yang/data/xView_YOLO/cat_samples/608/1_cls/rgb_histogram/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if syn:
        x_files = glob.glob(os.path.join(img_path, '*.png'))
        name = img_path.split('/')[-3] + '.png'
        title = '{} RGB Histogram'.format(img_path.split('/')[-3])
    else:
        x_files = glob.glob(os.path.join(img_path, '*.jpg'))
        name = 'xview_rgb_histogram.jpg'
        title = 'Xview RGB Histogram'
    arr_hist = np.zeros(shape=(3, 256, len(x_files)))
    color = ('b','g','r')
    for ix, f in enumerate(x_files):
        img = cv2.imread(f)
        for i,col in enumerate(color):
            # (256, 1)
            histr = cv2.calcHist([img],[i],None,[256],[0,256])
            arr_hist[i, :, ix] = histr[:, 0]

    hs = arr_hist.mean(axis=-1)
    print(hs.shape)
    for cx, c in enumerate(color):
        plt.plot(hs[cx, :],color = c)

    plt.xlim([0,256])
    plt.xlabel('Intensity Value')
    plt.ylabel('Count')
    plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.suptitle(title)
    plt.savefig(os.path.join(save_dir, name))
    plt.show()


def get_args(cmt=''):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str,
                        help="Path to folder containing image chips (ie 'Image_Chips/') ",
                        default='/media/lab/Yang/data/xView/train_images/')
    parser.add_argument("--raw_folder", type=str,
                        help="Path to folder containing raw images ",
                        default='/media/lab/Yang/data/xView/')

    #fixme
    if cmt: # certain_models, scale_models
        parser.add_argument("--syn_data_dir", type=str,
                        help="Path to folder containing synthetic images and annos ",
                        default='/media/lab/Yang/data/synthetic_data/syn_xview_bkg_{}'.format(cmt)+'_{}/')
        parser.add_argument("--syn_annos_dir", type=str, default='/media/lab/Yang/data/synthetic_data/syn_xview_bkg_{}_txt_xcycwh/'.format(cmt),
                            help="syn xview txt")
        parser.add_argument("--syn_txt_dir", type=str, default='/media/lab/Yang/data/synthetic_data/syn_xview_bkg_{}_gt_bbox/'.format(cmt),
                            help="syn xview txt related files")
    else: # cmt == ''
        parser.add_argument("--syn_data_dir", type=str,
                            help="Path to folder containing synthetic images and annos ",
                            default='/media/lab/Yang/data/synthetic_data/syn_xview_background_{}/')
        parser.add_argument("--syn_annos_dir", type=str, default='/media/lab/Yang/data/synthetic_data/syn_xview_background_txt_xcycwh',
                            help="syn xview txt")
        parser.add_argument("--syn_txt_dir", type=str, default='/media/lab/Yang/data/synthetic_data/syn_xview_background_gt_bbox',
                            help="syn xview txt related files")

    parser.add_argument("--syn_display_type", type=str, default='color',
                        help="texture, color, mixed")  # syn_color0, syn_texture0,
    #fixme ---***** min_region ***** change
    parser.add_argument("--min_region", type=int, default=100, help="300 100 the smallest #pixels (area) to form an object")
    parser.add_argument("--link_r", type=int, default=15,
                        help="the #pixels between two connected components to be grouped")
    parser.add_argument("--resolution", type=float, default=0.3, help="resolution of synthetic data")
    parser.add_argument("--tile_size", type=int, default=608, help="image size")
    parser.add_argument("--class_num", type=int, default=1, help="class number")

    args = parser.parse_args()
    # if not os.path.exists(args.syn_annos_dir):
    #     os.makedirs(args.syn_annos_dir)
    # if not os.path.exists(args.syn_txt_dir):
    #     os.makedirs(args.syn_txt_dir)

    return args


if __name__ == '__main__':

    '''
    get raw images contain more than 30% black pixels
    get tif those contain airplanes
    get no airplane raw images
    get shape raw images
    '''
    # get_black_raw_img_list()

    # get_tif_contain_airplanes()

    # get_no_airplane_raw_images()

    # get_shape_raw_images()

    '''
    merge all syn_xveiw_background data
    *****---change syn_data_dir first----******
    '''
    cmt = ''
    seedians = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    display_types = ['texture', 'color', 'mixed']
    # cmt = 'certain_models'
    # display_types = ['texture', 'color', 'mixed']
    # seedians = [0, 1, 2, 3, 4, 5, 6, 7]
    # cmt = 'scale_models'
    # display_types = ['color', 'mixed']
    # seedians = [0, 1, 2, 3, 4, 5, 6, 7]
    # cmt = 'small_models'
    # # # cmt = 'small_fw_models'
    # cmt = '6groups_models'
    # cmt = 'rnd_bwratio_models'
    # cmt = 'rnd_bwratio_flat0.8_models'
    # cmt = 'rnd_bwratio_asx_models'
    # cmt = 'xratio_xcolor_models'
    # cmt = 'sbwratio_xratio_xcolor_models'
    # cmt = 'sbwratio_xratio_xcolor_dark_models'
    # cmt = 'sbwratio_new_xratio_xcolor_models'
    # display_types = ['color', 'mixed']
    # seedians = [0, 1, 2, 3, 4]
    syn_args = get_args(cmt)
    for dt in display_types:
        merge_clean_origin_syn_image_files(seedians, dt)

    '''
    draw RGB histogram 
    '''
    # xview_patch_dir = '/media/lab/Yang/data/xView_YOLO/images/608_1cls/'
    # plot_rgb_histogram(xview_patch_dir, syn=False)

    # syn_patch_dir = '/media/lab/Yang/data/synthetic_data/syn_xview_bkg_sbwratio_xratio_xcolor_models_color/color_all_images_step182.4/'
    # syn_patch_dir = '/media/lab/Yang/data/synthetic_data/syn_xview_bkg_sbwratio_xratio_xcolor_models_mixed/mixed_all_images_step182.4/'
    # syn_patch_dir = '/media/lab/Yang/data/synthetic_data/syn_xview_bkg_sbwratio_xratio_xcolor_dark_models_color/color_all_images_step182.4/'
    # syn_patch_dir = '/media/lab/Yang/data/synthetic_data/syn_xview_bkg_sbwratio_xratio_xcolor_dark_models_mixed/mixed_all_images_step182.4/'
    # syn_patch_dir = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_color/syn_color_all_images_step182.4/'
    # syn_patch_dir = '/media/lab/Yang/data/synthetic_data/syn_xview_bkg_sbwratio_new_xratio_xcolor_models_mixed/mixed_all_images_step182.4/'
    # syn_patch_dir = '/media/lab/Yang/data/synthetic_data/syn_xview_bkg_sbwratio_new_xratio_xcolor_models_mixed/mixed_all_images_step182.4/'
    # plot_rgb_histogram(syn_patch_dir, syn=True)

    '''
    generate txt and bbox for syn_xveiw_background data
    bbox annotation meet certain conditions: px_thres, whr_thres
    '''
    # # px_thres=6
    # # whr_thres=4
    # # px_thres=20
    # # whr_thres=4
    # px_thres=23 #20 #30
    # whr_thres=4
    # # # display_types = ['texture', 'color', 'mixed']
    # # # # cmt = ''
    # # # # cmt = 'certain_models'
    # # # cmt = 'scale_models'
    # cmt = 'small_models'
    # # cmt = 'small_fw_models'
    # # # cmt = '6groups_models'
    # # cmt = 'rnd_bwratio_models'
    # cmt = 'rnd_bwratio_flat0.8_models'
    # cmt = 'rnd_bwratio_asx_models'
    # cmt = 'xratio_xcolor_models'
    # cmt = 'sbwratio_xratio_xcolor_models'
    # cmt = 'sbwratio_new_xratio_xcolor_models'
    # cmt = 'xbw_xrxc_spr_sml_models'
    # cmt = 'sbw_xcolor_model4'
    # cmt = 'sbw_xcolor_model4_v1'
    # cmt = 'sbw_xcolor_model4_v2'
    # cmt = 'xbw_xcolor_xbkg_gauss_model4_v3'
    # px_thres=15 #20 #30
    # cmt = 'sbw_xcolor_model0'
    # cmt = 'xbw_xrxc_spr_sml_models_gauss'
    # cmt = 'xbw_xrxc_spr_sml_models_gauss'
    # cmt = 'sbw_xcolor_model1'
    # cmt = 'xbw_xcolor_gauss_model1_v1'
    # cmt = 'xbw_xcolor_xbkg_gauss_model1_v2'
    # cmt = 'sbw_xcolor_xbkg_unif_model1_v3'
    cmt = 'xbsw_xcolor_xbkg_gauss_model1_v4'
    px_thres=23

    whr_thres=3
    display_types = ['color', 'mixed']
    syn_args = get_args(cmt)
    for dt in display_types:
        group_object_annotation_and_draw_bbox(dt, px_thres, whr_thres)

    '''
    draw bbox on rgb images for syn_xveiw_background data
    '''
    # # px_thres=6
    # # whr_thres=4
    # # px_thres=20
    # # whr_thres=4
    # px_thres=23
    # whr_thres=4
    # # display_types = ['texture', 'color', 'mixed']
    # # cmt = 'certain_models'
    # # cmt = 'scale_models'
    # cmt = 'small_models'
    # # cmt = 'small_fw_models'

    # cmt = '6groups_models'
    # cmt = 'rnd_bwratio_models'
    # cmt = 'rnd_bwratio_flat0.8_models'
    # cmt = 'rnd_bwratio_asx_models'
    # cmt = 'xratio_xcolor_models'
    # cmt = 'sbwratio_xratio_xcolor_models'
    # cmt = 'sbwratio_new_xratio_xcolor_models'
    # cmt = 'xbw_xcolor_model0'
    # # cmt = 'xbw_xrxc_spr_sml_models'

    # cmt = 'sbw_xcolor_model4'
    # cmt = 'sbw_xcolor_model4_v1'
    # cmt = 'sbw_xcolor_model4_v2'
    # cmt = 'xbw_xcolor_xbkg_gauss_model4_v3'
    # px_thres=15
    # cmt = 'sbw_xcolor_model0'
    # cmt = 'sbw_xcolor_model1'
    # cmt = 'xbw_xrxc_spr_sml_models_gauss'
    # cmt = 'xbw_xrxc_gauss_model1_v1'
    # cmt = 'xbw_xcolor_xbkg_gauss_model1_v2'
    # cmt = 'sbw_xcolor_xbkg_unif_model1_v3'
    cmt = 'xbsw_xcolor_xbkg_gauss_model1_v4'
    px_thres=23 #20 #30

    whr_thres=3
    display_types = ['color', 'mixed']
    syn_args = get_args(cmt)
    for dt in display_types:
        draw_bbx_on_rgb_images(dt, px_thres, whr_thres)






# 0 100
# 1 42
# 1 52

# 1 71
# 1 72

