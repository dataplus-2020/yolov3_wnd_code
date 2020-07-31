import glob
import numpy as np
import argparse
import os
from skimage import io, color
from ast import literal_eval
from matplotlib import pyplot as plt
import json
import shutil
from utils.xview_synthetic_util import process_wv_coco_for_yolo_patches_for_aircraft_split as pwv

IMG_FORMAT0 = '.jpg'
IMG_FORMAT = '.png'
TXT_FORMAT = '.txt'


def merge_clean_origin_syn_bkg_image_files(file_path, cities, streets, tile_size=608, resolution=0.3, white_thresh=0.5):
    '''
    merge all the origin synthetic data into one folder
    then remove rgb images those contain more than white_thresh*100% white pixels
    :param file_path:
    :param cities:
    :param streets:
    :param tile_size:
    :param resolution:
    :return:
    '''
    step = tile_size * resolution
    image_folder_name = '{}_{}_{}_images_step{}'

    new_img_folder = '{}_all_images_step{}'.format(syn_args.syn_display_type, step)
    new_lbl_folder = '{}_all_annos_step{}'.format(syn_args.syn_display_type, step)
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

    for i in range(len(cities)):
        image_path = os.path.join(file_path,
                                  image_folder_name.format(syn_args.syn_display_type, cities[i], streets[i], step))
        image_files = np.sort(glob.glob(os.path.join(image_path, '*{}'.format(IMG_FORMAT))))
        for img in image_files:
            shutil.copy(img, des_img_path)

    all_images = np.sort(glob.glob(os.path.join(des_img_path, '*{}'.format(IMG_FORMAT))))
    for ix, f in enumerate(all_images):
        img = io.imread(f)
        img = color.rgb2gray(img)
        white_num = np.sum(img == 1)
        white_ratio = white_num / img.shape[0] / img.shape[1]

        if white_ratio > white_thresh:
            os.remove(f)
        else:
            shutil.copy(f, os.path.join(syn_args.syn_images_save_dir, os.path.basename(f)))

            txt_name = os.path.basename(f).replace(IMG_FORMAT, TXT_FORMAT)
            f_txt = open(os.path.join(des_lbl_path, txt_name), 'w')
            f_txt.close()
            shutil.copy(os.path.join(des_lbl_path, txt_name), os.path.join(syn_args.syn_annos_save_dir, txt_name))


def get_part_syn_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--syn_bkg_img_anno_dir", type=str, help="images and annotations of synthetic airplanes",
                        default='/media/lab/Yang/data/synthetic_data/syn_background/')

    parser.add_argument("--syn_bkg_txt_dir", type=str, help="txt labels of synthetic airplanes",
                        default='/media/lab/Yang/data/synthetic_data/syn_background_txt_xcycwh/{}/')

    parser.add_argument("--syn_bkg_gt_bbox_dir", type=str, help="gt images with bbox of synthetic airplanes",
                        default='/media/lab/Yang/data/synthetic_data/syn_background_gt_bbox/{}/')

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

    parser.add_argument("--class_num", type=int, default=1, help="Number of Total Categories")  # 60  6
    parser.add_argument("--tile_size", type=int, default=608, help="image size")  # 300 416

    parser.add_argument("--syn_display_type", type=str, default='syn_background',
                        help="syn_background")  # ######*********************change
    parser.add_argument("--syn_ratio", type=float, default=0.75,
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
    syn_args.cities = literal_eval(syn_args.cities)
    syn_args.streets = literal_eval(syn_args.streets)
    syn_args.data_xview_dir = syn_args.data_xview_dir.format(syn_args.class_num)
    syn_args.data_txt_dir = syn_args.data_txt_dir.format(syn_args.tile_size, syn_args.class_num)
    syn_args.cat_sample_dir = syn_args.cat_sample_dir.format(syn_args.tile_size, syn_args.class_num)
    if not os.path.exists(syn_args.data_xview_dir):
        os.makedirs(syn_args.data_xview_dir)
    if not os.path.exists(syn_args.data_txt_dir):
        os.makedirs(syn_args.data_txt_dir)
    if not os.path.exists(syn_args.cat_sample_dir):
        os.makedirs(syn_args.cat_sample_dir)

    syn_args.syn_images_save_dir = syn_args.syn_images_save_dir.format(syn_args.tile_size, syn_args.syn_display_type)
    syn_args.syn_annos_save_dir = syn_args.syn_annos_save_dir.format(syn_args.tile_size, syn_args.syn_display_type, syn_args.class_num)
    syn_args.syn_txt_save_dir = syn_args.syn_txt_save_dir.format(syn_args.tile_size, syn_args.syn_display_type, syn_args.class_num)
    if not os.path.exists(syn_args.syn_images_save_dir):
        os.makedirs(syn_args.syn_images_save_dir)
    if not os.path.exists(syn_args.syn_txt_save_dir):
        os.makedirs(syn_args.syn_txt_save_dir)
    if not os.path.exists(syn_args.syn_annos_save_dir):
        os.makedirs(syn_args.syn_annos_save_dir)

    return syn_args


if __name__ == '__main__':
    syn_args = get_part_syn_args()

    # white_thresh = 0.5
    # merge_clean_origin_syn_bkg_image_files(syn_args.syn_bkg_img_anno_dir, syn_args.cities, syn_args.streets,
    #                                         syn_args.tile_size, syn_args.resolution, white_thresh)

    '''
    xview with syn_background
    split train:val randomly split chips
    default split 
    '''
    # comments = '_px6whr4_ng0'
    # seed = [3, 5, 9]
    # data_name = 'xview'
    # for s in seed:
    #     comments = '_px6whr4_ng0_seed{}'.format(s)
    #     pwv.split_trn_val_with_chips(data_name, comments, s)

    '''
    combine xview & syn_background dataset [0.1, 0.2, 0.3]
    '''
    # seed = [3, 5, 9]
    # syn_ratio = [0.1, 0.2, 0.3]
    # dt = 'syn_background'
    # for s in seed:
    #     comments = '_px6whr4_ng0_seed{}'.format(s)
    #     for sr in syn_ratio:
    #         pwv.combine_xview_syn_by_ratio(dt, sr, comments, seed=s)

    ''''
    create xview_syn_background_*_px6whr4_ng0_seed{}.data
    xview_syn_background_*_px6whr4_ng0.data
    '''
    # comments = '_px6whr4_ng0_seed{}'
    # seed = [3, 5, 9]
    # syn_ratio = [0.1, 0.2, 0.3, 0]
    # for s in seed:
    #     for sr in syn_ratio:
    #         pwv.create_xview_syn_data('syn_background', sr, comments, trn_comments=True, seed=s)
