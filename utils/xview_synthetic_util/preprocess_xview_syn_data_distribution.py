import glob
import numpy as np
import argparse
import os
from skimage import io, color
from utils.data_process_distribution_vis_util import process_wv_coco_for_yolo_patches_no_trnval as pwv
from utils.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc
import pandas as pd
from ast import literal_eval
from matplotlib import pyplot as plt
import scipy.stats as scs
from scipy.optimize import curve_fit
import json
import shutil
import cv2

IMG_FORMAT0 = '.jpg'
IMG_FORMAT = '.png'
TXT_FORMAT = '.txt'


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


def autolabel(ax, rects, x, labels, ylabel=None, rotation=90, txt_rotation=45):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height) if height != 0 else '',
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
        # ax.grid(True)


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def analyze_category_distribution(txt_path, json_number_file, json_name_file, px_thresh=6, whr_thres=4):
    txt_files = np.sort(glob.glob(os.path.join(txt_path, "*" + TXT_FORMAT)))
    plane_number_2_img_number_map = {}
    plane_number_2_txt_name_map = {}
    for i in range(len(txt_files)):
        if not is_non_zero_file(txt_files[i]):
            continue

        df_txt = pd.read_csv(txt_files[i], header=None, delimiter=' ')
        for ix in df_txt.index:
            df_txt.loc[ix, 1:] = df_txt.loc[ix, 1:] * syn_args.tile_size
            whr = np.maximum(df_txt.loc[ix, 3] / df_txt.loc[ix, 4], df_txt.loc[ix, 4] / df_txt.loc[ix, 3])
            if int(df_txt.loc[ix, 3]) <= px_thresh or int(df_txt.loc[ix, 4]) <= px_thresh or whr > whr_thres:
                df_txt = df_txt.drop(ix)
            else:
                df_txt.loc[ix, 1:] = df_txt.loc[ix, 1:] / syn_args.tile_size
        # df_txt = pd.read_csv(txt_files[i], header=None, delimiter=' ').to_numpy()
        # df_txt[:, 1:] = df_txt[:, 1:] * syn_args.tile_size
        # #fixme astype(np.int) ****
        # df_txt[:, 1:] = df_txt[:, 1:].astype(np.int)
        #
        # df_txt = df_txt[df_txt[:, 3] > px_thresh]
        # df_txt = df_txt[df_txt[:, 4] > px_thresh]
        # df_wh = np.vstack((df_txt[:, 3] / df_txt[:, 4], df_txt[:, 4] / df_txt[:, 3]))
        # df_txt = df_txt[np.max(df_wh, axis=0) <= whr_thres]
        # print(df_txt.shape)
        if df_txt.shape[0] == 0:
            continue
        plane_number = df_txt.shape[0]
        if plane_number not in plane_number_2_img_number_map.keys():
            plane_number_2_img_number_map[plane_number] = 1
            plane_number_2_txt_name_map[plane_number] = [txt_files[i].split('/')[-1]]
        else:
            plane_number_2_img_number_map[plane_number] += 1
            plane_number_2_txt_name_map[plane_number].append(txt_files[i].split('/')[-1])
    json.dump(plane_number_2_img_number_map, open(json_number_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)
    json.dump(plane_number_2_txt_name_map, open(json_name_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)


def draw_bar_of_image_numbers_for_certain_number_of_planes(png_name, cat_distribution_map, legend='xView', syn=False):
    '''
    x-axis: the number of planes
    y-axis: the number of images
    :param catid:
    :param cat_distribution_map:
    :return:
    '''
    if syn:
        save_dir = syn_args.syn_plane_txt_dir + 'data_distribution_fig/'
        args = syn_args
    else: # xview
        args = pwv.get_args()
        save_dir = args.txt_save_dir + 'data_distribution_fig/'

    num_planes = np.array([int(k) for k in cat_distribution_map.keys()])
    num_planes_sort_indices = np.argsort(num_planes)
    num_images = np.array([v for v in cat_distribution_map.values()])


    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    title = "Airplane Distribution (chips)"
    labels = num_planes[num_planes_sort_indices]

    x = num_planes[num_planes_sort_indices]
    ylist = num_images[num_planes_sort_indices]

    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    fig, ax = plt.subplots(1, 1)
    width = 0.35
    rects = ax.bar(np.array(x) - width / 2, ylist, width, label=legend)  # , label=labels
    autolabel(ax, rects, x, labels, ylist, rotation=0)
    ax.legend()
    xlabel = 'Number of Airplanes'
    ylabel = "Number of Images"
    plt.title(title, literal_eval(args.font2))
    plt.ylabel(ylabel, literal_eval(args.font2))
    plt.xlabel(xlabel, literal_eval(args.font2))
    plt.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.grid()
    plt.savefig(os.path.join(save_dir, png_name))
    plt.show()


def compare_bar_of_image_numbers_for_certain_number_of_planes(png_name, cat_distribution_map, cat_distribution_map_syn):
    '''
    x-axis: the number of planes
    y-axis: the number of images
    :param catid:
    :param cat_distribution_map:
    :return:
    '''
    num_planes = np.array([int(k) for k in cat_distribution_map.keys()])
    num_planes_sort_indices = np.argsort(num_planes)
    num_planes = num_planes[num_planes_sort_indices]
    num_images = np.array([v for v in cat_distribution_map.values()])
    num_images = num_images[num_planes_sort_indices]

    num_planes_syn = np.array([int(k) for k in cat_distribution_map_syn.keys()])
    num_planes_syn_sort_indices = np.argsort(num_planes_syn)
    num_planes_syn = num_planes_syn[num_planes_syn_sort_indices]
    num_images_syn = np.array([v for v in cat_distribution_map_syn.values()])
    num_images_syn = num_images_syn[num_planes_syn_sort_indices]
    syn_args = get_syn_args()
    save_dir = syn_args.syn_plane_txt_dir + 'data_distribution_fig/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    title = "Comparison Airplane Distribution (chips)"

    labels = num_planes.tolist()
    x = num_planes
    ylist = num_images

    labels_syn = num_planes_syn.tolist()
    x_syn = num_planes_syn
    ylist_syn = num_images_syn

    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    fig, ax = plt.subplots(1, 1)
    width = 0.35

    rects_syn = ax.bar(np.array(x_syn) + width / 2, ylist_syn, width, label='Synthetic')  # , label=labels
    autolabel(ax, rects_syn, x_syn, labels_syn, ylist_syn, rotation=0)

    rects = ax.bar(np.array(x) - width / 2, ylist, width, label='xView')  # , label=labels
    autolabel(ax, rects, x, labels, ylist, rotation=0)
    ax.legend()

    xlabel = 'Number of Airplanes'
    ylabel = "Number of Images"
    plt.title(title, literal_eval(syn_args.font2))
    plt.ylabel(ylabel, literal_eval(syn_args.font2))
    plt.xlabel(xlabel, literal_eval(syn_args.font2))
    plt.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.grid()
    plt.savefig(os.path.join(save_dir, png_name))
    plt.show()


def rename_images_groundtruth(cities, streets, tile_size=608, resolution=0.3):
    rgb_suffix = '_RGB'
    gt_suffix = '_GT'
    step_size = tile_size * resolution
    for i in range(len(cities)):
        rgb_path = os.path.join(syn_args.syn_plane_img_anno_dir,
                                '{}_{}_{}_images_step{}'.format(syn_args.syn_display_type, cities[i], streets[i],
                                                                step_size))
        gt_path = os.path.join(syn_args.syn_plane_img_anno_dir,
                               '{}_{}_{}_annos_step{}'.format(syn_args.syn_display_type, cities[i], streets[i],
                                                              step_size))
        rgb_files = np.sort(glob.glob(os.path.join(rgb_path, '*{}'.format(IMG_FORMAT))))
        rgb_names = [os.path.basename(f) for f in rgb_files]

        gt_files = np.sort(glob.glob(os.path.join(gt_path, '*{}'.format(IMG_FORMAT))))
        gt_names = [os.path.basename(f) for f in gt_files]

        for ix, f in enumerate(rgb_files):
            new_name = rgb_names[ix].replace(rgb_suffix, '')
            os.rename(f, os.path.join(rgb_path, new_name))

        for ix, f in enumerate(gt_files):
            new_name = gt_names[ix].replace(gt_suffix, '')
            os.rename(f, os.path.join(gt_path, new_name))


def rename_folder(dir):
    folder_names = os.listdir(dir)
    for f in folder_names:
        src_name = syn_args.syn_display_type[:-1] + '_'
        dst_name = syn_args.syn_display_type + '_'
        os.rename(os.path.join(dir, f), os.path.join(dir, f.replace(src_name, dst_name)))


def rename_color_texture_groundtruth(tile_size=608, resolution=0.3):
    cl_prefix = 'color_'
    tx_prefix = 'texture_'
    step_size = tile_size * resolution
    gt_path = os.path.join(syn_args.syn_plane_img_anno_dir,
                           'syn_color_francisco_200_annos_step{}'.format(step_size))

    gt_files = np.sort(glob.glob(os.path.join(gt_path, '*{}'.format(IMG_FORMAT))))
    gt_names = [os.path.basename(f) for f in gt_files]

    for ix, f in enumerate(gt_files):
        new_name = gt_names[ix].replace(tx_prefix, cl_prefix)
        os.rename(f, os.path.join(gt_path, new_name))


def merge_clean_origin_syn_image_files(file_path, cities, streets, tile_size=608, resolution=0.3, white_thresh=0.5):
    '''
    merge all the origin synthetic data into one folder
    then remove rgb images those contain more than white_thresh*100% white pixels
    and remove gt images that are all white pixels
    :param file_path:
    :param cities:
    :param streets:
    :param tile_size:
    :param resolution:
    :param white_thresh:
    :return:
    '''
    step = syn_args.tile_size * syn_args.resolution
    image_folder_name = '{}_{}_{}_images_step{}'
    label_folder_name = '{}_{}_{}_annos_step{}'

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

        lbl_path = os.path.join(file_path,
                                label_folder_name.format(syn_args.syn_display_type, cities[i], streets[i], step))
        lbl_files = np.sort(glob.glob(os.path.join(lbl_path, '*{}'.format(IMG_FORMAT))))

        for lbl in lbl_files:
            shutil.copy(lbl, des_lbl_path)

    all_images = np.sort(glob.glob(os.path.join(des_img_path, '*{}'.format(IMG_FORMAT))))
    for ix, f in enumerate(all_images):
        img = io.imread(f)
        img = color.rgb2gray(img)
        white_num = np.sum(img == 1)
        white_ratio = white_num / img.shape[0] / img.shape[1]

        fname = f.split('/')[-1]
        gt = io.imread(os.path.join(des_lbl_path, fname))
        gt = color.rgb2gray(gt)
        gt_white_num = np.sum(gt == 1)
        gt_white_ratio = gt_white_num / gt.shape[0] / gt.shape[1]
        if white_ratio > white_thresh or gt_white_ratio == 1:
            os.remove(f)
            os.remove(os.path.join(des_lbl_path, fname))


def group_object_annotation_and_draw_bbox(pxwhr='px6whr4_ng0', px_thres=20, whr_thres=4):
    '''
    group annotation files, generate bbox for each object,

    and draw bbox for each ground truth files
    '''
    step = syn_args.tile_size * syn_args.resolution
    folder_name = '{}_all_annos_step{}'.format(syn_args.syn_display_type, step)
    lbl_path = os.path.join(syn_args.syn_plane_img_anno_dir, folder_name)
    txt_folder_name = 'minr{}_linkr{}_{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, pxwhr,
                                                                      syn_args.syn_display_type, step)
    save_txt_path = os.path.join(syn_args.syn_plane_txt_dir, txt_folder_name)
    if not os.path.exists(save_txt_path):
        os.makedirs(save_txt_path)
    else:
        shutil.rmtree(save_txt_path)
        os.makedirs(save_txt_path)

    gbc.get_object_bbox_after_group(lbl_path, save_txt_path, class_label=0, min_region=syn_args.min_region,
                                    link_r=syn_args.link_r, px_thresh=px_thres, whr_thres=whr_thres)
    gt_files = np.sort(glob.glob(os.path.join(lbl_path, '*{}'.format(IMG_FORMAT))))
    bbox_folder_name = 'minr{}_linkr{}_{}_{}_all_annos_with_bbox_step{}'.format(syn_args.min_region, syn_args.link_r, pxwhr,
                                                                             syn_args.syn_display_type, step)
    save_bbx_path = os.path.join(syn_args.syn_plane_gt_bbox_dir, bbox_folder_name)
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


def draw_bbx_on_rgb_images(pxwhr='px6whr4_ng0'):
    step = syn_args.tile_size * syn_args.resolution
    img_folder_name = '{}_all_images_step{}'.format(syn_args.syn_display_type, step)
    img_path = os.path.join(syn_args.syn_plane_img_anno_dir, img_folder_name)
    img_files = np.sort(glob.glob(os.path.join(img_path, '*{}'.format(IMG_FORMAT))))
    img_names = [os.path.basename(f) for f in img_files]

    txt_folder_name = 'minr{}_linkr{}_{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, pxwhr,
                                                                      syn_args.syn_display_type, step)
    txt_path = os.path.join(syn_args.syn_plane_txt_dir, txt_folder_name)

    bbox_folder_name = 'minr{}_linkr{}_{}_{}_all_images_with_bbox_step{}'.format(syn_args.min_region, syn_args.link_r, pxwhr,
                                                                              syn_args.syn_display_type, step)
    save_bbx_path = os.path.join(syn_args.syn_plane_gt_bbox_dir, bbox_folder_name)
    if not os.path.exists(save_bbx_path):
        os.makedirs(save_bbx_path)
    else:
        shutil.rmtree(save_bbx_path)
        os.makedirs(save_bbx_path)

    for ix, f in enumerate(img_files):
        txt_file = os.path.join(txt_path, img_names[ix].replace(IMG_FORMAT, TXT_FORMAT))
        gbc.plot_img_with_bbx(f, txt_file, save_bbx_path, label_index=False)


def draw_bbx_on_rgb_images_with_indices(syn=True, dt='syn_texture', pxwhr='px6whr4_ng0', px_thres=None, whr_thres=None):
    if syn:
        step = syn_args.tile_size * syn_args.resolution
        img_folder_name = '{}_all_images_step{}'.format(dt, step)
        img_path = os.path.join(syn_args.syn_plane_img_anno_dir.replace('/' + syn_args.syn_display_type, '/' + dt), img_folder_name)
        files = np.sort(glob.glob(os.path.join(img_path, '*{}'.format(IMG_FORMAT))))
        file_names = [os.path.basename(f).replace(IMG_FORMAT, TXT_FORMAT) for f in files]

        txt_folder_name = 'minr{}_linkr{}_{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, pxwhr,
                                                                          dt, step)
        txt_path = os.path.join(syn_args.syn_plane_txt_dir.replace('/' + syn_args.syn_display_type, '/' + dt), txt_folder_name)

        bbox_folder_name = 'minr{}_linkr{}_{}_{}_all_images_with_bbox_step{}_with_indices'.format(syn_args.min_region, syn_args.link_r, pxwhr,
                                                                                                dt, step)
        save_bbx_path = os.path.join(syn_args.syn_plane_gt_bbox_dir.replace('/' + syn_args.syn_display_type, '/' + dt), bbox_folder_name)
        if not os.path.exists(save_bbx_path):
            os.makedirs(save_bbx_path)
        else:
            shutil.rmtree(save_bbx_path)
            os.makedirs(save_bbx_path)

        for ix, f in enumerate(files):
            txt_file = os.path.join(txt_path, file_names[ix].replace(IMG_FORMAT, TXT_FORMAT))
            gbc.plot_img_with_bbx(f, txt_file, save_bbx_path, label_index=True)
    else:
        args = pwv.get_args(px_thres, whr_thres)
        files = np.sort(glob.glob(os.path.join(args.annos_save_dir, '*{}'.format(TXT_FORMAT))))
        file_names = [os.path.basename(f).replace(TXT_FORMAT, IMG_FORMAT0) for f in files]

        bbox_folder_name = '{}_images_with_bbox_with_indices'.format(pxwhr)
        save_bbx_path = os.path.join(args.cat_sample_dir, 'image_with_bbox_indices', bbox_folder_name)

        if not os.path.exists(save_bbx_path):
            os.makedirs(save_bbx_path)
        else:
            shutil.rmtree(save_bbx_path)
            os.makedirs(save_bbx_path)

        for ix, f in enumerate(files):
            img_file = os.path.join(args.images_save_dir, file_names[ix].replace(TXT_FORMAT, IMG_FORMAT0))
            gbc.plot_img_with_bbx(img_file, f, save_bbx_path, label_index=True)

def draw_bbx_on_rgb_images_with_model_id(syn=True, dt='syn_texture', pxwhr='px23whr3', px_thres=None, whr_thres=None):
    if syn:
        step = syn_args.tile_size * syn_args.resolution
        img_folder_name = '{}_all_images_step{}'.format(dt, step)
        img_path = os.path.join(syn_args.syn_plane_img_anno_dir.replace('/' + syn_args.syn_display_type, '/' + dt), img_folder_name)
        files = np.sort(glob.glob(os.path.join(img_path, '*{}'.format(IMG_FORMAT))))
        file_names = [os.path.basename(f).replace(IMG_FORMAT, TXT_FORMAT) for f in files]

        txt_folder_name = 'minr{}_linkr{}_{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, pxwhr,
                                                                          dt, step)
        txt_path = os.path.join(syn_args.syn_plane_txt_dir.replace('/' + syn_args.syn_display_type, '/' + dt), txt_folder_name)

        bbox_folder_name = 'minr{}_linkr{}_{}_{}_all_images_with_bbox_step{}_with_model_id'.format(syn_args.min_region, syn_args.link_r, pxwhr,
                                                                                                dt, step)
        save_bbx_path = os.path.join(syn_args.syn_plane_gt_bbox_dir.replace('/' + syn_args.syn_display_type, '/' + dt), bbox_folder_name)
        if not os.path.exists(save_bbx_path):
            os.makedirs(save_bbx_path)
        else:
            shutil.rmtree(save_bbx_path)
            os.makedirs(save_bbx_path)

        for ix, f in enumerate(files):
            txt_file = os.path.join(txt_path, file_names[ix].replace(IMG_FORMAT, TXT_FORMAT))
            gbc.plot_img_with_bbx(f, txt_file, save_bbx_path, Model_id=True)
    else:
        args = pwv.get_args(px_thres, whr_thres)
        files = np.sort(glob.glob(os.path.join(args.annos_save_dir[:-1] + '_all_model', '*{}'.format(TXT_FORMAT))))
        file_names = [os.path.basename(f).replace(TXT_FORMAT, IMG_FORMAT0) for f in files]

        bbox_folder_name = '{}_images_with_bbox_with_model_id'.format(pxwhr)
        save_bbx_path = os.path.join(args.cat_sample_dir, 'image_with_bbox_model_ids', bbox_folder_name)

        if not os.path.exists(save_bbx_path):
            os.makedirs(save_bbx_path)
        else:
            shutil.rmtree(save_bbx_path)
            os.makedirs(save_bbx_path)

        for ix, f in enumerate(files):
            img_file = os.path.join(args.images_save_dir, file_names[ix].replace(TXT_FORMAT, IMG_FORMAT0))
            gbc.plot_img_with_bbx(img_file, f, save_bbx_path, Model_id=True)


def clean_annos_txt_copy_imgs_files():
    step = syn_args.tile_size * syn_args.resolution
    src_dir = os.path.join(syn_args.syn_plane_txt_dir,
                           'minr{}_linkr{}_{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, pxwhr,
                                                                           syn_args.syn_display_type, step))
    src_files = glob.glob(os.path.join(src_dir, '*{}'.format(TXT_FORMAT)))

    img_folder_name = '{}_all_images_step{}'.format(syn_args.syn_display_type, step)
    lbl_folder_name = '{}_all_annos_step{}'.format(syn_args.syn_display_type, step)
    img_path = os.path.join(syn_args.syn_plane_img_anno_dir, img_folder_name)
    lbl_path = os.path.join(syn_args.syn_plane_img_anno_dir, lbl_folder_name)
    if not os.path.exists(syn_args.syn_annos_save_dir):
        os.mkdir(syn_args.syn_annos_save_dir)
    else:
        shutil.rmtree(syn_args.syn_annos_save_dir)
        os.mkdir(syn_args.syn_annos_save_dir)

    for sf in src_files:
        if is_non_zero_file(sf):
            shutil.copy(sf, syn_args.syn_annos_save_dir)
        else:
            os.remove(os.path.join(lbl_path, sf.split('/')[-1].replace(TXT_FORMAT,
                                                                       IMG_FORMAT)))
            os.remove(os.path.join(img_path, sf.split('/')[-1].replace(TXT_FORMAT,
                                                                       IMG_FORMAT)))
            os.remove(sf)
    ####------- copy to images data folder
    all_images = np.sort(glob.glob(os.path.join(img_path, '*{}'.format(IMG_FORMAT))))
    if not os.path.exists(syn_args.syn_annos_save_dir):
        os.mkdir(syn_args.syn_images_save_dir)
    else:
        shutil.rmtree(syn_args.syn_images_save_dir)
        os.mkdir(syn_args.syn_images_save_dir)

    for f in all_images:
        shutil.copy(f, syn_args.syn_images_save_dir)


def generate_syn_texture_syn_colr_label_with_model_based_on_syn_color():
    color_path = '/media/lab/Yang/data/xView_YOLO/labels/608/syn_color_1_cls_xcycwh_model/'
    color_files = np.sort(glob.glob(os.path.join(color_path, '*.txt')))
    texture_path = '/media/lab/Yang/data/xView_YOLO/labels/608/syn_texture_1_cls_xcycwh_model/'
    if not os.path.exists(texture_path):
        os.mkdir(texture_path)
    mixed_path = '/media/lab/Yang/data/xView_YOLO/labels/608/syn_mixed_1_cls_xcycwh_model/'
    if not os.path.exists(mixed_path):
        os.mkdir(mixed_path)

    txt_names = [os.path.basename(f) for f in color_files]
    for cn in txt_names:
        shutil.copy(os.path.join(color_path, cn), os.path.join(texture_path, cn.replace('color_', 'texture_')))
        shutil.copy(os.path.join(color_path, cn), os.path.join(mixed_path, cn.replace('color_', 'mixed_')))


def plot_img_with_bbx_model_id(img_file, lbl_file, save_path):
    if not is_non_zero_file(lbl_file):
        # print(is_non_zero_file(lbl_file))
        return
    img = cv2.imread(img_file) # h, w, c
    h, w = img.shape[:2]
    colomns=np.arange(0, 6) # ********
    df_lbl = pd.read_csv(lbl_file, header=None, sep=' ', names=colomns).to_numpy() # delimiter , error_bad_lines=False
    df_lbl[:, 1] = df_lbl[:, 1]*w
    df_lbl[:, 3] = df_lbl[:, 3]*w

    df_lbl[:, 2] = df_lbl[:, 2]*h
    df_lbl[:, 4] = df_lbl[:, 4]*h

    df_lbl[:, 1] -= df_lbl[:, 3]/2
    df_lbl[:, 2] -= df_lbl[:, 4]/2

    df_lbl[:, 3] += df_lbl[:, 1]
    df_lbl[:, 4] += df_lbl[:, 2]
    # print(df_lbl.shape[0])
    # df_lbl_uni = np.unique(df_lbl[:, 1:],axis=0)
    # print('after unique ', df_lbl_uni.shape[0])

    for ix in range(df_lbl.shape[0]):
        cat_id = int(df_lbl[ix, 0])
        gt_bbx = df_lbl[ix, 1:-1].astype(np.int64) # except the last colomn
        img = cv2.rectangle(img, (gt_bbx[0], gt_bbx[1]), (gt_bbx[2], gt_bbx[3]), (255, 0, 0), 2)
        pl = ''
        if not np.isnan(df_lbl[ix, -1]):
            pl = '{}_{}'.format(ix, df_lbl[ix, -1])
        else:
            pl = '{}'.format(ix)
        cv2.putText(img, text=pl, org=(gt_bbx[0] + 10, gt_bbx[1] + 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))
    cv2.imwrite(os.path.join(save_path, img_file.split('/')[-1]), img)


def check_img_with_bbox_with_indices_model_id():
    color_path = '/media/lab/Yang/data/xView_YOLO/labels/608/syn_color_1_cls_xcycwh_model/'
    texture_path = '/media/lab/Yang/data/xView_YOLO/labels/608/syn_texture_1_cls_xcycwh_model/'
    mixed_path = '/media/lab/Yang/data/xView_YOLO/labels/608/syn_mixed_1_cls_xcycwh_model/'

    timg_path = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_texture/syn_texture_all_images_step182.4/'
    mimg_path = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_mixed/syn_mixed_all_images_step182.4/'
    time_files = np.sort(glob.glob(os.path.join(timg_path, '*' + IMG_FORMAT)))
    mimg_files = np.sort(glob.glob(os.path.join(mimg_path, '*' + IMG_FORMAT)))

    save_t_path = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_texture/syn_texture_all_images_step182.4_with_bbox_model/'
    save_m_path = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_mixed/syn_mixed_all_images_step182.4_with_bbox_model/'
    if not os.path.exists(save_t_path):
        os.mkdir(save_t_path).replace(IMG_FORMAT, TXT_FORMAT)
    if not os.path.exists(save_m_path):
        os.mkdir(save_m_path)

    timg_names = [os.path.basename(f) for f in time_files]
    mimg_names = [os.path.basename(f) for f in mimg_files]

    for tn in timg_names:
        plot_img_with_bbx_model_id(os.path.join(timg_path, tn), os.path.join(texture_path, tn.replace(IMG_FORMAT, TXT_FORMAT)), save_t_path,)

    for mn in mimg_names:
        plot_img_with_bbx_model_id(os.path.join(mimg_path, mn), os.path.join(mixed_path, mn.replace(IMG_FORMAT, TXT_FORMAT)), save_m_path)


def statistic_model_number(comments='px23whr3_seed17', type='all'):
    json_name = '{}_model_num_maps.json'.format(type)
    png_name = '{}_number_3d-model.jpg'.format(type)
    Num = {}

    if type == 'train':
        data_path = pd.read_csv(os.path.join(args.data_save_dir, comments, 'xview{}_lbl_{}_with_model.txt'.format(type, comments)), header=None)
        lbl_files = [f for f in data_path.loc[:, 0]]
        title = "Model Numbers of Training Dataset"
    elif type == 'val':
        data_path = pd.read_csv(os.path.join(args.data_save_dir, comments, 'xview{}_lbl_{}_with_model.txt'.format(type, comments)), header=None)
        lbl_files = [f for f in data_path.loc[:, 0]]
        title = "Model Numbers of Validation Dataset"
    elif type == 'syn':
        syn_file = '/media/lab/Yang/code/yolov3/data_xview/syn_xview_bkg_px23whr3_xbw_xrxc_spr_sml_gauss_models_color_1_cls/syn_xview_bkg_px23whr3_xbw_xrxc_spr_sml_gauss_models_color_seed17/syn_xview_bkg_px23whr3_xbw_xrxc_spr_sml_gauss_models_color_all_lbl_seed17.txt'
        data_path = pd.read_csv(syn_file, header=None)
        lbl_files = [f for f in data_path.loc[:, 0]]
        Num['all'] = 0
        title = "Model Numbers of Synthetic Dataset"
    else:
        lbl_files = glob.glob(os.path.join(args.annos_save_dir[:-1] + '_all_model', '*.txt'))
        title = 'Model Numbers of xView '.format(type)

    for f in lbl_files:
        if not is_non_zero_file(f):
            continue
        # print(f)
        df_lbl = pd.read_csv(f, header=None, sep=' ')
        if type=='syn':
            Num['all'] += df_lbl.shape[0]
        else:
            for m in df_lbl.loc[:, 5]:
                if m not in Num.keys():
                    Num[m] = 1
                    # print(m)
                    # print(f)
                else:
                    Num[m] += 1
    json_dir = os.path.join(args.txt_save_dir, 'model_number', comments)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    json.dump(Num, open(os.path.join(json_dir, json_name),
                        'w'), ensure_ascii=False, indent=2, cls=MyEncoder)
    cnt_models = [s for s in Num.values()]
    sum_models = sum(cnt_models)
    print('sum models :', sum_models)
    ratios = np.array([c/sum_models for c in cnt_models])
    keys = np.array([s for s in Num.keys()])
    kids = np.argsort(keys)
    print('keys: ', keys[kids])
    print('values: ', ratios[kids])
    save_dir = os.path.join(args.txt_save_dir, 'model_number', comments)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    fig, ax = plt.subplots(1, 1)
    width = 0.35
    x = [k for k in Num.keys()]
    xticklabels = x.copy()
    xticklabels[xticklabels.index(max(xticklabels))] = 'unlabeled'
    ylist = [v for v in Num.values()]
    rects = ax.bar(np.array(x), ylist, width)  # , label=labels
    autolabel(ax, rects, x, xticklabels, ylist, rotation=0)
    ylabel = 'Number of Bbox'
    xlabel = "Model ID"
    plt.title(title, literal_eval(syn_args.font2))
    plt.ylabel(ylabel, literal_eval(syn_args.font2))
    plt.xlabel(xlabel, literal_eval(syn_args.font2))
    plt.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.grid()
    plt.savefig(os.path.join(save_dir, png_name))
    plt.show()





def plot_aircraft_body_wing_hist_pdf(bw_ratio=False):
    xls_file = '/media/lab/Yang/data/xView/aircraft_body_wing/Syn_Body_Wing.xlsx'
    save_path = '/media/lab/Yang/data/xView/aircraft_body_wing/new/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    df_xls = pd.read_excel(xls_file, sheet_name='all_models_more_images')
    body_col = 'm{}_body'
    wing_col = 'm{}_wing'
    sample_size = 500
    num_bins = 45
    total_models = 7
    for i in range(total_models):
        bcl = body_col.format(i)
        body = df_xls[bcl].dropna()
        bmx = body.max()
        bmn = body.min()
        bmu = np.mean(body)
        bsgm = np.std(body)
        bxlabel = 'Body Length'
        txt_bname = 'sample_body_model{}.txt'.format(i)

        wcl = wing_col.format(i)
        wing = df_xls[wcl].dropna()
        bw = body/wing
        wmx = bw.max()
        wmn = bw.min()
        wmu = np.mean(bw)
        wsgm = np.std(bw)
        wxlabel = 'Body/Wing Ratio'
        txt_wname = 'sample_bw_ratio_model{}.txt'.format(i)

        fig, axs = plt.subplots(1, 2, figsize=(12, 8))
        bn, bbins, bpatches = axs[0].hist(body, num_bins, density=1, alpha=0.5, label='actual hist') # facecolor='blue'
        wn, wbins, wpatches = axs[1].hist(bw, num_bins, density=1, alpha=0.5, label='actual hist') # facecolor='blue'
        #直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象
        by = scs.norm.pdf(bbins, bmu, bsgm)#拟合一条最佳正态分布曲线y
        axs[0].plot(bbins, by, color='c', label = 'actual data') #绘制y的曲线
        bX = scs.norm.rvs(loc=bmu, scale=bsgm, size=sample_size).T
        bX = bX[bX >= bmn]
        bX = bX[bX <= bmx]
        print('bX model {}'.format(i), bX.shape)
        byf = scs.norm.pdf(bX, bmu, bsgm)
        txt_bx = open(os.path.join(save_path, txt_bname), 'w')
        for xi in bX:
            txt_bx.write('{:.2f};'.format(xi))
        txt_bx.close()
        axs[0].scatter(bX, byf, color="yellow", label = "gaussian samples")
        axs[0].hist(bX, bins=num_bins, facecolor='darkorange', density=1, label = 'gaussian sample hist', alpha=0.5)
        axs[0].legend()
        axs[0].grid()
        axs[0].set_xlabel(bxlabel) #绘制x轴
        axs[0].set_ylabel('Probability') #绘制y轴
        axs[0].set_title(r'Model {} : $\mu={:.2f}$,$\sigma={:.2f}$'.format(i, bmu, bsgm))

        wy = scs.norm.pdf(wbins, wmu, wsgm)#拟合一条最佳正态分布曲线y
        axs[1].plot(wbins, wy, color='c', label = 'actual data') #绘制y的曲线

        wX = scs.norm.rvs(loc=wmu, scale=wsgm, size=sample_size).T
        wX = wX[wX >= wmn]
        wX = wX[wX <= wmx]
        print('wX model {}'.format(i), wX.shape)
        wyf = scs.norm.pdf(wX, wmu, wsgm)
        txt_wx = open(os.path.join(save_path, txt_wname), 'w')
        for xi in wX:
            txt_wx.write('{:.2f};'.format(xi))
        txt_wx.close()
        # np.savetxt(os.path.join(save_path, txt_name), X)
        axs[1].scatter(wX, wyf, color="yellow", label = "gaussian samples")
        axs[1].hist(wX, bins=num_bins, facecolor='darkorange', density=1, label = 'gaussian sample hist', alpha=0.5)
        axs[1].legend()
        axs[1].grid()
        axs[1].set_xlabel(wxlabel) #绘制x轴
        axs[1].set_ylabel('Probability') #绘制y轴
        axs[1].set_title(r'Model {} : $\mu={:.2f}$,$\sigma={:.2f}$'.format(i, wmu, wsgm))#中文标题 u'xxx'
        png_name = 'hist_body_bwratio_model{}.png'.format(i)
        plt.savefig(os.path.join(save_path, png_name))
        # plt.show()
        plt.close()


def get_syn_args(cmt=''):
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_filepath", type=str, help="Filepath to GEOJSON coordinate file",
                        default='/media/lab/Yang/data/xView/xView_train.geojson')
    #fixme
    if cmt: # certain_models, scale_models
        parser.add_argument("--syn_plane_img_anno_dir", type=str,
                        help="Path to folder containing synthetic images and annos ",
                        default='/media/lab/Yang/data/synthetic_data/syn_xview_bkg_{}'.format(cmt)+'_{}/')
        parser.add_argument("--syn_plane_txt_dir", type=str, default='/media/lab/Yang/data/synthetic_data/syn_xview_bkg_{}_txt_xcycwh/'.format(cmt),
                            help="syn xview txt")
        parser.add_argument("--syn_plane_gt_bbox_dir", type=str, default='/media/lab/Yang/data/synthetic_data/syn_xview_bkg_{}_gt_bbox/'.format(cmt),
                            help="syn xview txt related files")
    else: # cmt == ''
        parser.add_argument("--syn_plane_img_anno_dir", type=str,
                            help="Path to folder containing synthetic images and annos ",
                            default='/media/lab/Yang/data/synthetic_data/syn_xview_background_{}/')
        parser.add_argument("--syn_plane_txt_dir", type=str, default='/media/lab/Yang/data/synthetic_data/syn_xview_background_txt_xcycwh',
                            help="syn xview txt")
        parser.add_argument("--syn_plane_gt_bbox_dir", type=str, default='/media/lab/Yang/data/synthetic_data/syn_xview_background_gt_bbox',
                            help="syn xview txt related files")

    parser.add_argument("--syn_images_save_dir", type=str, help="rgb images of synthetic airplanes",
                        default='/media/lab/Yang/data/xView_YOLO/images/{}_{}/')
    parser.add_argument("--syn_annos_save_dir", type=str, help="gt of synthetic airplanes",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_{}_cls_xcycwh/')
    parser.add_argument("--syn_txt_save_dir", type=str, help="gt related files of synthetic airplanes",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_{}_cls/')

    parser.add_argument("--data_xview_dir", type=str, help="to save data files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_cls/')

    parser.add_argument("--syn_data_list_dir", type=str, help="to syn data list files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_{}_cls/')

    parser.add_argument("--results_dir", type=str, help="to save category files",
                        default='/media/lab/Yang/code/yolov3/result_output/{}_cls/{}_{}/')

    parser.add_argument("--cat_sample_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/')

    parser.add_argument("--val_percent", type=float, default=0.20,
                        help="Percent to split into validation (ie .25 = val set is 25% total)")
    parser.add_argument("--font3", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 10}")
    parser.add_argument("-ft2", "--font2", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 13}")

    parser.add_argument("--class_num", type=int, default=1, help="Number of Total Categories")  # 60  6
    parser.add_argument("--seed", type=int, default=17, help="random seed") #fixme  1024 17
    parser.add_argument("--tile_size", type=int, default=608, help="image size")  # 300 416

    # #####*********************change
    parser.add_argument("--syn_display_type", type=str, default='',
                        help="syn_texture, syn_color, syn_mixed,  texture syn (match 0), syn_background")  # syn_color0, syn_texture0,
    # ######*********************change
    parser.add_argument("--syn_ratio", type=float, default=None,
                        help="ratio of synthetic data: 0.25, 0.5, 0.75, 1.0  0")  # ######*********************change

    parser.add_argument("--min_region", type=int, default=300, help="100 the smallest #pixels (area) to form an object")
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
    syn_args.syn_images_save_dir = syn_args.syn_images_save_dir.format(syn_args.tile_size, syn_args.syn_display_type)
    syn_args.syn_annos_save_dir = syn_args.syn_annos_save_dir.format(syn_args.tile_size, syn_args.syn_display_type,
                                                                     syn_args.class_num)
    syn_args.syn_txt_save_dir = syn_args.syn_txt_save_dir.format(syn_args.tile_size, syn_args.syn_display_type,
                                                                 syn_args.class_num)
    syn_args.results_dir = syn_args.results_dir.format(syn_args.class_num, syn_args.syn_display_type,
                                                       syn_args.syn_ratio)

    if syn_args.syn_ratio:
        syn_args.syn_data_list_dir = syn_args.syn_data_list_dir.format(syn_args.syn_display_type, syn_args.class_num)
        if not os.path.exists(syn_args.syn_data_list_dir):
            os.mkdir(syn_args.syn_data_list_dir)

    syn_args.syn_plane_img_anno_dir = syn_args.syn_plane_img_anno_dir.format(syn_args.syn_display_type)
    syn_args.syn_plane_txt_dir = syn_args.syn_plane_txt_dir.format(syn_args.syn_display_type)
    syn_args.syn_plane_gt_bbox_dir = syn_args.syn_plane_gt_bbox_dir.format(syn_args.syn_display_type)

    if not os.path.exists(syn_args.syn_plane_img_anno_dir):
        os.mkdir(syn_args.syn_plane_img_anno_dir)

    if not os.path.exists(syn_args.syn_plane_txt_dir):
        os.mkdir(syn_args.syn_plane_txt_dir)

    if not os.path.exists(syn_args.syn_plane_gt_bbox_dir):
        os.mkdir(syn_args.syn_plane_gt_bbox_dir)

    return syn_args


if __name__ == "__main__":
    syn_args = get_syn_args()

    '''
    count aircraft distribution: count the number of images that contain certain number of planes
    '''
    # args = pwv.get_args()
    # catid = 0
    # # # whr_thres = 4  # 3
    # # # px_thresh = 6 # 4
    # # # comments = '_px{}whr{}_ng0'.format(px_thresh, whr_thres)
    # whr_thres = 4 # 4  # 3
    # px_thresh = 20 # 6 # 4
    # comments = '_px{}whr{}'.format(px_thresh, whr_thres)
    #
    # txt_path = args.annos_save_dir[:-1] + comments + '/'
    # # txt_path = args.syn_plane_txt_dir
    # json_number_file = os.path.join(args.txt_save_dir,
    #                                 'xView_number_of_cat_{}_to_imagenumber_map_inputsize{}{}.json'.format(catid, args.input_size, comments))
    # json_txt_file = os.path.join(args.txt_save_dir,
    #                              'xView_number_of_cat_{}_to_imagetxt_map_inputsize{}{}.json'.format(catid, args.input_size, comments))
    # analyze_category_distribution(txt_path, json_number_file, json_txt_file, px_thresh, whr_thres)

    '''
    plot count aircraft distribution
    '''
    # args = pwv.get_args()
    # catid = 0
    # # whr_thres = 4  # 3
    # # px_thresh = 6 # 4
    # # comments = '_px{}whr{}_ng0'.format(px_thresh, whr_thres)
    # whr_thres = 4 # 4  # 3
    # px_thresh = 20 # 6 # 4
    # comments = '_px{}whr{}'.format(px_thresh, whr_thres)
    # png_name = 'xView_cat_{}_nubmers_imagesnumber_dis{}.png'.format(catid, comments)
    # cat_distribution_map = json.load(open(os.path.join(args.txt_save_dir,
    #                                                    'xView_number_of_cat_{}_to_imagenumber_map_inputsize{}{}.json'.format(catid,
    #                                                        args.input_size, comments))))
    # draw_bar_of_image_numbers_for_certain_number_of_planes(png_name, cat_distribution_map, 'xView')

    '''
    rename all the images and annos with suffix '_RGB' '_GT' if needed
    '''
    # rename_images_groundtruth(syn_args.cities, syn_args.streets, syn_args.tile_size, syn_args.resolution)

    '''
    rename francisco syn_color ground truth
    '''
    # rename_color_texture_groundtruth(syn_args.tile_size, syn_args.resolution)

    '''
    rename folder
    syn_texture_ --> syn_texture0_
    '''
    # rename_folder(syn_args.syn_plane_img_anno_dir)
    # rename_folder(syn_args.syn_plane_txt_dir)
    # rename_folder(syn_args.syn_plane_gt_bbox_dir)

    '''
    remove labels that contain more than 50% white pixels (next time for 20%)
    clean up useless images
    '''
    # white_thresh = 0.5
    # merge_clean_origin_syn_image_files(syn_args.syn_plane_img_anno_dir, syn_args.cities, syn_args.streets,
    #                                    syn_args.tile_size, syn_args.resolution, white_thresh)

    '''
    group annotation files, generate bbox for each object, 
    and draw bbox for each ground truth files
    '''
    # pxwhr = '_px6whr4'
    # px_thres=6
    # whr_thres=4
    # pxwhr = 'px20whr4'
    # px_thres=20
    # whr_thres=4
    # group_object_annotation_and_draw_bbox(pxwhr, px_thres, whr_thres)

    '''
    remove non files (no labels) ***************
    copy images to images data folder
    '''
    # clean_annos_txt_copy_imgs_files()

    '''
    check bbox for each image files
    draw image with gt bbox 
    '''
    # pxwhr='px6whr4_ng0'
    # pxwhr='px20whr4'
    # pxwhr='px23whr4'
    # draw_bbx_on_rgb_images(pxwhr)

    '''
    draw rgb with gt bbox and gt indices
    '''
    # syn = False
    # pxwhr='px6whr4_ng0'
    # pxwhr='px20whr4'
    # draw_bbx_on_rgb_images_with_indices(syn, pxwhr)

    # pxwhr='px6whr4_ng0'
    # pxwhr='px20whr4'
    # syn = True
    # display_type = ['syn_texture', 'syn_color', 'syn_mixed']
    # for dt in display_type:
    #     draw_bbx_on_rgb_images_with_indices(syn, dt, pxwhr)

    '''
    analyze synthetic data distribution
    '''
    # catid = 0
    # # whr_thres = 4  # 3
    # # px_thresh = 6  # 4
    # whr_thres = 4 # 4  # 3
    # px_thresh = 20 # 6 # 4
    # comments = '_px{}whr{}'.format(px_thresh, whr_thres)
    # # txt_path = syn_args.syn_annos_save_dir
    # step = syn_args.tile_size * syn_args.resolution
    # json_number_file = os.path.join(syn_args.syn_plane_txt_dir,
    #                                 '{}_number_of_cat_{}_to_imagenumber_map_inputsize{}{}.json'.format(
    #                                     syn_args.syn_display_type, catid, syn_args.tile_size, comments))
    # json_name_file = os.path.join(syn_args.syn_plane_txt_dir,
    #                               '{}_number_of_cat_{}_to_imagetxt_map_inputsize{}{}.json'.format(
    #                                   syn_args.syn_display_type,
    #                                   catid, syn_args.tile_size, comments))
    # txt_path = os.path.join(syn_args.syn_plane_txt_dir,
    #                         'minr{}_linkr{}{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, comments,
    #                                                                         syn_args.syn_display_type, step))
    #
    # analyze_category_distribution(txt_path, json_number_file, json_name_file, px_thresh, whr_thres)

    '''
    plot synthetic data aircraft distribution
    '''
    # catid = 0
    # whr_thres = 4 # 4  # 3
    # px_thresh = 20 # 6 # 4
    # comments = '_px{}whr{}'.format(px_thresh, whr_thres)
    # png_name = '{}_cat_{}_nubmers_imagesnumber_dis.png'.format(syn_args.syn_display_type, catid)
    # cat_distribution_map = json.load(open(os.path.join(syn_args.syn_plane_txt_dir,
    #                                                    '{}_number_of_cat_{}_to_imagenumber_map_inputsize{}{}.json'.format(
    #                                                        syn_args.syn_display_type, catid, syn_args.tile_size, comments))))
    #
    # draw_bar_of_image_numbers_for_certain_number_of_planes(png_name, cat_distribution_map,
    #                                                        legend=syn_args.syn_display_type + comments, syn=True)

    ''''
    compare xview and synthetic aircraft distribution
    '''
    # args = pwv.get_args()
    # catid = 0
    # whr_thres = 4 # 4  # 3
    # px_thresh = 20 # 6 # 4
    # comments = '_px{}whr{}'.format(px_thresh, whr_thres)
    # png_name = 'xview_vs_{}{}_cat_{}_nubmers_imagesnumber_dis.png'.format(syn_args.syn_display_type, comments, catid)
    # cat_distribution_map = json.load(open(os.path.join(args.txt_save_dir,
    #                                                    'xView_number_of_cat_{}_to_imagenumber_map_inputsize{}{}.json'.format(
    #                                                        catid, syn_args.tile_size, comments))))
    # cat_distribution_map_syn = json.load(open(os.path.join(syn_args.syn_plane_txt_dir,
    #                                                        '{}_number_of_cat_{}_to_imagenumber_map_inputsize{}{}.json'.format(
    #                                                            syn_args.syn_display_type, catid, syn_args.tile_size, comments))))
    # compare_bar_of_image_numbers_for_certain_number_of_planes(png_name, cat_distribution_map, cat_distribution_map_syn)


    '''
    statistic aircraft number of each model 
    '''
    # whr_thres = 3 # 4  # 3
    # px_thresh = 23 # 6 # 4
    # seed = 17
    # args = pwv.get_args(px_thresh, whr_thres)
    # # type = 'all'
    # # type = 'train'
    # # type = 'val'
    # type = 'syn'
    # statistic_model_number(comments='px{}whr{}_seed{}'.format(px_thresh, whr_thres, seed), type=type)

    '''
    plot body wing hist and probility distribution function
    '''
    # bw_ratio = False
    # # bw_ratio = True
    # plot_aircraft_body_wing_hist_pdf(bw_ratio)












