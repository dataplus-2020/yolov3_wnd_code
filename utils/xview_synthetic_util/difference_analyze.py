import numpy as np
from matplotlib import pyplot as plt
from utils.xview_synthetic_util import preprocess_xview_syn_data_distribution as pps
import os
from glob import glob
from skimage.color import rgb2gray
from skimage import io
import shutil
import pandas as pd


def compare_overlap():
    # file1 = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_radial_200_images_test_ts608/airplanes_radial_200_59_RGB.jpg'
    # file2 = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_radial_200_images_test_ts608/airplanes_radial_200_60_RGB.jpg'
    file1 = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_radial_200_images_test_ts608/airplanes_radial_200_75_RGB.jpg'
    file2 = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_radial_200_images_test_ts608/airplanes_radial_200_76_RGB.jpg'

    img_file1 = io.imread(file1)
    img_file2 =io.imread(file2)

    res = 0.3
    size = img_file1.shape[0]
    step = size//2

    common1 = img_file1[:, step:]
    common2 = img_file2[:, :step]

    plt.figure(figsize=(15, 8))
    plt.subplot(131)
    plt.imshow(common1)

    plt.subplot(132)
    plt.imshow(common2)

    plt.subplot(133)
    plt.imshow(common2-common1)

    plt.tight_layout()
    plt.show()

    # plt.savefig('/media/lab/Yang/data/synthetic_data/Airplanes/59_60common_change.jpg')

    plt.savefig('/media/lab/Yang/data/synthetic_data/Airplanes/75_76common_change.jpg')


def compare_images_with_different_display_type(two=False):
    syn_args = pps.get_syn_args()
    cities = ['barcelona', 'berlin', 'francisco', 'hexagon', 'radial', 'siena', 'spiral']
    streets = [200, 200, 200, 200, 200, 250, 130]
    syn_plane_img_anno_dir = '/media/lab/Yang/data/synthetic_data/Airplanes/{}'
    city = cities[0]
    sts = streets[0]

    step = syn_args.tile_size * syn_args.resolution

    if two:
        '''
        first --> 0 
        '''
        IMG_FORMAT = '.jpg'
        save_dir = os.path.join('/media/lab/Yang/data/synthetic_data/Airplanes/', 'check_rgb_gt0', city)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        tx_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture0'), '{}_{}_{}_images_step{}'.format('syn_texture0', city, sts, step))
        tx_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture0'), '{}_{}_{}_annos_step{}'.format('syn_texture0',  city, sts, step))

        clr_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color0'), '{}_{}_{}_images_step{}'.format('syn_color0', city, sts, step))
        clr_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color0'), '{}_{}_{}_annos_step{}'.format('syn_color0',  city, sts, step))
        tx_image_files = np.sort(glob(os.path.join(tx_img_dir, '*' + IMG_FORMAT)))
        tx_img_names = [os.path.basename(f) for f in tx_image_files]
        clr_image_files = np.sort(glob(os.path.join(clr_img_dir, '*' + IMG_FORMAT)))
        clr_img_names = [os.path.basename(f) for f in clr_image_files]

        tx_lbl_files = np.sort(glob(os.path.join(tx_lbl_dir, '*' + IMG_FORMAT)))
        # tx_lbl_names = [os.path.basename(f) for f in tx_lbl_files]
        clr_lbl_files = np.sort(glob(os.path.join(clr_lbl_dir, '*' + IMG_FORMAT)))
        # clr_lbl_names = [os.path.basename(f) for f in clr_lbl_files]

        for ix in range(len(tx_image_files)):
            file1 = tx_image_files[ix]
            file2 = os.path.join(clr_img_dir, clr_img_names[ix])

            img_file1 = io.imread(file1)
            img_file2 = io.imread(file2)

            file21 = tx_lbl_files[ix]
            file22 = clr_lbl_files[ix]

            img_file21 = io.imread(file21)
            img_file22 = io.imread(file22)

            fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

            axs[0, 0].imshow(img_file1)
            axs[0, 0].set_title('texture')
            axs[0, 1].imshow(img_file2)
            axs[0, 1].set_title('color')
            axs[1, 0].imshow(img_file21)
            axs[1, 1].imshow(img_file22)
            axs[1, 1].imshow(img_file21- img_file22)
            axs[1, 0].set_xlabel(tx_img_names[ix])
            axs[1, 2].set_xlabel(clr_img_names[ix])
            axs[1, 2].set_xlabel('left - right')
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, tx_img_names[ix].replace('texture_', '')))
            # plt.show()
    else:
        '''
        second 
        '''
        IMG_FORMAT = '.png'
        save_dir = os.path.join('/media/lab/Yang/data/synthetic_data/Airplanes/', 'check_rgb_gt', city)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        tx_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture'), '{}_{}_{}_images_step{}'.format('syn_texture', city, sts, step))
        tx_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture'), '{}_{}_{}_annos_step{}'.format('syn_texture',  city, sts, step))

        clr_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color'), '{}_{}_{}_images_step{}'.format('syn_color', city, sts, step))
        clr_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color'), '{}_{}_{}_annos_step{}'.format('syn_color',  city, sts, step))

        mx_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_mixed'), '{}_{}_{}_images_step{}'.format('syn_mixed', city, sts, step))
        mx_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_mixed'), '{}_{}_{}_annos_step{}'.format('syn_mixed',  city, sts, step))

        tx_image_files = np.sort(glob(os.path.join(tx_img_dir, '*' + IMG_FORMAT)))
        tx_img_names = [os.path.basename(f) for f in tx_image_files]
        clr_image_files = np.sort(glob(os.path.join(clr_img_dir, '*' + IMG_FORMAT)))
        clr_img_names = [os.path.basename(f) for f in clr_image_files]
        mx_image_files = np.sort(glob(os.path.join(mx_img_dir, '*' + IMG_FORMAT)))
        mx_img_names = [os.path.basename(f) for f in mx_image_files]

        tx_lbl_files = np.sort(glob(os.path.join(tx_lbl_dir, '*' + IMG_FORMAT)))
        # tx_lbl_names = [os.path.basename(f) for f in tx_lbl_files]
        clr_lbl_files = np.sort(glob(os.path.join(clr_lbl_dir, '*' + IMG_FORMAT)))
        # clr_lbl_names = [os.path.basename(f) for f in clr_lbl_files]
        mx_lbl_files = np.sort(glob(os.path.join(mx_lbl_dir, '*' + IMG_FORMAT)))
        # mx_lbl_names = [os.path.basename(f) for f in mx_lbl_files]

        for ix in range(len(tx_image_files)):
            file1 = tx_image_files[ix]
            file2 = os.path.join(clr_img_dir, clr_img_names[ix])
            file3 = os.path.join(mx_img_dir, mx_img_names[ix])

            img_file1 = io.imread(file1)
            img_file2 = io.imread(file2)
            img_file3 = io.imread(file3)

            file21 = tx_lbl_files[ix]
            file22 = clr_lbl_files[ix]
            file23 = mx_lbl_files[ix]

            img_file21 = io.imread(file21)
            img_file22 = io.imread(file22)
            img_file23 = io.imread(file23)

            # if np.all(img_file3==img_file2) and np.all(img_file2==img_file1):
            #     continue

            # print(tx_image_files[ix])

            fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

            axs[0, 0].imshow(img_file1)
            axs[0, 0].set_title('texture')
            axs[0, 1].imshow(img_file2)
            axs[0, 1].set_title('color')
            axs[0, 2].imshow(img_file3)
            axs[0, 2].set_title('mixed')
            axs[1, 0].imshow(img_file21)
            axs[1, 1].imshow(img_file22)
            axs[1, 2].imshow(img_file23)
            axs[1, 0].set_xlabel(tx_img_names[ix])
            axs[1, 1].set_xlabel(clr_img_names[ix])
            axs[1, 2].set_xlabel(mx_img_names[ix])
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, tx_img_names[ix].replace('texture_', '')))
            # plt.show()


def compare_images_after_combine(two=False):

    '''
    all images annos
    '''
    syn_args = pps.get_syn_args()
    syn_plane_img_anno_dir = '/media/lab/Yang/data/synthetic_data/Airplanes/{}'

    step = syn_args.tile_size * syn_args.resolution

    if two:
        '''
        first --> 0 
        '''
        IMG_FORMAT = '.jpg'
        save_dir = os.path.join('/media/lab/Yang/data/synthetic_data/Airplanes/', 'check_rgb_gt0', 'all')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            shutil.rmtree(save_dir)
            os.makedirs(save_dir)
        tx_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture0'), '{}_all_images_step{}'.format('syn_texture0', step))
        tx_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture0'), '{}_all_annos_step{}'.format('syn_texture0', step))

        clr_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color0'), '{}_all_images_step{}'.format('syn_color0', step))
        clr_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color0'), '{}_all_annos_step{}'.format('syn_color0', step))

        tx_image_files = np.sort(glob(os.path.join(tx_img_dir, '*' + IMG_FORMAT)))
        tx_img_names = [os.path.basename(f) for f in tx_image_files]
        clr_image_files = np.sort(glob(os.path.join(clr_img_dir, '*' + IMG_FORMAT)))
        clr_img_names = [os.path.basename(f) for f in clr_image_files]

        tx_lbl_files = np.sort(glob(os.path.join(tx_lbl_dir, '*' + IMG_FORMAT)))
        # tx_lbl_names = [os.path.basename(f) for f in tx_lbl_files]
        clr_lbl_files = np.sort(glob(os.path.join(clr_lbl_dir, '*' + IMG_FORMAT)))
        # clr_lbl_names = [os.path.basename(f) for f in clr_lbl_files]

        for ix in range(len(tx_image_files)):
            file1 = tx_image_files[ix]
            file2 = os.path.join(clr_img_dir, clr_img_names[ix])

            img_file1 = io.imread(file1)
            img_file2 = io.imread(file2)

            file21 = tx_lbl_files[ix]
            file22 = clr_lbl_files[ix]

            img_file21 = io.imread(file21)
            img_file22 = io.imread(file22)

            fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

            axs[0, 0].imshow(img_file1)
            axs[0, 0].set_title('texture')
            axs[0, 1].imshow(img_file2)
            axs[0, 1].set_title('color')
            axs[1, 0].imshow(img_file21)
            axs[1, 1].imshow(img_file22)
            axs[1, 0].set_xlabel(tx_img_names[ix])
            axs[1, 1].set_xlabel(clr_img_names[ix])
            axs[1, 2].imshow(img_file21- img_file22)
            axs[1, 2].set_xlabel('left - right')
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, tx_img_names[ix].replace('texture_', '')))
            # plt.show()
    else:

        IMG_FORMAT = '.png'
        save_dir = os.path.join('/media/lab/Yang/data/synthetic_data/Airplanes/', 'check_rgb_gt0', 'all')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        tx_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture'), '{}_all_images_step{}'.format('syn_texture', step))
        tx_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture'), '{}_all_annos_step{}'.format('syn_texture', step))

        clr_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color'), '{}_all_images_step{}'.format('syn_color', step))
        clr_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color'), '{}_all_annos_step{}'.format('syn_color', step))

        mx_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_mixed'), '{}_all_images_step{}'.format('syn_mixed', step))
        mx_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_mixed'), '{}_all_annos_step{}'.format('syn_mixed', step))
        tx_image_files = np.sort(glob(os.path.join(tx_img_dir, '*' + IMG_FORMAT)))
        tx_img_names = [os.path.basename(f) for f in tx_image_files]
        clr_image_files = np.sort(glob(os.path.join(clr_img_dir, '*' + IMG_FORMAT)))
        clr_img_names = [os.path.basename(f) for f in clr_image_files]
        mx_image_files = np.sort(glob(os.path.join(mx_img_dir, '*' + IMG_FORMAT)))
        mx_img_names = [os.path.basename(f) for f in mx_image_files]

        tx_lbl_files = np.sort(glob(os.path.join(tx_lbl_dir, '*' + IMG_FORMAT)))
        # tx_lbl_names = [os.path.basename(f) for f in tx_lbl_files]
        clr_lbl_files = np.sort(glob(os.path.join(clr_lbl_dir, '*' + IMG_FORMAT)))
        # clr_lbl_names = [os.path.basename(f) for f in clr_lbl_files]
        mx_lbl_files = np.sort(glob(os.path.join(mx_lbl_dir, '*' + IMG_FORMAT)))
        # mx_lbl_names = [os.path.basename(f) for f in mx_lbl_files]
        for ix in range(len(tx_image_files)):
            file1 = tx_image_files[ix]
            file2 = os.path.join(clr_img_dir, clr_img_names[ix])
            file3 = os.path.join(mx_img_dir, mx_img_names[ix])

            img_file1 = io.imread(file1)
            img_file2 = io.imread(file2)
            img_file3 = io.imread(file3)

            file21 = tx_lbl_files[ix]
            file22 = clr_lbl_files[ix]
            file23 = mx_lbl_files[ix]

            img_file21 = io.imread(file21)
            img_file22 = io.imread(file22)
            img_file23 = io.imread(file23)

            fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

            axs[0, 0].imshow(img_file1)
            axs[0, 0].set_title('texture')
            axs[0, 1].imshow(img_file2)
            axs[0, 1].set_title('color')
            axs[0, 2].imshow(img_file3)
            axs[0, 2].set_title('mixed')
            axs[1, 0].imshow(img_file21)
            axs[1, 1].imshow(img_file22)
            axs[1, 2].imshow(img_file23)
            axs[1, 0].set_xlabel(tx_img_names[ix])
            axs[1, 1].set_xlabel(clr_img_names[ix])
            axs[1, 2].set_xlabel(mx_img_names[ix])
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, tx_img_names[ix].replace('texture_', '')))
            # plt.show()


def compare_first_second_dataset():
    syn_args = pps.get_syn_args()
    syn_plane_img_anno_dir = '/media/lab/Yang/data/synthetic_data/Airplanes/{}'
    save_dir = os.path.join('/media/lab/Yang/data/synthetic_data/Airplanes/', 'comp_first_sec_rgb_gt', 'rgb')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    step = syn_args.tile_size * syn_args.resolution

    IMG_FORMAT0 = '.jpg'
    tx0_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture0'), '{}_all_images_step{}'.format('syn_texture0', step))
    tx0_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture0'), '{}_all_annos_step{}'.format('syn_texture0', step))

    clr0_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color0'), '{}_all_images_step{}'.format('syn_color0', step))
    clr0_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color0'), '{}_all_annos_step{}'.format('syn_color0', step))

    tx0_image_files = np.sort(glob(os.path.join(tx0_img_dir, '*' + IMG_FORMAT0)))
    tx0_img_names = [os.path.basename(f) for f in tx0_image_files]
    clr0_image_files = np.sort(glob(os.path.join(clr0_img_dir, '*' + IMG_FORMAT0)))
    clr0_img_names = [os.path.basename(f) for f in clr0_image_files]

    # tx0_lbl_files = np.sort(glob(os.path.join(tx0_lbl_dir, '*' + IMG_FORMAT0)))
    # # tx0_lbl_names = [os.path.basename(f) for f in tx_lbl_files]
    # clr0_lbl_files = np.sort(glob(os.path.join(clr0_lbl_dir, '*' + IMG_FORMAT0)))

    IMG_FORMAT = '.png'
    tx_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture'), '{}_all_images_step{}'.format('syn_texture', step))
    tx_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture'), '{}_all_annos_step{}'.format('syn_texture', step))

    clr_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color'), '{}_all_images_step{}'.format('syn_color', step))
    clr_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color'), '{}_all_annos_step{}'.format('syn_color', step))
    tx_image_files = np.sort(glob(os.path.join(tx_img_dir, '*' + IMG_FORMAT)))
    tx_img_names = [os.path.basename(f) for f in tx_image_files]
    clr_image_files = np.sort(glob(os.path.join(clr_img_dir, '*' + IMG_FORMAT)))
    clr_img_names = [os.path.basename(f) for f in clr_image_files]

    # tx_lbl_files = np.sort(glob(os.path.join(tx_lbl_dir, '*' + IMG_FORMAT)))
    # # tx_lbl_names = [os.path.basename(f) for f in tx_lbl_files]
    # clr_lbl_files = np.sort(glob(os.path.join(clr_lbl_dir, '*' + IMG_FORMAT)))
    # # clr_lbl_names = [os.path.basename(f) for f in clr_lbl_files]
    tx_img_not_in_tx0 = [f for f in tx_img_names if f.replace(IMG_FORMAT, IMG_FORMAT0) not in tx0_img_names]
    tx0_img_not_in_tx = [f for f in tx0_img_names if f.replace(IMG_FORMAT0, IMG_FORMAT) not in tx_img_names]
    print('tx_img_not_in_tx0', len(tx_img_not_in_tx0), tx_img_not_in_tx0)
    print('tx0_img_not_in_tx', len(tx0_img_not_in_tx), tx0_img_not_in_tx)
    for ix in range(len(tx0_img_names)):
        if tx0_img_names[ix] in tx0_img_not_in_tx:
            continue
        txf0 = os.path.join(tx0_img_dir, tx0_img_names[ix])
        clf0 = os.path.join(clr0_img_dir, clr0_img_names[ix])
        txf = os.path.join(tx_img_dir, tx0_img_names[ix].replace(IMG_FORMAT0, IMG_FORMAT))
        clf = os.path.join(clr_img_dir, clr0_img_names[ix].replace(IMG_FORMAT0, IMG_FORMAT))

        img_file01 = io.imread(txf0)
        img_file02 = io.imread(clf0)
        img_file1 = io.imread(txf)
        img_file2 = io.imread(clf)

        fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

        axs[0, 0].imshow(img_file01)
        axs[0, 0].set_title('syn_texture0')
        axs[1, 0].imshow(img_file02)
        axs[1, 0].set_title('syn_color0')
        axs[0, 1].imshow(img_file1)
        axs[0, 1].set_title('syn_texture')
        axs[1, 1].imshow(img_file2)
        axs[1, 1].set_title('syn_color')
        axs[0, 2].imshow(img_file01 - img_file1)
        axs[0, 2].set_title('syn_texture0 - syn_texture')
        axs[1, 2].imshow(img_file01 - img_file1)
        axs[1, 2].set_title('syn_color0 - syn_color')
        axs[0, 0].set_ylabel(tx0_img_names[ix])
        axs[1, 0].set_ylabel(clr0_img_names[ix])
        plt.tight_layout()
        # plt.show()
        fig.savefig(os.path.join(save_dir, tx0_img_names[ix].replace('texture_', '')))

    tx_save_dir = os.path.join('/media/lab/Yang/data/synthetic_data/Airplanes/', 'comp_first_sec_rgb_gt', 'tx_img_not_in_tx0')
    if not os.path.exists(tx_save_dir):
        os.makedirs(tx_save_dir)
    for tx in tx_img_not_in_tx0:
        txf = os.path.join(tx_img_dir, tx)
        clf = os.path.join(clr_img_dir, tx.replace('texture_', 'color_'))
        tx_lbl = os.path.join(tx_lbl_dir, tx)
        clf_lbl = os.path.join(clr_lbl_dir, tx.replace('texture_', 'color_'))
        img_file1 = io.imread(txf)
        img_file2 = io.imread(clf)
        img_file11 = io.imread(tx_lbl)
        img_file12 = io.imread(clf_lbl)
        fig, axs = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)

        axs[0, 0].imshow(img_file1)
        axs[0, 0].set_title('syn_texture')
        axs[0, 1].imshow(img_file2)
        axs[0, 1].set_title('syn_color')
        axs[1, 0].imshow(img_file11)
        # axs[1, 0].set_title('syn_texture')
        axs[1, 1].imshow(img_file12)
        # axs[1, 1].set_title('syn_color')
        axs[1, 0].set_xlabel(tx)
        axs[1, 1].set_xlabel(tx.replace('texture_', 'color_'))
        plt.tight_layout()
        fig.savefig(os.path.join(tx_save_dir, tx.replace('texture_', '')))

    tx0_save_dir = os.path.join('/media/lab/Yang/data/synthetic_data/Airplanes/', 'comp_first_sec_rgb_gt', 'tx0_img_not_in_tx')
    if not os.path.exists(tx0_save_dir):
        os.makedirs(tx0_save_dir)
    for tx0 in tx0_img_not_in_tx:
        txf0 = os.path.join(tx0_img_dir, tx0)
        clf0 = os.path.join(clr0_img_dir, tx0.replace('texture_', 'color_'))
        txf0_lbl = os.path.join(tx0_lbl_dir, tx0)
        clf0_lbl = os.path.join(clr0_lbl_dir, tx0.replace('texture_', 'color_'))
        img_file01 = io.imread(txf0)
        img_file02 = io.imread(clf0)
        img_file11 = io.imread(txf0_lbl)
        img_file12 = io.imread(clf0_lbl)

        fig, axs = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)

        axs[0, 0].imshow(img_file01)
        axs[0, 0].set_title('syn_texture')
        axs[0, 1].imshow(img_file02)
        axs[0, 1].set_title('syn_color')
        axs[1, 0].imshow(img_file11)
        # axs[0, 0].set_title('syn_texture')
        axs[1, 1].imshow(img_file12)
        # axs[0, 1].set_title('syn_color')
        axs[1, 0].set_xlabel(tx0)
        axs[1, 1].set_xlabel(tx0.replace('texture_', 'color_'))
        plt.tight_layout()
        fig.savefig(os.path.join(tx0_save_dir, tx0.replace('texture_', '')))


if __name__ == "__main__":
    # display_type = ['syn_texture', 'syn_color', 'syn_mixed']
    # # display_type = ['syn_texture0', 'syn_color0']
    # city = ['barcelona', 'berlin', 'francisco', 'hexagon', 'radial', 'siena', 'spiral']
    # streets = [200, 200, 200, 200, 200, 250, 130]

    # two = True
    # compare_images_separate_cities(two)

    # two = True
    # compare_images_after_combine(two)

    # compare_first_second_dataset()

    '''
    check the different data selected in dataset1 and dataset2
    '''
    tex0_files = '/media/lab/Yang/code/yolov3/data_xview/1_cls/first_data_set_backup/xview_syn_texture_0.25_train_lbl.txt'
    # tex1_files = '/media/lab/Yang/code/yolov3/data_xview/1_cls/xview_syn_texture0_0.25/xview_syn_texture0_0.25_train_lbl_px4whr3.txt'
    tex1_files = '/media/lab/Yang/code/yolov3/data_xview/1_cls/xview_syn_texture_0.25/xview_syn_texture_0.25_train_lbl.txt'
    df_tex0 = pd.read_csv(tex0_files, header=None)
    df_tex1 = pd.read_csv(tex1_files, header=None)
    tex0_names = [os.path.basename(f) for f in df_tex0.loc[:, 0]]
    tex1_names = [os.path.basename(f) for f in df_tex1.loc[:, 0]]
    if np.all(tex0_names == tex1_names):
        print(True)
    else:
        print(False)
