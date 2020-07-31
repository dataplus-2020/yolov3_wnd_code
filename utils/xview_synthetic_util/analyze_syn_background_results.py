import glob
import numpy as np
import argparse
import os
import pandas as pd
from ast import literal_eval
import seaborn as sns
from matplotlib import pyplot as plt
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


def get_map_results_2seeds_4ratios_by_step(step=171):
    map_comp = np.zeros((4, 2))
    dt = 'syn_background'
    comments = '_px6whr4_ng0_seed1024'
    syn_ratio = [0, 0.1, 0.2, 0.3]
    for ix, sr in enumerate(syn_ratio):
        syn_args = get_part_syn_args(dt, sr)
        syn_args.results_dir = syn_args.results_dir.format(syn_args.class_num, syn_args.syn_display_type, sr)
        result_file = np.sort(glob.glob(os.path.join(syn_args.results_dir, '*' + comments.split('_seed')[0], 'results_{}_{}.txt'.format(syn_args.syn_display_type, sr))))[-1]
        map_comp[ix, 0] = np.loadtxt(result_file, usecols=[10])[step] # 0, 1,  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    comments1 = '_px6whr4_ng0_seed17'
    for jx, sr in enumerate(syn_ratio):
        syn_args = get_part_syn_args(dt, sr)
        syn_args.results_dir = syn_args.results_dir.format(syn_args.class_num, syn_args.syn_display_type, sr)
        result_file = np.sort(glob.glob(os.path.join(syn_args.results_dir, '*' + comments1, 'results_{}_{}.txt'.format(syn_args.syn_display_type, sr))))[-1]
        map_comp[jx, 1] = np.loadtxt(result_file, usecols=[10])[step] # 0, 1,  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    comments2 = '_px6whr4_ng0_seed1024_mosaic_rect'
    for ix, sr in enumerate(syn_ratio):
        syn_args = get_part_syn_args(dt, sr)
        syn_args.results_dir = syn_args.results_dir.format(syn_args.class_num, syn_args.syn_display_type, sr)
        result_file = np.sort(glob.glob(os.path.join(syn_args.results_dir, '*' + comments2.split('_seed')[0], 'results_{}_{}.txt'.format(syn_args.syn_display_type, sr))))[-1]
        map_comp[ix, 0] = np.loadtxt(result_file, usecols=[10])[step] # 0, 1,  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    comments3 = '_px6whr4_ng0_seed17_mosaic_rect'
    for jx, sr in enumerate(syn_ratio):
        syn_args = get_part_syn_args(dt, sr)
        syn_args.results_dir = syn_args.results_dir.format(syn_args.class_num, syn_args.syn_display_type, sr)
        result_file = np.sort(glob.glob(os.path.join(syn_args.results_dir, '*' + comments3, 'results_{}_{}.txt'.format(syn_args.syn_display_type, sr))))[-1]
        map_comp[jx, 1] = np.loadtxt(result_file, usecols=[10])[step] # 0, 1,  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    save_dir = os.path.join(syn_args.data_txt_dir, 'MAP_comp', dt)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df_map = pd.DataFrame(map_comp, columns=[comments[1:], comments1[1:]], index=syn_ratio)
    df_map.to_csv(os.path.join(save_dir, '{}_4ratio_px6whr4_ng0_5seeds_map_step{}.csv'.format(dt, step)))
    return df_map


def get_map_results_5seeds_4ratios_by_step(step=171):
    dt = 'syn_background'
    syn_ratio = [0, 0.1, 0.2, 0.3]
    split_seeds = [1024, 17, 3, 5, 7]
    map_arr = np.zeros((4, 5), dtype=np.float32)
    columns = []
    for ix, sr in enumerate(syn_ratio):
        for jx, sd in enumerate(split_seeds):
            syn_args = get_part_syn_args(dt, sr, sd)
            syn_args.results_dir = syn_args.results_dir.format(syn_args.class_num, syn_args.syn_display_type, sr)

            if sd == 0:
                comments = '_px6whr4_ng0_seed'
                columns.append('px6whr4_ng0_seed1024')
                result_file = np.sort(glob.glob(os.path.join(syn_args.results_dir, '*{}'.format(comments), 'results_{}_{}.txt'.format(syn_args.syn_display_type, sr))))[-1]
            else:
                comments = '_px6whr4_ng0_seed{}'.format(sd)
                columns.append(comments[1:])
                result_file = np.sort(glob.glob(os.path.join(syn_args.results_dir, '*{}*'.format(comments), 'results_{}_{}.txt'.format(syn_args.syn_display_type, sr))))[-1]

            # 0, 1,  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            map_arr[ix, jx] = np.loadtxt(result_file, usecols=[10])[step]

    save_dir = os.path.join(syn_args.data_txt_dir, 'MAP_comp', dt)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df_map = pd.DataFrame(map_arr, columns=columns, index=syn_ratio)
    df_map.to_csv(os.path.join(save_dir, '{}_4ratio_px6whr4_ng0_5seeds_map_step{}.csv'.format(dt, step)))
    return df_map


def get_map_json_results_5seeds_4ratios(epochs=179):
    '''
    :return: json {ratio0:[seed0: [], seed1: [] ...]...}
    '''
    dt = 'syn_background'
    syn_ratio = [0, 0.1, 0.2, 0.3]
    split_seeds = [1024, 17, 3, 5, 9]
    map_json = {}
    columns = []
    for ix, sr in enumerate(syn_ratio):
        if sr not in map_json.keys():
            map_json[sr] = {}
        print('sr', sr)
        for jx, sd in enumerate(split_seeds):
            if sd not in map_json[sr].keys():
                map_json[sr][sd] = []
            syn_args = get_part_syn_args(dt, sr, sd)
            syn_args.results_dir = syn_args.results_dir.format(syn_args.class_num, syn_args.syn_display_type, sr)
            print('sd', sd)
            if sd == 1024:
                comments = '_px6whr4_ng0'
                columns.append('px6whr4_ng0_seed1024')
                result_file = np.sort(glob.glob(os.path.join(syn_args.results_dir, '*{}'.format(comments), 'results_{}_{}.txt'.format(syn_args.syn_display_type, sr))))[-1]
            else:
                comments = '_px6whr4_ng0_seed{}'.format(sd)
                columns.append(comments[1:])
                result_file = np.sort(glob.glob(os.path.join(syn_args.results_dir, '*{}'.format(comments), 'results_{}_{}.txt'.format(syn_args.syn_display_type, sr))))[-1]

            map_json[sr][sd] = np.loadtxt(result_file, usecols=[10])[:epochs].tolist()
    save_dir = os.path.join(syn_args.data_txt_dir, 'MAP_comp', dt)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    json_file = os.path.join(save_dir, '{}_px6whr4_ng0_4ratio_5seeds_map_epochs{}.json'.format(dt, epochs))
    json.dump(map_json, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)
    return json_file


def plot_syn_background_4ratios_5seeds_map(json_file, epochs=179):
    dict_map = json.load(open(json_file))
    dt = 'syn_background'
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    keys = [k for k in dict_map.keys()]
    for ix, sr in enumerate(keys[:2]):
        for sd in dict_map[sr].keys():
            y = dict_map[sr][sd]
            x = np.arange(len(y))
            if sd == '1024' or sd == '17':
                ysd = 0
            else:
                ysd = sd
            axs[0, ix].plot(x, y, '-', label='seed {} yolov3 seed {}'.format(sd, ysd))
        axs[0, ix].set_xlabel('Epochs')
        axs[0, ix].set_ylabel('MAP')
        axs[0, ix].set_title('Syn_background Ratio {}'.format(sr))
        axs[0, ix].grid(True)
        axs[0, ix].legend()
    for ix, sr in enumerate(keys[2:]):
        for sd in dict_map[sr].keys():
            y = dict_map[sr][sd]
            x = np.arange(len(y))
            if sd == '1024' or sd == '17':
                ysd = 0
            else:
                ysd = sd
            axs[1, ix].plot(x, y, '-', label='seed {} yolov3 seed {}'.format(sd, ysd))
        axs[1, ix].set_xlabel('Epochs')
        axs[1, ix].set_ylabel('MAP')
        axs[1, ix].set_title('Syn_background Ratio {}'.format(sr))
        axs[1, ix].grid(True)
        axs[1, ix].legend()

    fig.suptitle('{}_px6whr4_ng0_4ratio_5seeds_map_epochs'.format(dt), fontsize=14)
    syn_args = get_part_syn_args()
    save_dir = os.path.join(syn_args.data_txt_dir, 'MAP_comp', dt)
    fig.savefig(os.path.join(save_dir, '{}_px6whr4_ng0_4ratio_5seeds_map_epochs2.jpg'.format(dt)))


def boxplot_5seeds_4ratios(axis='ratio'):
    syn_args = get_part_syn_args()
    dt = 'syn_background'
    epochs = 180 - 1
    json_file = get_map_json_results_5seeds_4ratios(epochs) # (4, 5)
    json_map = json.load(open(json_file))
    ratios = [k for k in json_map.keys()] # ratio
    map_ratios = np.zeros((5, 4))
    map_seeds = np.zeros((4, 5))
    step = 171
    seeds = []
    for ix, sr in enumerate(ratios):
        if not seeds:
            seeds = [k for k in json_map[sr].keys()] # seeds
        print('ix--', ix)
        for jx, sd in enumerate(seeds):
            print('jx', jx)
            map_ratios[jx, ix] = json_map[sr][sd][step]
            map_seeds[ix, jx] = json_map[sr][sd][step]

    if axis == 'ratio':
        '''
        axis ratios 
        '''
        df_map_ratios = pd.DataFrame(map_ratios, columns=ratios, index=seeds)
        boxplot0 = df_map_ratios.boxplot(column=[df_map_ratios.columns[0], df_map_ratios.columns[1], df_map_ratios.columns[2], df_map_ratios.columns[3]], return_type='axes')
        boxplot0.set_title('syn_background_4ratio_to_5seeds(1024, 17, 3, 5, 9)')
        boxplot0.set_xlabel('{}_ratio'.format(dt))
        boxplot0.set_ylabel('MAP')
        save_dir = os.path.join(syn_args.data_txt_dir, 'MAP_comp', dt)
        boxplot0.figure.savefig(os.path.join(save_dir, '{}_5seeds_4ratio_boxplot_axis_ratio.jpg'.format(dt)))
    else:
        '''
        axis seeds
        '''
        df_map_seeds = pd.DataFrame(map_seeds, columns=seeds, index=ratios)
        boxplot1 = df_map_seeds.boxplot(column=[df_map_seeds.columns[0], df_map_seeds.columns[1], df_map_seeds.columns[2], df_map_seeds.columns[3], df_map_seeds.columns[4]], return_type='axes')
        boxplot1.set_title('syn_background_5seeds_to_4ratio (0, 0.1 0.2, 0.3)')
        boxplot1.set_xlabel('{}_seed'.format(dt))
        boxplot1.set_ylabel('MAP')
        save_dir = os.path.join(syn_args.data_txt_dir, 'MAP_comp', dt)
        boxplot1.figure.savefig(os.path.join(save_dir, '{}_5seeds_4ratio_boxplot_axis_seed.jpg'.format(dt)))


def get_part_syn_args(dt='syn_background', sr=0, seed=1024):
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

    parser.add_argument("--syn_display_type", type=str, default=dt,
                        help="syn_background")  # ######*********************change
    parser.add_argument("--syn_ratio", type=float, default=sr,
                        help="Percent")
    parser.add_argument("--seed", type=int, default=seed,
                        help="data split seed")

    parser.add_argument("--min_region", type=int, default=100, help="the smallest #pixels (area) to form an object")
    parser.add_argument("--link_r", type=int, default=15,
                        help="the #pixels between two connected components to be grouped")

    parser.add_argument("--resolution", type=float, default=0.3, help="resolution of synthetic data")

    syn_args = parser.parse_args()
    syn_args.data_xview_dir = syn_args.data_xview_dir.format(syn_args.class_num)
    syn_args.data_txt_dir = syn_args.data_txt_dir.format(syn_args.tile_size, syn_args.class_num)
    syn_args.cat_sample_dir = syn_args.cat_sample_dir.format(syn_args.tile_size, syn_args.class_num)

    syn_args.syn_images_save_dir = syn_args.syn_images_save_dir.format(syn_args.tile_size, syn_args.syn_display_type)
    syn_args.syn_annos_save_dir = syn_args.syn_annos_save_dir.format(syn_args.tile_size, syn_args.syn_display_type, syn_args.class_num)
    syn_args.syn_txt_save_dir = syn_args.syn_txt_save_dir.format(syn_args.tile_size, syn_args.syn_display_type, syn_args.class_num)

    return syn_args


if __name__ == '__main__':
    # ('%10s' * 2 + '%10.3g' * 6) % (
    #                 '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
    # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
    '''
    boxplot different seeds different ratio and two data split seeds: 1024, 17
    '''
    '''
    syn_background_with_different_ratio (0, 0.1, 0.2, 0.3)
    seed (1024, 17)
    '''
    # step = 171
    # df_map = get_map_results_2seeds_4ratios_by_step(step)

    # boxplot = df_map.boxplot(column=[df_map.columns[0], df_map.columns[1]], return_type='axes')
    # boxplot.set_title('syn_background_with_different_ratio (0, 0.1, 0.2, 0.3)')
    # boxplot.set_xlabel('{}_seed'.format(dt))
    # boxplot.set_ylabel('MAP')
    # boxplot.figure.savefig(os.path.join(save_dir, '{}_different_ratio_boxplot.jpg'.format(dt, comments1)))
    '''
    syn_background_with_different_seed (1024, 17)
    ratio (0, 0.1, 0.2, 0.3)
    '''
    # step = 171
    # df_map = get_map_results_2seeds_4ratios_by_step(step)
    # df_map = df_map.T
    # boxplot = df_map.boxplot(column=[df_map.columns[0], df_map.columns[1], df_map.columns[2], df_map.columns[3]], return_type='axes')
    # boxplot.set_title('{}_with_different_seed (1024, 17)'.format(dt))
    # boxplot.set_xlabel('{}_ratio'.format(dt))
    # boxplot.set_ylabel('MAP')
    # boxplot.figure.savefig(os.path.join(save_dir, '{}_different_seed_boxplot.jpg'.format(dt, comments1)))

    '''
    5 data split seeds 1024 17 3 5 9
    5 yolov3 seeds 0 0 3 5 9
    4 ratios 0 0.1 0.2 0.3 
    MAP json
    MAP jpg
    '''
    epochs = 180 - 1 # drop the last one
    json_file = get_map_json_results_5seeds_4ratios(epochs) # (4, 5)

    plot_syn_background_4ratios_5seeds_map(json_file, epochs)

    '''
    boxplot
    x-axis: syn_background_ratio
    legend: seed
    '''
    # axis='ratio'
    # axis='seed'
    # boxplot_5seeds_4ratios(axis)
