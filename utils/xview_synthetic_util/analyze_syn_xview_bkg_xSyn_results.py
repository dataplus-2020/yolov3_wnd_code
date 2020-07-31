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


def get_map_results_5seeds_by_step(step=171):

    comments = 'px6whr4_hgiou1'
    split_seeds = [1024, 17, 3, 5, 9]
    map_arr = np.zeros((5, 1), dtype=np.float32)
    for jx, sd in enumerate(split_seeds):
        dt = 'xview_background_double_seed{}'.format(sd)
        syn_args = get_part_syn_args(sd)
        syn_args.results_dir = syn_args.results_dir.format(syn_args.class_num, dt)
        # print(os.path.join(syn_args.results_dir, '*{}'.format(comments), 'results_background_dobule_seed{}.txt'.format(sd)))
        # /media/lab/Yang/code/yolov3/result_output/1_cls/xview_background_double_seed1024/2020-04-03_22.14_px6whr4_hgiou1/results_background_double_seed1024.txt
        result_file = np.sort(glob.glob(os.path.join(syn_args.results_dir, '*{}'.format(comments), 'results_background_double_seed{}.txt'.format(sd))))[-1]

        # 0, 1,  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        map_arr[jx] = np.loadtxt(result_file, usecols=[10])[step]

    save_dir = os.path.join(syn_args.data_txt_dir, 'MAP_comp', 'xview_background_double')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df_map = pd.DataFrame(map_arr, columns=[comments], index=None)
    df_map.to_csv(os.path.join(save_dir, 'xview_background_double_px6whr4_hgiou1_5seeds_map_step{}.csv'.format(step)))
    return df_map


def get_map_json_results_5seeds(epochs=179):
    '''
    :return: json {seed0: [], seed1: [] ......}
    '''
    split_seeds = [17] #  [1024, 17, 3, 5, 9]
    ratios = [0, 1, 2]
    # ratios = [0, 1]
    map_json = {}
    # cmt = 'xview_syn_xview_bkg_texture'
    # cmt = 'xview_syn_xview_bkg_color'
    cmt = 'xview_syn_xview_bkg_mixed'
    for jx, sd in enumerate(split_seeds):
        if sd not in map_json.keys():
            map_json[sd] = {}
        syn_args = get_part_syn_args(sd)

        comments = '_hgiou1_seed{}'.format(sd)
        for r in ratios:
            if r not in map_json[sd].keys():
                map_json[sd][r] = []
            if r == 0:
                 dt = 'px6whr4_ng0_seed{}'.format(sd)
                 results_dir = syn_args.results_dir.format(syn_args.class_num, dt)
                 result_file = np.sort(glob.glob(os.path.join(results_dir, '*{}*'.format(comments), 'results_{}.txt'.format(dt))))[-1]
            else:
                dt = '{}_seed{}'.format(cmt, sd)
                results_dir = syn_args.results_dir.format(syn_args.class_num, dt)
                dt = dt + '_{}xSyn'.format(r)
                result_file = np.sort(glob.glob(os.path.join(results_dir, '*{}*'.format(comments), 'results_*seed{}_{}xSyn.txt'.format(sd, r))))[-1]

            map_json[sd][r] = np.loadtxt(result_file, usecols=[10])[:epochs].tolist()

    save_dir = os.path.join(syn_args.data_txt_dir, 'MAP_comp', cmt)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    json_file = os.path.join(save_dir, '{}_3ratios_5seeds_map_epochs{}.json'.format(cmt, epochs))
    json.dump(map_json, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)
    return json_file


def plot_xview_syn_xview_bkg_5seeds_map_by_dt(json_file):
    dict_map = json.load(open(json_file))
    cmt = 'xview_syn_xview_bkg_mixed'
    dt = 'mixed'
    # cmt = 'xview_syn_xview_bkg_texture'
    # dt = 'texture'
    # cmt = 'xview_syn_xview_bkg_color'
    # dt = 'color'
    fig, axs = plt.subplots(figsize=(15, 10))
    keys = [k for k in dict_map.keys()]
    for ix, sd in enumerate(keys):
        for r in dict_map[sd].keys():
            y = dict_map[sd][r]
            x = np.arange(len(y))
            axs.plot(x, y, '-', label='syn_{}_ratio {} seed {}'.format(dt, r, sd))
            axs.set_xlabel('Epochs', fontsize=16)
            axs.set_ylabel('MAP', fontsize=16)
            axs.grid(True)
            axs.legend()
    fig.suptitle('{}_3ratios_px6whr4_hgiou1_map_epochs'.format(cmt), fontsize=18)
    syn_args = get_part_syn_args()
    save_dir = os.path.join(syn_args.data_txt_dir, 'MAP_comp', cmt)
    fig.savefig(os.path.join(save_dir, '{}_3ratios_px6whr4_hgiou1_map_epochs.jpg'.format(cmt)))


def plot_xview_syn_xview_bkg_3dts_map(seed=17):
    # dict_map = json.load(open(json_file))
    # cmt = 'xview_syn_xview_bkg_mixed'
    # dt = 'mixed'
    # cmt = 'xview_syn_xview_bkg_texture'
    # dt = 'texture'
    # cmt = 'xview_syn_xview_bkg_color'
    # dt = 'color'
    syn_args = get_part_syn_args(seed)
    comments = ['xview_syn_xview_bkg_texture', 'xview_syn_xview_bkg_color', 'xview_syn_xview_bkg_mixed']
    display_types = ['texture', 'color', 'mixed']
    sd = '{}'.format(seed)
    fig, axs = plt.subplots(figsize=(15, 10))
    for ix, cmt in enumerate(comments):
        dict_map = json.load(open(os.path.join(syn_args.data_txt_dir, 'MAP_comp', cmt, '{}_3ratios_5seeds_map_epochs{}.json'.format(cmt, 179))))
        for r in dict_map[sd].keys():
            if ix > 0 and r == '0':
                continue
            dt = display_types[ix] if r!='0' else ''
            y = dict_map[sd][r]
            x = np.arange(len(y))
            axs.plot(x, y, '-', label='syn_{}_ratio {} seed {}'.format(dt, r, sd))
            axs.set_xlabel('Epochs', fontsize=16)
            axs.set_ylabel('MAP', fontsize=16)
            axs.grid(True)
            axs.legend()
    fig.suptitle('{}_ratio{}_3displaye_types_px6whr4_hgiou1_map_epochs'.format(cmt, r), fontsize=18)
    syn_args = get_part_syn_args()
    save_dir = os.path.join(syn_args.data_txt_dir, 'MAP_comp', cmt)
    fig.savefig(os.path.join(save_dir, '{}_3ratios_3displaye_types_px6whr4_hgiou1_map_epochs.jpg'.format(cmt)))


def boxplot_5seeds(axis='seeds'):
    syn_args = get_part_syn_args()
    dt = 'xview_background_double'
    epochs = 180 - 1
    json_file = get_map_json_results_5seeds(epochs) # (4, 5)
    json_map = json.load(open(json_file))
    map_seeds = np.zeros((5, 1))
    step = 171
    seeds = ['1024', '17', '3', '5', '9']
    for jx, sd in enumerate(seeds):
        print('jx', jx)
        map_seeds[jx] = json_map[sd][step]

    '''
    axis px6whr4
    '''
    df_map_seeds = pd.DataFrame(map_seeds, columns=['px6whr4_hgiou1'], index=None)
    boxplot1 = df_map_seeds.boxplot(column=[df_map_seeds.columns[0]], return_type='axes')
    boxplot1.set_title('xview_background_double_5seeds_px6whr4_hgiou1 [1024, 17, 3, 5, 9]')
    boxplot1.set_ylabel('MAP')
    save_dir = os.path.join(syn_args.data_txt_dir, 'MAP_comp', dt)
    boxplot1.figure.savefig(os.path.join(save_dir, 'xview_background_double_boxplot_of_seed.jpg'.format(dt)))


def get_part_syn_args(seed=1024):
    parser = argparse.ArgumentParser()

    parser.add_argument("--syn_bkg_img_anno_dir", type=str, help="images and annotations of synthetic airplanes",
                        default='/media/lab/Yang/data/synthetic_data/syn_background/')

    parser.add_argument("--syn_bkg_txt_dir", type=str, help="txt labels of synthetic airplanes",
                        default='/media/lab/Yang/data/synthetic_data/syn_background_txt_xcycwh/{}/')

    parser.add_argument("--syn_bkg_gt_bbox_dir", type=str, help="gt images with bbox of synthetic airplanes",
                        default='/media/lab/Yang/data/synthetic_data/syn_background_gt_bbox/{}/')

    parser.add_argument("--data_txt_dir", type=str, help="to save txt files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_cls/')

    parser.add_argument("--data_xview_dir", type=str, help="to save data files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_cls/')

    parser.add_argument("--syn_data_list_dir", type=str, help="to syn data list files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_{}_cls/')

    parser.add_argument("--results_dir", type=str, help="to save category files",
                        default='/media/lab/Yang/code/yolov3/result_output/{}_cls/{}/')

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

    return syn_args


if __name__ == '__main__':
    # ('%10s' * 2 + '%10.3g' * 6) % (
    #                 '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
    # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
    '''
    boxplot different seeds : 1024, 17, 3, 5, 9
    '''
    # step = 171
    # df_map = get_map_results_5seeds_by_step(step)

    '''
    5 data split seeds 1024 17 3 5 9
    5 yolov3 seeds 0 0 3 5 9
    4 ratios 0 1 2 
    MAP json
    MAP jpg
    '''
    # epochs = 180 - 1 # drop the last one
    # json_file = get_map_json_results_5seeds(epochs)
    # plot_xview_syn_xview_bkg_5seeds_map_by_dt(json_file)

    # plot_xview_syn_xview_bkg_3dts_map(17)
    '''
    boxplot
    '''
    # axis='seed'
    # boxplot_5seeds(axis)
