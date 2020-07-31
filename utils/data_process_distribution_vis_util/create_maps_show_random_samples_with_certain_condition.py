import glob
import numpy as np
import argparse
import os
import wv_util as wv
import pandas as pd
from ast import literal_eval
import json
import datetime
from matplotlib import pyplot as plt
import cv2

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


def create_maps_for_cat_img_anns(typestr='all'):
    df_category_id = pd.read_csv(args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num), sep="\t")
    cat_names = df_category_id['category'].to_list()
    cat_ids = df_category_id['category_id'].to_list()
    # cat_labels = df_category_id['category_label'].to_list()
    annos_js = json.load(open(args.label_save_dir + 'xView{}_{}_{}cls_xtlytlwh.json'.format(typestr, args.input_size, args.class_num)))
    annos_list = annos_js['annotations']
    image_list = annos_js['images']
    image_dict = {}
    image_anno_ids_dict = {}
    image_to_cat_to_anno_ids_dict = {}
    cat_img_ids_dict = {}
    category_anno_ids_dict = {}

    #fixme
    # assign annotations to category which they are belonged to; category-a contains many annotations
    cat_img_ids_json_file = os.path.join(args.label_save_dir, '{}_cat_img_ids_dict_{}cls.json'.format(typestr, args.class_num))
    cat_annos_id_json_file = os.path.join(args.label_save_dir, '{}_cat_anno_ids_dict_{}cls.json'.format(typestr, args.class_num))
    image_id_name_json_file = os.path.join(args.label_save_dir, '{}_image_ids_names_dict_{}cls.json'.format(typestr, args.class_num))
    image_annos_id_json_file = os.path.join(args.label_save_dir, '{}_image_anno_ids_dict_{}cls.json'.format(typestr, args.class_num))
    image_to_cat_to_anno_ids_json_file = os.path.join(args.label_save_dir, '{}_image_to_cat_to_anno_ids_dict_{}cls.json'.format(typestr, args.class_num))

    for i in range(len(image_list)):
        image_dict[image_list[i]['id']] = image_list[i]['file_name']
        image_anno_ids_dict[image_list[i]['id']] = []
        image_to_cat_to_anno_ids_dict[image_list[i]['id']] = {}
        for j in range(len(cat_ids)):
            image_to_cat_to_anno_ids_dict[image_list[i]['id']][cat_ids[j]] = []

    for c in cat_ids:
        category_anno_ids_dict[c] = []
        cat_img_ids_dict[c] = []

    for j in range(len(annos_list)):
        category_anno_ids_dict[annos_list[j]['category_id']].append(j)
        image_anno_ids_dict[annos_list[j]['image_id']].append(j)
        image_to_cat_to_anno_ids_dict[annos_list[j]['image_id']][annos_list[j]['category_id']].append(j)
        if annos_list[j]['image_id'] not in cat_img_ids_dict[annos_list[j]['category_id']]:
            cat_img_ids_dict[annos_list[j]['category_id']].append(annos_list[j]['image_id'])

    json.dump(cat_img_ids_dict, open(cat_img_ids_json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)
    json.dump(category_anno_ids_dict, open(cat_annos_id_json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)
    json.dump(image_dict, open(image_id_name_json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)
    json.dump(image_anno_ids_dict, open(image_annos_id_json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)
    json.dump(image_to_cat_to_anno_ids_dict, open(image_to_cat_to_anno_ids_json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)


def show_cat_annos_by_cat_id(cat_id, N=2, typestr='all'):
    save_dir = args.cat_sample_dir + 'cat_id_2_img_with_gt_bbx_figures/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    df_category_id = pd.read_csv(args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num), sep="\t")
    cat_names = df_category_id['category'].to_list()
    cat_ids = df_category_id['category_id'].to_list()
    # cat_labels = df_category_id['category_label'].to_list()
    annos_js = json.load(open(args.label_save_dir + 'xView{}_{}_{}cls_xtlytlwh.json'.format(typestr, args.input_size, args.class_num)))
    annos_list = annos_js['annotations']
    # image_list = annos_js['images']

    cat_annos_id_json_file = os.path.join(args.label_save_dir, '{}_cat_anno_ids_dict_{}cls.json'.format(typestr, args.class_num))
    image_id_name_json_file = os.path.join(args.label_save_dir, '{}_image_ids_names_dict_{}cls.json'.format(typestr, args.class_num))
    image_annos_id_json_file = os.path.join(args.label_save_dir, '{}_image_anno_ids_dict_{}cls.json'.format(typestr, args.class_num))
    image_to_cat_to_anno_ids_json_file = os.path.join(args.label_save_dir, '{}_image_to_cat_to_anno_ids_dict_{}cls.json'.format(typestr, args.class_num))

    category_anno_ids_dict = json.load(open(cat_annos_id_json_file))
    image_dict = json.load(open(image_id_name_json_file))
    # image_anno_ids_dict = json.load(open(image_annos_id_json_file))
    image_to_cat_to_anno_ids_dict = json.load(open(image_to_cat_to_anno_ids_json_file))

    cat_name = cat_names[cat_ids.index(int(cat_id))]
    cat_name = cat_name.replace('/', '|') # note replace
    cat_anno_ids = category_anno_ids_dict.get(cat_id) # cat[i]--> anns
    np.random.seed(args.seed)
    rand_annos = np.random.permutation(cat_anno_ids)
    for r in range(N):
        image_id = str(annos_list[rand_annos[r]]['image_id']) # anns[r]--> img
        img_name = image_dict.get(image_id)
        img = cv2.imread(args.images_save_dir + img_name)
        anno_ids = image_to_cat_to_anno_ids_dict[image_id][cat_id] # img-->cat[i]-->anns
        for ai in range(len(anno_ids)):
            bbx = annos_list[anno_ids[ai]]['bbox']
            img = cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[0] + bbx[2], bbx[1] + bbx[3]), (0, 255, 0), 2)
            cv2.putText(img, text=str(cat_id), org=(bbx[0] + 10, bbx[1] + 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))
        cv2.imwrite(os.path.join(save_dir, 'cat{}_{}_sample{}_img{}.png'.format(cat_id, cat_name, r, img_name.split('.')[0])), img)


def draw_fig_with_all_cats_bbx_by_catid(cat_id, typestr='all'):
    """
    draw figure with bbx by catid
    """

    cat_img_ids_maps = json.load(open(os.path.join(args.label_save_dir, '{}_cat_img_ids_dict_{}cls.json'.format(typestr, args.class_num))))
    img_ids_names_maps = json.load(open(os.path.join(args.label_save_dir, '{}_image_ids_names_dict_{}cls.json'.format(typestr, args.class_num))))

    cat_img_names = []
    cat_img_ids = cat_img_ids_maps[cat_id]
    for i in range(len(cat_img_ids)):
        cat_img_names.append(img_ids_names_maps[str(cat_img_ids[i])])

    json_name = 'xView{}_{}_{}cls_xtlytlwh.json'.format(typestr, args.input_size, args.class_num)
    json_file = json.load(open(os.path.join(args.label_save_dir, json_name)))
    annos_list = json_file['annotations']

    fig_save_dir = args.cat_sample_dir + 'catid2imgs_all_cats_bbx_figures/'
    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)

    for ig in cat_img_ids:
        bbx_list = []
        cat_list = []
        for a in range(len(annos_list)):
            if annos_list[a]['image_id'] == ig:
                bbx_list.append(annos_list[a]['bbox'])
                cat_list.append(annos_list[a]['category_id'])

        img = cv2.imread(args.images_save_dir + img_ids_names_maps[str(ig)])
        df_cat_id_color = pd.read_csv('categories/categories_id_color_diverse_{}.txt'.format(args.class_num), sep="\t")
        for k in range(len(bbx_list)):
            c_id = cat_list[k]
            bbx = bbx_list[k]
            color = literal_eval(df_cat_id_color[df_cat_id_color['category_id'] == c_id]['color'].iloc[0])
            img = cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[0] + bbx[2], bbx[1] + bbx[3]), color, 2)

        cv2.imwrite(fig_save_dir + img_ids_names_maps[str(ig)], img)

if __name__ == '__main__':
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

    parser.add_argument("--annos_save_dir", type=str, help="to save txt annotation files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')

    parser.add_argument("--label_save_dir", type=str, help="to related label files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')

    parser.add_argument("--fig_save_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/figures/')

    parser.add_argument("--cat_sample_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/')

    parser.add_argument("--data_save_dir", type=str, help="to save data files",
                        default='../data_xview/')

    # parser.add_argument("--cat_dir", type=str, help="to save category files",
    #                     default='categories/')

    parser.add_argument("-ft2", "--font2", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 23}")

    parser.add_argument("--class_num", type=int, default=6, help="Number of Total Categories")  # 60  6
    parser.add_argument("--input_size", type=int, default=608, help="Number of Total Categories")
    parser.add_argument("--seed", type=int, default=1024, help="random seed")

    args = parser.parse_args()

    args.label_save_dir = args.label_save_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)
    args.annos_save_dir = args.annos_save_dir + '{}/{}_cls_xcycwh/'.format(args.input_size, args.class_num)
    args.images_save_dir = args.images_save_dir + '{}/'.format(args.input_size)
    args.cat_sample_dir = args.cat_sample_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)
    args.data_save_dir = args.data_save_dir + '{}_cls/'.format(args.class_num)

    if not os.path.exists(args.label_save_dir):
        os.makedirs(args.label_save_dir)
    if not os.path.exists(args.annos_save_dir):
        os.makedirs(args.annos_save_dir)
    if not os.path.exists(args.images_save_dir):
        os.makedirs(args.images_save_dir)
    if not os.path.exists(args.cat_sample_dir):
        os.makedirs(args.cat_sample_dir)
    if not os.path.exists(args.data_save_dir):
        os.makedirs(args.data_save_dir)

    '''
    1. create maps between cats imgs annos
    '''
    # typestr = "all"
    # create_maps_for_cat_img_anns()

    '''
    2. show *.tif that contains * category
       group according to categories
    '''
    # N = 2
    # cat_id = '43'
    # cat_id = '18'
    # cat_id = '33'
    # cat_id = '23'
    # cat_id = '21'
    # cat_id = '46' # ###
    # cat_id = '39'
    # cat_id = '47'
    # cat_id = '48'

    # cat_id = '36'
    # cat_id = '18'
    # cat_id = '46'
    # cat_id = '23'
    # cat_id = '37'
    # cat_id = '39'
    # cat_id = '21'
    # cat_id = '19'
    # cat_id = '48'
    # cat_id = '26'
    # cat_id = '30'
    # cat_id = '24'
    # cat_id = '47'
    # cat_id = '55'
    # cat_id = '0'
    # cat_id = '59'
    # cat_id = '22'
    # cat_id = '35'
    # cat_id = '54'
    # cat_id = '14'
    # cat_id = '20'
    # cat_id = '25'
    # cat_id = '0'
    # cat_id = '1'
    # cat_id = '2'
    # cat_id = '3'
    # cat_id = '4'
    # cat_id = '5'
    # typestr = "all"
    # show_cat_annos_by_cat_id(cat_id, N, typestr=typestr)


    '''
    cat-->imgs-->annos
    '''
    # args.images_save_dir = args.images_save_dir + '{}/'.format(args.input_size)
    # all_jpg_files = glob.glob(args.images_save_dir + '*.jpg')
    # df_category_id = pd.read_csv(args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num), sep="\t")
    # cat_names = df_category_id['category'].to_list()
    # cat_ids = df_category_id['category_id'].to_list()
    #
    # cat_imgs_map_dict = {}
    #
    # for c in cat_ids:
    #     cat_imgs_map_dict[c] = []


    '''
    draw figures with bbx by catid
    '''
    # cat_id = '18'
    # draw_fig_with_all_cats_bbx_by_catid(cat_id)


    # cat_anno_ids = category_anno_ids_dict.get(str(cat_id))
    # np.random.seed(args.seed)
    # rand_annos = np.random.permutation(cat_anno_ids)
    # if not os.path.exists(args.cat_sample_dir):
    #     os.makedirs(args.cat_sample_dir)
    # for r in range(N):
    #     image_id = annos_list[rand_annos[r]]['image_id']
    #     img = cv2.imread(args.images_save_dir + image_dict.get(image_id))
    #     anno_ids = image_to_cat_to_anno_ids_dict[image_id][cat_id]
    #     bbx = annos_list[rand_annos[r]]['bbox']
    #     img = cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[0] + bbx[2], bbx[1] + bbx[3]), (255, 0, 0), 2)
    #     cv2.putText(img, text=str(cat_id), org=(bbx[0] + 10, bbx[1] + 10),
    #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #                 fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))
    #     cv2.imwrite(os.path.join(args.cat_sample_dir, 'cat{}_sample{}.png'.format(cat_id, r)), img)
