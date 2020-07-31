"""
Copyright 2018 Defense Innovation Unit Experimental
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
import os
import cv2
import csv
from ast import literal_eval
"""
xView processing helper functions for use in data processing.
"""


def scale(x, range1=(0, 0), range2=(0, 0)):
    """
    Linear scaling for a value x
    """
    return range2[0] * (1 - (x - range1[0]) / (range1[1] - range1[0])) + range2[1] * (
            (x - range1[0]) / (range1[1] - range1[0]))


def get_image(fname):
    """
    Get an image from a filepath in ndarray format H, W, C
    """
    return np.array(Image.open(fname))

def get_labels(fname, catNum=60):
    """
    Gets label data from a geojson label file

    Args:
        fname: file path to an xView geojson label file

    Output:
        Returns three arrays: coords, chips, and classes corresponding to the
            coordinates, file-names, and classes for each ground truth.
    """
    with open(fname) as f:
        data = json.load(f)

    # l_coords = np.zeros((len(data['features']), 4))
    # l_chips = np.zeros((len(data['features'])), dtype="object")
    # l_classes = np.ones((len(data['features']))) * (-1)
    # feature_ids = np.ones((len(data['features'])), dtype=np.int64) * (-1)
    l_coords = []
    l_chips = []
    l_classes = []
    feature_ids = []
    if catNum == 60:
        cat_index_id_file = '../../data_xview/{}_cls/categories_id_color_diverse_{}.txt'.format(catNum, catNum)
    elif catNum == 6:
        cat_index_id_file = '../../data_xview/{}_cls/categories_id_color_{}_new_group.txt'.format(catNum, catNum)
    else:
        cat_index_id_file = '../../data_xview/{}_cls/categories_id_label_children_color_{}_classes.txt'.format(catNum, catNum)


    df_cat_idx_id = pd.read_csv(cat_index_id_file, delimiter='\t')
    feas = data['features']
    for i in range(len(feas)):

        '''
        # when the category_label==type_id, l_classes[i]=category_id corresponding to category_label
        '''
        df_cat_type = df_cat_idx_id[df_cat_idx_id['category_label'] == feas[i]['properties']['type_id']]
        # skip 75 82
        if df_cat_type.empty:
            continue
        '''
        cat num 
        '''
        l_classes.append(df_cat_type['category_id'].iloc[0])

        b_id = feas[i]['properties']['image_id']
        imcoords = np.array(literal_eval(feas[i]['properties']['bounds_imcoords']))
        l_chips.append(b_id)

        ''' cat num 62'''
        # if df_cat_type.empty:
        #     # print(i, data['features'][i]['properties']['type_id'])
        #     if data['features'][i]['properties']['type_id'] == 75:
        #         l_classes[i] = 60
        #     elif data['features'][i]['properties']['type_id'] == 82:
        #         l_classes[i] = 61
        #     else:
        #         print(i, data['features'][i]['properties']['type_id'])
        #         l_classes[i] = -1
        # else:
        #     l_classes[i] = df_cat_type['category_id'].iloc[0]

        # feature_ids[i] = data['features'][i]['properties']['feature_id']
        feature_ids.append(feas[i]['properties']['feature_id'])

        if imcoords.shape[0] != 4:
            print("Issues at %d!" % i)
        else:
            # l_coords[i] = imcoords
            l_coords.append(imcoords) #  # (w1, h1, w2, h2)

    return np.array(l_coords), np.array(l_chips), np.array(l_classes), np.array(feature_ids)
    # return l_coords, l_chips, l_classes, feature_ids


def get_all_categories(catNum=60):
    """
    Gets categories from a geojson label file
    create category info

    Output:
        Returns one list: categories.
    """
    categories = []

    # df_category_id = pd.read_csv('categories_id_color_diverse_60.txt', sep="\t")
    # df_category_id = pd.read_csv('../../data_xview/{}_cls/categories_id_color_diverse_{}.txt'.format(catNum, catNum), sep="\t")
    df_category_id = pd.read_csv('../../data_xview/{}_cls/categories_id_label_children_color_{}_classes.txt'.format(catNum, catNum), sep="\t")

    for i in range(df_category_id.shape[0]):
        # category_id = df_category_id['category_id'].iloc[i]
        # category = df_category_id['category'].iloc[i]
        # category_label = df_category_id['category_label'].iloc[i]
        # super_id = df_category_id['super_category_id'].iloc[i]
        # super_label = df_category_id['super_category_label'].iloc[i]
        # super_category = df_category_id['super_category'].iloc[i]
        #
        # category_info = {
        #     'id': category_id,
        #     'cat_label': category_label,
        #     'name': category,
        #     'super_id': super_id,
        #     'super_label': super_label,
        #     'super_category': super_category
        # }

        category_id = df_category_id['category_id'].iloc[i]
        category = df_category_id['category'].iloc[i]
        category_label = df_category_id['category_label'].iloc[i]
        child_cat = df_category_id['children_category'].iloc[i]

        category_info = {
            'id': category_id,
            'cat_label': category_label,
            'name': category,
            'children_category': child_cat
        }
        categories.append(category_info)
    return categories


def boxes_from_coords(coords):
    """
    Processes a coordinate array from a geojson into (xmin,ymin,xmax,ymax) format

    Args:
        coords: an array of bounding box coordinates

    Output:
        Returns an array of shape (N,4) with coordinates in proper format
    """
    nc = np.zeros((coords.shape[0], 4))
    for ind in range(coords.shape[0]):
        x1, x2 = coords[ind, :, 0].min(), coords[ind, :, 0].max()
        y1, y2 = coords[ind, :, 1].min(), coords[ind, :, 1].max()
        nc[ind] = [x1, y1, x2, y2]
    return nc


def chip_image(img, ci_coords, ci_classes, feature_ids, shape=(300, 300), name="", dir=''):
    """
    Chip an image and get relative coordinates and classes.  Bounding boxes that pass into
        multiple chips are clipped: each portion that is in a chip is labeled. For example,
        half a building will be labeled if it is cut off in a chip. If there are no boxes,
        the boxes array will be [[0,0,0,0]] and classes [0].
        Note: This chip_image method is only tested on xView data-- there are some image manipulations that can mess up different images.

    Args:
        img: the image to be chipped in array format
        ci_coords: an (N,4) array of bounding box coordinates for that image
        ci_classes: an (N,1) array of classes for each bounding box
        shape: an (W,H) tuple indicating width and height of chips

    Output:
        An image array of shape (M,W,H,C), where M is the number of chips,
        W and H are the dimensions of the image, and C is the number of color
        channels.  Also returns boxes and classes dictionaries for each corresponding chip.
        :param ci_classes:
        :param ci_coords:
        :param img:
        :param dir:
        :param name:
        :param shape:
        :param feature_ids:
    """

    height, width, _ = img.shape
    wn, hn = shape

    # w_num, h_num = (int(width / wn), int(height / hn))
    w_num, h_num = (round(width / wn), round(height / hn))
    images = {}
    image_names = {}
    total_boxes = {}
    total_classes = {}
    total_box_ids = {}
    #FIXME --YANG
    # if shape[0] == 300:
    #     thr = 0.3
    # else:
    #     thr = 0.1
    thr = 0.3

    k = 0
    for i in range(w_num):
        for j in range(h_num):
            hmin = hn * j
            hmax = hn * (j + 1)
            wmin = wn * i
            wmax = wn * (i + 1)

            x = np.logical_or(np.logical_and((ci_coords[:, 0] < wmax), (ci_coords[:, 0] >= wmin)),
                              np.logical_and((ci_coords[:, 2] < wmax), (ci_coords[:, 2] >= wmin)))
            out = ci_coords[x]
            y = np.logical_or(np.logical_and((out[:, 1] < hmax), (out[:, 1] >= hmin)),
                              np.logical_and((out[:, 3] < hmax), (out[:, 3] >= hmin)))
            out_drop = out[y]
            if out_drop.shape[0] == 0:
                continue

            chip = np.zeros_like(img)
            if hmax >= height and wmax >= width:
                chip[:height-hmin, :width-wmin, :3] = img[hmin : height, wmin : width, :3]
            elif hmax >= height and wmax < width:
                chip[:height-hmin, :wmax-wmin, :3] = img[hmin : height, wmin : wmax, :3]
            elif hmax < height and wmax >= width:
                chip[:hmax-hmin, :width-wmin, :3] = img[hmin : hmax, wmin : width, :3]
            else:
                chip = img[hmin : hmax, wmin : wmax, :3]
            # bounding boxes partially overlapped with a chip
            # were cropped at the chip edge
            out = np.transpose(np.vstack((np.clip(out_drop[:, 0] - (wn * i), 0, wn),
                                          np.clip(out_drop[:, 1] - (hn * j), 0, hn),
                                          np.clip(out_drop[:, 2] - (wn * i), 0, wn),
                                          np.clip(out_drop[:, 3] - (hn * j), 0, hn))))
            # s_height = None
            # s_width = None
            # if hmax >= height and wmax >= width:
            #     chip = img[height-hn : height, width-wn : width, :3]
            #     s_height = (height-hn)
            #     s_width = (width-wn)
            # elif hmax >= height and wmax < width:
            #     chip = img[height-hn : height, wmin : wmax, :3]
            #     s_height = (height-hn)
            # elif hmax < height and wmax >= width:
            #     chip = img[hmin : hmax, width-wn : width, :3]
            #     s_width = (width-wn)
            # else:
            #     chip = img[hmin : hmax, wmin : wmax, :3]
            #
            # if not s_height and not s_width:
            #     out = np.transpose(np.vstack((np.clip(out_drop[:, 0] - wmin, 0, wn),
            #                                   np.clip(out_drop[:, 1] - hmin, 0, hn),
            #                                   np.clip(out_drop[:, 2] - wmin, 0, wn),
            #                                   np.clip(out_drop[:, 3] - hmin, 0, hn))))
            # elif s_height and s_width:
            #         out = np.transpose(np.vstack((np.clip(out_drop[:, 0] - s_width, 0, wn),
            #                                       np.clip(out_drop[:, 1] - s_height, 0, hn),
            #                                       np.clip(out_drop[:, 2] - s_width, 0, wn),
            #                                       np.clip(out_drop[:, 3] - s_height, 0, hn))))
            # elif s_height and not s_width:
            #     out = np.transpose(np.vstack((np.clip(out_drop[:, 0] - wmin, 0, wn),
            #                                   np.clip(out_drop[:, 1] - s_height, 0, hn),
            #                                   np.clip(out_drop[:, 2] - wmin, 0, wn),
            #                                   np.clip(out_drop[:, 3] - s_height, 0, hn))))
            # elif not s_height and s_width: # s_width
            #     out = np.transpose(np.vstack((np.clip(out_drop[:, 0] - s_width, 0, wn),
            #                                   np.clip(out_drop[:, 1] - hmin, 0, hn),
            #                                   np.clip(out_drop[:, 2] - s_width, 0, wn),
            #                                   np.clip(out_drop[:, 3] - hmin, 0, hn))))

            print('out shape', out.shape)
            if out.shape[0] == 0:
                continue
            total_boxes[k] = out
            total_classes[k] = ci_classes[x][y]
            total_box_ids[k] = feature_ids[x][y]

            # remove figures with more than 10% black pixels
            im = Image.fromarray(chip)
            im_gray = np.array(im.convert('L'))
            non_black_percent = np.count_nonzero(im_gray) / (im_gray.shape[0] * im_gray.shape[1])

            # NOTE: when the image is full of black pixel  or larger than (1-thr) covered with black pixel
            if non_black_percent < thr:
                continue
            image_name = name + '_' + str(k) + '.jpg'
            im.save(os.path.join(dir, image_name))
            image_names[k] = image_name
            images[k] = chip

            k += 1

    return images, image_names, total_boxes, total_classes, total_box_ids

    '''
    crop with 50% overlap, after no overlap cropping above, then crop with no overlap containing all the crop edges above with the same step 
    '''
    # w_st = np.int(0.5 * shape[1])
    # w_ed = width - wn
    # h_st = np.int(0.5 * shape[0])
    # h_ed = height - hn
    # for i in np.arange(w_st, w_ed, wn):
    #     for j in np.arange(h_st, h_ed, hn):
    #
    #         chip = img[j:(j + hn), i: (i + wn), :3]
    #         im = Image.fromarray(chip)
    #         im_gray = np.array(im.convert('L'))
    #         non_black_percent = np.count_nonzero(im_gray) / (im_gray.shape[0] * im_gray.shape[1])
    #         if non_black_percent < 0.9:
    #             continue
    #
    #         x = np.logical_or(np.logical_and(coords[:, 0] < (i + wn), (coords[:, 0] > i)),
    #                           np.logical_and(coords[:, 2] < (i + wn), (coords[:, 2] > i)))
    #         out = coords[x]
    #         y = np.logical_or(np.logical_and(out[:, 1] < (j + hn), out[:, 1] > j),
    #                           np.logical_and(out[:, 3] < (j + hn), out[:, 3] > j))
    #         # outn = out[y]
    #         # out = np.transpose(np.vstack((np.clip(outn[:, 0] - i, 0, wn),
    #         #                               np.clip(outn[:, 1] - j, 0, hn),
    #         #                               np.clip(outn[:, 2] - i, 0, wn),
    #         #                               np.clip(outn[:, 3] - j, 0, hn))))
    #         # out = np.transpose(np.vstack((outn[:, 0], outn[:, 1], outn[:, 2], outn[:, 3])))
    #         # out = outn
    #
    #         out = out[y]
    #         box_classes = classes[x][y]
    #         box_ids = feature_ids[x][y]
    #
    #         if out.shape[0] == 0:
    #             continue
    #         total_boxes[k] = out
    #         total_classes[k] = box_classes
    #         total_box_ids[k] = box_ids
    #
    #         image_name = name + '_' + str(k) + '.jpg'
    #         im.save(os.path.join(dir, image_name))
    #         image_names[k] = image_name
    #         images[k] = chip
    #
    #         k += 1
    # print(len(images), images[0].shape)
    # images = np.array(images).astype(np.uint8)
    # return images, image_names, total_boxes, total_classes, total_box_ids


def max_min_h_w_xview():
    trn_dir = '/media/lab/Yang/data/xView/train_images'
    val_dir = '/media/lab/Yang/data/xView/val_images'

    trn_list = os.listdir(trn_dir)
    val_list = os.listdir(val_dir)

    trn_list = [f if '_' not in f else None for f in trn_list]
    trn_list = list(filter(None, trn_list))

    num_trn = len(trn_list)
    num_val = len(val_list)
    trn_hw_sizes = np.zeros((num_trn, 2))
    val_hw_sizes = np.zeros((num_val, 2))

    for t in range(num_trn):
        arr = get_image(os.path.join(trn_dir, trn_list[t]))
        trn_hw_sizes[t, 0] = arr.shape[0]
        trn_hw_sizes[t, 1] = arr.shape[1]

    for v in range(num_val):
        arr = get_image(os.path.join(val_dir, val_list[v]))
        val_hw_sizes[v, 0] = arr.shape[0]
        val_hw_sizes[v, 1] = arr.shape[1]

    trn_mxH, trn_mxW = np.max(trn_hw_sizes, axis=0)
    trn_minH, trn_minW = np.min(trn_hw_sizes, axis=0)

    print("trn_wh_size: max Height: %1f, max Width: %1f \n \t \t min Height: %1f, min Width: %1f " % (
        trn_mxH, trn_mxW, trn_minH, trn_minW))  # 1925.tif

    val_mxH, val_mxW = np.max(val_hw_sizes, axis=0)
    val_minH, val_minW = np.min(val_hw_sizes, axis=0)

    print("val_wh_size: max Height: %1f, max Width: %1f \n \t \t min Height: %1f, min Width: %1f " % (
        val_mxH, val_mxW, val_minH, val_minW))


if __name__ == '__main__':

    '''
    check how many images in the xView_train.geojson
    '''
    import json
    # js = json.load(open('/media/lab/Yang/data/xView/xView_train.geojson'))
    # coords, chips, classes, features_ids = get_labels('/media/lab/Yang/data/xView/xView_train.geojson', 60) # 62
    '''
    find the images not in the train folder but in the geojson
    '''
    # annos = js['features']
    # images = []
    # for an in annos:
    #     images.append(an['properties']['image_id'])
    #
    # images = list(set(images))
    # print(len(images))
    #
    # from glob import glob
    # img_list = glob('/media/lab/Yang/data/xView/train_images/' + '*.tif')
    # imgs = [os.path.basename(a) for a in img_list]
    # excu = [i for i in images if i not in imgs]
    #
    # #output: 1395.tif

    '''
    find the images containing class 75 and class 82
    '''
    #
    # c75_imgs = []
    # c82_imgs = []
    # for a in annos:
    #     if a['properties']['type_id'] == 75:
    #         c75_imgs.append(a['properties']['image_id'])
    #     elif a['properties']['type_id'] == 82:
    #         c82_imgs.append(a['properties']['image_id'])
    #     else:
    #         continue
    # c75_imgs = list(set(c75_imgs))
    # c82_imgs = list(set(c82_imgs))
    #
    # print(len(c75_imgs), len(c82_imgs))


    '''
    chip one certain image
    '''

    # catNum = 62 # 60
    # categories = get_all_categories(catNum)
    # max_min_h_w_xview()

    # save_dir = '/media/lab/Yang/data/xView_COCO/chip_imgs/'
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # chip_name = '100.tif'
    # arr = get_image('/media/lab/Yang/data/xView/train_images/' + chip_name)
    # coords, chips, classes, features_ids = get_labels(
    #     '/media/lab/Yang/data/xView/xView_train.geojson')
    # coords = coords[chips == chip_name]
    # classes = classes[chips == chip_name].astype(np.int64)
    # features_ids = features_ids[chips == chip_name]
    # labels = {}

    # with open('xview_class_labels.txt') as f:
    #     for row in csv.reader(f):
    #         labels[int(row[0].split(":")[0])] = row[0].split(":")[1]
    # print([labels[i] for i in np.unique(classes)])
    # ims, img_names, box, classes_final, box_ids = chip_image(img=arr, ci_coords=coords, ci_classes=classes,
    #                                                          feature_ids=features_ids, shape=(300, 300), dir=save_dir)

