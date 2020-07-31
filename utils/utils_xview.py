import glob
import math
import os
import random
import shutil
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import pandas as pd

import models_xview
from utils import torch_utils # , google_utils
import json

matplotlib.rc('font', **{'size': 11})

# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


def floatn(x, n=3):  # format floats to n decimals
    return float(format(x, '.%gf' % n))


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


# Built-in
import os
import time
import json
import pickle
import collections.abc
# from glob import glob
from functools import wraps

# Libs
import torch
import scipy.signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def labels_to_class_weights(labels, nc=60):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    print(min(classes), max(classes))
    weights = np.bincount(classes, minlength=nc)  # occurences per class

    # Prepend gridpoint count (for uCE trianing)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=60, class_weights=np.ones(60)):
    # Produces image weights based on class mAPs
    n = len(labels)
    class_counts = np.array([np.bincount(labels[i][:, 0].astype(np.int), minlength=nc) for i in range(n)])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco_class_weights():  # frequency of each class in coco train2014
    n = [187437, 4955, 30920, 6033, 3838, 4332, 3160, 7051, 7677, 9167, 1316, 1372, 833, 6757, 7355, 3302, 3776, 4671,
         6769, 5706, 3908, 903, 3686, 3596, 6200, 7920, 8779, 4505, 4272, 1862, 4698, 1962, 4403, 6659, 2402, 2689,
         4012, 4175, 3411, 17048, 5637, 14553, 3923, 5539, 4289, 10084, 7018, 4314, 3099, 4638, 4939, 5543, 2038, 4004,
         5053, 4578, 27292, 4113, 5931, 2905, 11174, 2873, 4036, 3415, 1517, 4122, 1980, 4464, 1190, 2302, 156, 3933,
         1877, 17630, 4337, 4624, 1075, 3468, 135, 1380]
    weights = 1 / torch.Tensor(n)
    weights /= weights.sum()
    # with open('data/coco.names', 'r') as f:
    #     for k, v in zip(f.read().splitlines(), n):
    #         print('%20s: %g' % (k, v))
    return weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xview_classes2indices(classes, class_num=60):  # remap xview classes 11-94 to 0-61
    # FIXME
    # indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    #            29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    #            55, 56, 57, 58, 59]
    # labels = [74, 45, 73, 19, 24, 35, 65, 79, 55, 51, 32, 76, 60, 53, 64, 77, 49, 47, 11, 36, 63, 66, 61, 15, 84, 71,
    #            38, 40, 59, 41, 52, 34, 17, 13, 20, 93, 33, 56, 42, 62, 72, 91, 89, 12, 18, 86, 57, 37, 94, 54, 27, 23,
    #            26, 25, 28, 29, 44, 21, 83, 50]
    df_cat = pd.read_csv('cfg/categories_id_color_diverse_{}.txt'.format(class_num), delimiter='\t')
    cat_labels = list(df_cat['category_label'])
    return [cat_labels.index(c) for c in classes]


def xview_indices2classes(indices, class_num=60):  # remap xview classes 11-94 to 0-61
    # FIXME -- yang
    # df_cat = pd.read_csv('/media/lab/Yang/code/xview-yolov3-changed/cfg/categories_id_color_diverse_60.txt', delimiter='\t')
    df_cat = pd.read_csv('cfg/categories_id_color_diverse_{}.txt'.format(class_num), delimiter='\t')
    cat_labels = list(df_cat['category_label'])
    return [cat_labels[i] for i in indices]


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.03)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.03)
        torch.nn.init.constant_(m.bias.data, 0.0)


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


# def xywh2xyxy(box):
#     # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
#     if isinstance(box, torch.Tensor):
#         x, y, w, h = box.t()
#         return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).t()
#     else:  # numpy
#         x, y, w, h = box.T
#         return np.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).T
#
#
# def xyxy2xywh(box):
#     # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h]
#     if isinstance(box, torch.Tensor):
#         x1, y1, x2, y2 = box.t()
#         return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)).t()
#     else:  # numpy
#         x1, y1, x2, y2 = box.T
#         return np.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)).T


#fixme
def iou_metric(truth, pred, divide=False):
    """
    Compute IoU, i.e., jaccard index
    :param truth: truth data matrix, should be H*W
    :param pred: prediction data matrix, should be the same dimension as the truth data matrix
    :param divide: if True, will return the IoU, otherwise return the numerator and denominator
    :return:
    """
    truth = truth.flatten()
    pred = pred.flatten()
    intersect = truth*pred
    if not divide:
        return float(np.sum(intersect == 1)), float(np.sum(truth+pred >= 1))
    else:
        return float(np.sum(intersect == 1) / np.sum(truth+pred >= 1))


def coord_iou(coords_a, coords_b):
    """
    This code comes from https://stackoverflow.com/a/42874377
    :param coords_a: [xtl, ytl, xbr, ybr]
    :param coords_b: [xtl, ytl, xbr, ybr]
    :return:
    """

    # y1, x1 = np.min(coords_a, axis=0)
    # y2, x2 = np.max(coords_a, axis=0)
    # # bb1 = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
    # y1, x1 = np.min(coords_b, axis=0)
    # y2, x2 = np.max(coords_b, axis=0)
    # bb2 = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
    # print('coords_a {} {}'.format(coords_a[0], coords_a[2]))
    assert coords_a[0] <= coords_a[2]
    assert coords_a[1] <= coords_a[3]
    assert coords_b[0] <= coords_b[2]
    assert coords_b[1] <= coords_b[3]

    x_left = max(coords_a[0], coords_b[0])
    y_top = max(coords_a[1], coords_b[1])
    x_right = min(coords_a[2], coords_b[2])
    y_bottom = min(coords_a[3], coords_b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (coords_a[2] - coords_a[0]) * (coords_a[3] - coords_a[1])
    bb2_area = (coords_b[2] - coords_b[0]) * (coords_b[3] - coords_b[1])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    # if iou>0.5:
    #     print(x_left, y_top, x_right, y_bottom)
    # if not iou:
    #     print('bb1_area', bb1_area)
    #     print('bb2_area', bb2_area)
    #     print('intersection_area', intersection_area)
    assert 0.0 <= iou <= 1.0, print(iou)
    return iou


def compute_iou(coords_a, coords_b, size):
    """
    Compute object-wise IoU
    :param self:
    :param coords_a:
    :param coords_b:
    :param size:
    :return:
    """
    # compute bbox IoU since this is faster
    iou = coord_iou(coords_a, coords_b)
    if iou > 0:
        # if bboxes overlaps, compute object-wise IoU
        tile_a = np.zeros(size)
        tile_a[coords_a[:, 0], coords_a[:, 1]] = 1
        tile_b = np.zeros(size)
        tile_b[coords_b[:, 0], coords_b[:, 1]] = 1
        return iou_metric(tile_a, tile_b, divide=True)
    else:
        return 0


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=img_shape[1])  # clip x
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=img_shape[0])  # clip y


def ap_per_class(tp, conf, pred_cls, target_cls, pr_path='', pr_name='', model_id=None):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    s = [len(unique_classes), tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        if model_id is not None:
            i = pred_cls ==0
            n_gt = (target_cls == model_id).sum()
            n_p = i.sum()
        else:
            i = pred_cls == c
            n_gt = (target_cls == c).sum()  # Number of ground truth objects
            n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:

            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)
            # print('fpc', fpc.shape) # (178, 1)  , np.concatenate(([0.], recall).shape)
            # print('tp', tp.shape) #  , np.concatenate(([0.], recall).shape)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = recall[-1]

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = precision[-1]

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

            # Plot
            if pr_path:
                fig1, ax1= plt.subplots(1, figsize=(10, 8))
                # print('r', recall.shape) #  , np.concatenate(([0.], recall).shape)
                # print('p', precision.shape) # , np.concatenate(([0.], precision).shape)
                # ax.plot(np.concatenate(([0.], recall)), np.concatenate(([0.], precision)))
                np.savetxt(os.path.join(pr_path, 'recall.txt'), recall)
                np.savetxt(os.path.join(pr_path, 'precision.txt'), precision)
                ax1.plot(recall, precision, label=pr_name + '  mAP: %.3f' % ap[ci])
                ax1.legend()
                ax1.set_title('YOLOv3-SPP PR-Curve')
                ax1.set_xlabel('Recall')
                ax1.set_ylabel('Precision')
                ax1.set_xlim(0, 1)
                ax1.set_ylim(0, 1)
                ax1.grid()
                fig1.savefig(os.path.join(pr_path, pr_name + '_PR_curve.png'), dpi=300)


    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    # print('r', r.shape) #  (1,1)
    return p, r, ap, f1, unique_classes.astype('int32')

def plot_roc(tp, conf, pred_cls, target_cls, pr_path='', pr_name='', model_id=None, area=0):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
        model_id: the specified id
        area: all image covered area (square kilometers)
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # Sort by objectness
    i = np.argsort(-conf) # 175
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    n_gt = len(target_cls==model_id)
    far_list = [0]
    rec_list = [0]
    n_t = 0
    n_f = 0
    for ix, t in enumerate(conf):
        if tp[ix]: # and conf[ix] >= t:
            n_t += 1
        elif tp[ix]==0: # and conf[ix]>=t:
            n_f += 1
        far = n_f/area
        rec = n_t/n_gt
        far_list.append(far)
        rec_list.append(rec)
    np.savetxt(os.path.join(pr_path, 'far_list.txt'), far_list)
    np.savetxt(os.path.join(pr_path, 'rec_list.txt'), rec_list)
    rec_arr = np.array(rec_list)
    far_arr = np.array(far_list)
    fx = np.where(far_arr[1:] != far_arr[:-1])[0]
    auc = np.sum((far_arr[fx + 1] - far_arr[fx]) * rec_arr[fx + 1])
    if model_id is not None:
        title = 'YOLOv3-SPP ROC of $T_{%s}$' % 'xview\_model{}'.format(model_id)
    else:
        title = 'YOLOv3-SPP ROC of $T_{xview}$ '
    if pr_path:
        font_title = {'family': 'serif', 'weight': 'normal', 'size': 15}
        font_label = {'family': 'serif', 'weight': 'normal', 'size': 12}

        fig2, ax2= plt.subplots(1, figsize=(10, 8))
        ax2.plot(far_list, rec_list, label=pr_name + '  AUC: %.3f'%(auc))
        ax2.legend()
        ax2.set_title(title, font_title)
        ax2.set_xlabel('FAR', font_label)
        ax2.set_ylabel('Recall', font_label)
        ax2.grid()
        fig2.savefig(os.path.join(pr_path, pr_name + '_ROC_curve.png'), dpi=300)
        plt.close(fig2)


    # Find unique classes
    # unique_classes = np.unique(target_cls)
    # # for c in unique_classes:
    # #     gt_p = (target_cls == c).sum()
    # #     gt_n = len(target_cls) - gt_p
    # #     for ic in range(len(conf)):
    # #         temp_tp = tp[:ic+1]
    # #         tpc = np.sum(temp_tp)
    # #         fpc = np.sum(temp_tp == 0)
    # #         recall = tpc / gt_p
    # #         precision = tpc / (tpc + fpc)
    #         # fpr = fpc / gt_n
    #
    # # Create Precision-Recall curve and compute AP for each class
    # s = [1, tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    # ap, p, r, auc = np.zeros(s), np.zeros(s), np.zeros(s), np.zeros(s)
    # # if syn_trn:
    # #     gt_p = (target_cls == model_id).sum()
    # #     prd_p = len(pred_cls)  # Number of predicted objects
    # # else:
    # gt_p = (target_cls == unique_classes[0]).sum()
    # prd_p = len(pred_cls)  # Number of predicted objects
    #
    # # print('gt_p', gt_p) # 19
    # # print('prd_p', prd_p) # 184
    # # exit(0)
    # rec_list = []
    # prec_list = []
    # far_list = []
    # fpc_list = []
    # ci = 0
    # for ix in range(len(conf)):
    #     # Accumulate FPs and TPs
    #     tix = tp[:ix+1, 0]
    #     tpc = np.sum(tix==1)
    #     fpc = np.sum(tix==0)
    #
    #     # print('fpc', fpc.shape) # (184, 1) (178, 1)  , np.concatenate(([0.], recall).shape)
    #     # print('tp', tp.shape) #  (184, 1), np.concatenate(([0.], recall).shape)
    #
    #     # Recall
    #     recall = tpc / gt_p  # recall curve
    #     rec_list.append(recall)
    #
    #     # Precision
    #     precision = tpc / (tpc + fpc) # (tpc + fpc)  # precision curve
    #     prec_list.append(precision)
    #
    #     fpc_list.append(fpc)
    #
    #     fpa = fpc / area  # roc curve fpr = fpc/(fpc + tnc)
    #     far_list.append(fpa)
    #
    # r = recall
    # p = precision
    # fx = np.where(far_list[1:] != far_list[:-1])[0]
    # far_arr = np.array(far_list)
    # rec_arr = np.array(rec_list)
    # np.savetxt(os.path.join(pr_path, 'fpc.txt'), fpc_list)
    # auc[ci] = np.sum((far_arr[fx + 1] - far_arr[fx]) * rec_arr[fx + 1])
    # # AP from recall-precision curve
    # ap[ci, 0] = compute_ap(rec_list, prec_list)
    #
    # if pr_path:
    #     fig1, ax1= plt.subplots(1, figsize=(10, 8))
    #     font_title = {'family': 'serif', 'weight': 'normal', 'size': 15}
    #     font_label = {'family': 'serif', 'weight': 'normal', 'size': 12}
    #     # print('r', recall.shape) #  , np.concatenate(([0.], recall).shape)
    #     # print('p', precision.shape) # , np.concatenate(([0.], precision).shape)
    #     # ax.plot(np.concatenate(([0.], recall)), np.concatenate(([0.], precision)))
    #     np.savetxt(os.path.join(pr_path, 'recall.txt'), rec_list)
    #     np.savetxt(os.path.join(pr_path, 'precision.txt'), prec_list)
    #     ax1.plot(rec_list, prec_list, label=pr_name + '  mAP: %.3f'%(ap[ci]))
    #     ax1.legend()
    #     ax1.set_title('YOLOv3-SPP PR-Curve', font_title)
    #     ax1.set_xlabel('Recall', font_label)
    #     ax1.set_ylabel('Precision', font_label)
    #     ax1.set_xlim(0, 1)
    #     ax1.grid()
    #     fig1.savefig(os.path.join(pr_path, pr_name + '_PR_curve.png'), dpi=300)
    #     plt.close(fig1)
    #
    #     fig2, ax2= plt.subplots(1, figsize=(10, 8))
    #     np.savetxt(os.path.join(pr_path, 'far.txt'), far_list)
    #     ax2.plot(far_list, rec_list, label=pr_name + '  AUC: %.3f'%(auc[ci]))
    #     ax2.legend()
    #     ax2.set_title('YOLOv3-SPP ROC', font_title)
    #     ax2.set_xlabel('FP', font_label)
    #     ax2.set_ylabel('Recall', font_label)
    #     ax2.grid()
    #     fig2.savefig(os.path.join(pr_path, pr_name + '_ROC_curve.png'), dpi=300)
    #     plt.close(fig2)
    # Compute F1 score (harmonic mean of precision and recall)
    # f1 = 2 * p * r / (p + r + 1e-16)
    # # print('r', r.shape) #  (1,1)
    # return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def box_iou(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.t())
    area2 = box_area(boxes2.t())

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = (rb - lt).clamp(min=0).prod(2)  # [N,M]
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn() https://arxiv.org/pdf/1708.02002.pdf
    # i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=2.5)
    def __init__(self, loss_fcn, gamma=0.5, alpha=1, reduction='mean'):
        super(FocalLoss, self).__init__()
        loss_fcn.reduction = 'none'  # required to apply FL to each element
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        loss = self.loss_fcn(input, target)
        loss *= self.alpha * (1.000001 - torch.exp(-loss)) ** self.gamma  # non-zero power for gradient stability

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def compute_loss(p, targets, model):  # predictions, targets, model
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
    tcls, tbox, indices, anchor_vec = build_targets(model, targets)

    h = model.hyp  # hyperparameters
    arc = model.arc  # # (default, uCE, uBCE) detection architectures
    red = 'sum'  # Loss reduction (sum or mean)

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]), reduction=red)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']]), reduction=red)
    BCE = nn.BCEWithLogitsLoss(reduction=red)
    CE = nn.CrossEntropyLoss(reduction=red)  # weight=model.class_weights

    if 'F' in arc:  # add focal loss
        g = h['fl_gamma']
        BCEcls, BCEobj, BCE, CE = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g), FocalLoss(BCE, g), FocalLoss(CE, g)

    # Compute losses
    np, ng = 0, 0  # number grid points, targets
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # target image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0])  # target obj
        # print('pi shape ', pi.shape) # pi shape  torch.Size([8, 3, 19, 19, 6])
        # print('tobj size ', tobj.shape) # torch.Size([8, 3, 19, 19]) torch.Size([8, 3, 38, 38])  torch.Size([8, 3, 76, 76])
        np += tobj.numel()

        # Compute losses
        nb = len(b)
        #fixme NOTE only compute loss for non empty labels
        if nb:  # number of targets
            ng += nb
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            # ps[:, 2:4] = torch.sigmoid(ps[:, 2:4])  # wh power loss (uncomment)

            # GIoU
            pxy = torch.sigmoid(ps[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            pwh = torch.exp(ps[:, 2:4]).clamp(max=1E3) * anchor_vec[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            #fixme
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou computation
            # print(giou)
            # exit(0)
            # print('tbox[i].size----', tbox[i].size)
            # print('tbox[i]----', tbox[i])
            #fixme
            # if len(tbox[i].size()):
            #     giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou computation
            # else:
            #     giou = torch.zeros_like(tbox[i])
            #     print('giou----', giou)
            lbox += (1.0 - giou).sum() if red == 'sum' else (1.0 - giou).mean()  # giou loss
            # print('b, a, gj, gi', b, a, gj, gi)
            tobj[b, a, gj, gi] = giou.detach().type(tobj.dtype)

            if 'default' in arc and model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.zeros_like(ps[:, 5:])  # targets
                t[range(nb), tcls[i]] = 1.0
                lcls += BCEcls(ps[:, 5:], t)  # BCE
                # lcls += CE(ps[:, 5:], tcls[i])  # CE

                # Instance-class weighting (use with reduction='none')
                # nt = t.sum(0) + 1  # number of targets per class
                # lcls += (BCEcls(ps[:, 5:], t) / nt).mean() * nt.mean()  # v1
                # lcls += (BCEcls(ps[:, 5:], t) / nt[tcls[i]].view(-1,1)).mean() * nt.mean()  # v2

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        #fixme lobj is also for non empty lables
        if 'default' in arc:  # separate obj and cls
            # print('tobj---', tobj.shape)
            # print('pi---', pi[..., 4].shape)
            lobj += BCEobj(pi[..., 4], tobj)  # obj loss
            # print('lobj---', lobj)

        elif 'BCE' in arc:  # unified BCE (80 classes)
            t = torch.zeros_like(pi[..., 5:])  # targets
            if nb:
                t[b, a, gj, gi, tcls[i]] = 1.0
            lobj += BCE(pi[..., 5:], t)

        elif 'CE' in arc:  # unified CE (1 background + 80 classes)
            t = torch.zeros_like(pi[..., 0], dtype=torch.long)  # targets
            if nb:
                t[b, a, gj, gi] = tcls[i] + 1
            lcls += CE(pi[..., 4:].view(-1, model.nc + 1), t.view(-1))

    lbox *= h['giou']
    lobj *= h['obj']
    lcls *= h['cls']
    #fixme
    # if red == 'sum' and ng > 0:
    #     bs = tobj.shape[0]  # batch size
    #     lbox *= 3 / ng
    #     lobj *= 3 / (6300 * bs) * 2  # 3 / np * 2 # ?????
    #     lcls *= 3 / ng / model.nc
    if red == 'sum':

        bs = tobj.shape[0]  # batch size 8

        lobj *= 3 / (6300 * bs) * 2  # 3 / np * 2 # 22743 https://github.com/ultralytics/yolov3/issues/804 608/32x608/32=19x19 grid by 3 anchors for the first layer, 38x38 grid with 3 anchors for second, 76x76 grid with 3 anchors for third
        if ng:
            lcls *= 3 / ng / model.nc
            lbox *= 3 / ng

    loss = lbox + lobj + lcls
    return loss, torch.cat((lbox, lobj, lcls, loss)).detach()


def build_targets(model, targets):
    '''
    :param model:
    :param targets:
    :return: tcls, tbox, indices, anchor_vec
    '''
    # targets = [image, class, x, y, w, h]

    nt = len(targets)
    tcls, tbox, indices, av = [], [], [], []
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    # print('multi gpu ', multi_gpu)
    reject, use_all_anchors = True, True
    for i in model.yolo_layers:
        # get number of grid points and anchor vec for this yolo layer
        # print(type(model)) # <class 'models_xview.DataParallelPassThrough'>
        if multi_gpu:
            # print(type(model.module.module_list[i])) # <class 'models_xview.YOLOLayer'>
            ng, anchor_vec = model.module.module_list[i].ng, model.module.module_list[i].anchor_vec
        else:
            ng, anchor_vec = model.module_list[i].ng, model.module_list[i].anchor_vec

        # iou of targets-anchors
        t, a = targets, []
        gwh = t[:, 4:6] * ng
        if nt:
            iou = wh_iou(anchor_vec, gwh)

            if use_all_anchors:
                na = len(anchor_vec)  # number of anchors
                a = torch.arange(na).view((-1, 1)).repeat([1, nt]).view(-1)
                t = targets.repeat([na, 1])
                gwh = gwh.repeat([na, 1])
            else:  # use best anchor only
                iou, a = iou.max(0)  # best iou and anchor

            # reject anchors below iou_thres (OPTIONAL, increases P, lowers R)
            if reject:
                j = iou.view(-1) > model.hyp['iou_t']  # iou threshold hyperparameter
                t, a, gwh = t[j], a[j], gwh[j]

        # Indices
        b, c = t[:, :2].long().t()  # target image, class
        gxy = t[:, 2:4] * ng  # grid x, y
        gi, gj = gxy.long().t()  # grid x, y indices
        indices.append((b, a, gj, gi))

        # Box
        gxy -= gxy.floor()  # xy
        tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)
        av.append(anchor_vec[a])  # anchor vec

        # Class
        tcls.append(c)
        if c.shape[0]:  # if any targets
            assert c.max() < model.nc, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
                                       'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
                                           model.nc, model.nc - 1, c.max())

    return tcls, tbox, indices, av


def non_max_suppression(prediction, conf_thres=0.5, iou_thres=0.5, multi_cls=True, method='vision_batch', classes=None):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, conf, class)
    """
    # NMS methods https://github.com/ultralytics/yolov3/issues/679 'or', 'and', 'merge', 'vision', 'vision_batch'

    # Box constraints
    #fixme
    # min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 4, 608

    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        # Apply conf constraint
        pred = pred[pred[:, 4] > conf_thres]

        # Apply width-height constraint
        pred = pred[(pred[:, 2:4] > min_wh).all(1) & (pred[:, 2:4] < max_wh).all(1)]

        # If none remain process next image
        if len(pred) == 0:
            continue

        # Compute conf
        torch.sigmoid_(pred[..., 5:])
        pred[..., 5:] *= pred[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(pred[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_cls:
            i, j = (pred[:, 5:] > conf_thres).nonzero().t()
            pred = torch.cat((box[i], pred[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # best class only
            conf, j = pred[:, 5:].max(1)
            pred = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)

        # Filter by class
        if classes:
            pred = pred[(j.view(-1, 1) == torch.Tensor(classes)).any(1)]

        # Apply finite constraint
        if not torch.isfinite(pred).all():
            pred = pred[torch.isfinite(pred).all(1)]

        # Batched NMS
        if method == 'vision_batch':
            output[image_i] = pred[torchvision.ops.boxes.batched_nms(pred[:, :4], pred[:, 4], pred[:, 5], iou_thres)]
            continue

        # Sort by confidence
        if not method.startswith('vision'):
            pred = pred[pred[:, 4].argsort(descending=True)]

        # All other NMS methods
        det_max = []
        cls = pred[:, -1]
        for c in cls.unique():
            dc = pred[cls == c]  # select class c
            n = len(dc)
            if n == 1:
                det_max.append(dc)  # No NMS required if only 1 prediction
                continue
            elif n > 500:
                dc = dc[:500]  # limit to first 500 boxes: https://github.com/ultralytics/yolov3/issues/117

            if method == 'vision':
                det_max.append(dc[torchvision.ops.boxes.nms(dc[:, :4], dc[:, 4], iou_thres)])

            elif method == 'or':  # default
                # METHOD1
                # ind = list(range(len(dc)))
                # while len(ind):
                # j = ind[0]
                # det_max.append(dc[j:j + 1])  # save highest conf detection
                # reject = (bbox_iou(dc[j], dc[ind]) > iou_thres).nonzero()
                # [ind.pop(i) for i in reversed(reject)]

                # METHOD2
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < iou_thres]  # remove ious > threshold

            elif method == 'and':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < iou_thres]  # remove ious > threshold

            elif method == 'merge':  # weighted mixture box
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = bbox_iou(dc[0], dc) > iou_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

            elif method == 'soft':  # soft-NMS https://arxiv.org/abs/1704.04503
                sigma = 0.5  # soft-nms sigma parameter
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    det_max.append(dc[:1])
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:]
                    dc[:, 4] *= torch.exp(-iou ** 2 / sigma)  # decay confidences
                    dc = dc[dc[:, 4] > conf_thres]  # https://github.com/ultralytics/yolov3/issues/362

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort

    return output


def get_yolo_layers(model):
    bool_vec = [x['type'] == 'yolo' for x in model.module_defs]
    return [i for i, x in enumerate(bool_vec) if x]  # [82, 94, 106] for yolov3


def print_model_biases(model):
    # prints the bias neurons preceding each yolo layer
    print('\nModel Bias Summary (per output layer):')
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    for l in model.yolo_layers:  # print pretrained biases
        if multi_gpu:
            na = model.module.module_list[l].na  # number of anchors
            b = model.module.module_list[l - 1][0].bias.view(na, -1)  # bias 3x85
        else:
            na = model.module_list[l].na
            b = model.module_list[l - 1][0].bias.view(na, -1)  # bias 3x85
        print('regression: %5.2f+/-%-5.2f ' % (b[:, :4].mean(), b[:, :4].std()),
              'objectness: %5.2f+/-%-5.2f ' % (b[:, 4].mean(), b[:, 4].std()),
              'classification: %5.2f+/-%-5.2f' % (b[:, 5:].mean(), b[:, 5:].std()))


def strip_optimizer(f='weights/last.pt'):  # from utils.utils import *; strip_optimizer()
    # Strip optimizer from *.pt files for lighter files (reduced by 2/3 size)
    x = torch.load(f, map_location=torch.device('cpu'))
    x['optimizer'] = None
    # x['training_results'] = None  # uncomment to create a backbone
    # x['epoch'] = -1  # uncomment to create a backbone
    torch.save(x, f)


def create_backbone(f='weights/last.pt'):  # from utils.utils import *; create_backbone()
    # create a backbone from a *.pt file
    x = torch.load(f, map_location=torch.device('cpu'))
    x['optimizer'] = None
    x['training_results'] = None
    x['epoch'] = -1
    for p in x['model'].values():
        try:
            p.requires_grad = True
        except:
            pass
    torch.save(x, 'weights/backbone.pt')


def coco_class_count(path='../coco/labels/train2014/'):
    # Histogram of occurrences per class
    nc = 80  # number classes
    x = np.zeros(nc, dtype='int32')
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        x += np.bincount(labels[:, 0].astype('int32'), minlength=nc)
        print(i, len(files))


def coco_only_people(path='../coco/labels/train2017/'):  # from utils.utils import *; coco_only_people()
    # Find images with only people
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        if all(labels[:, 0] == 0):
            print(labels.shape[0], file)


def select_best_evolve(path='evolve*.txt'):  # from utils.utils import *; select_best_evolve()
    # Find best evolved mutation
    for file in sorted(glob.glob(path)):
        x = np.loadtxt(file, dtype=np.float32, ndmin=2)
        print(file, x[fitness(x).argmax()])


def crop_images_random(path='../images/', scale=0.50):  # from utils.utils import *; crop_images_random()
    # crops images into random squares up to scale fraction
    # WARNING: overwrites images!
    for file in tqdm(sorted(glob.glob('%s/*.*' % path))):
        img = cv2.imread(file)  # BGR
        if img is not None:
            h, w = img.shape[:2]

            # create random mask
            a = 30  # minimum size (pixels)
            mask_h = random.randint(a, int(max(a, h * scale)))  # mask height
            mask_w = mask_h  # mask width

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            cv2.imwrite(file, img[ymin:ymax, xmin:xmax])


def coco_single_class_labels(path='../coco/labels/train2014/', label_class=43):
    # Makes single-class coco datasets. from utils.utils import *; coco_single_class_labels()
    if os.path.exists('new/'):
        shutil.rmtree('new/')  # delete output folder
    os.makedirs('new/')  # make new output folder
    os.makedirs('new/labels/')
    os.makedirs('new/images/')
    for file in tqdm(sorted(glob.glob('%s/*.*' % path))):
        with open(file, 'r') as f:
            labels = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
        i = labels[:, 0] == label_class
        if any(i):
            img_file = file.replace('labels', 'images').replace('txt', 'jpg')
            labels[:, 0] = 0  # reset class to 0
            with open('new/images.txt', 'a') as f:  # add image to dataset list
                f.write(img_file + '\n')
            with open('new/labels/' + Path(file).name, 'a') as f:  # write label
                for l in labels[i]:
                    f.write('%g %.6f %.6f %.6f %.6f\n' % tuple(l))
            shutil.copyfile(src=img_file, dst='new/images/' + Path(file).name.replace('txt', 'jpg'))  # copy images


def kmean_anchors(path='../coco/train2017.txt', n=9, img_size=(320, 640)):
    # from utils.utils import *; _ = kmean_anchors()
    # Produces a list of target kmeans suitable for use in *.cfg files
    from utils.datasets import LoadImagesAndLabels
    thr = 0.20  # IoU threshold

    def print_results(thr, wh, k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        iou = wh_iou(torch.Tensor(wh), torch.Tensor(k))
        max_iou, min_iou = iou.max(1)[0], iou.min(1)[0]
        bpr, aat = (max_iou > thr).float().mean(), (iou > thr).float().mean() * n  # best possible recall, anch > thr
        print('%.2f iou_thr: %.3f best possible recall, %.2f anchors > thr' % (thr, bpr, aat))
        print('kmeans anchors (n=%g, img_size=%s, IoU=%.3f/%.3f/%.3f-min/mean/best): ' %
              (n, img_size, min_iou.mean(), iou.mean(), max_iou.mean()), end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    def fitness(thr, wh, k):  # mutation fitness
        iou = wh_iou(wh, torch.Tensor(k)).max(1)[0]  # max iou
        bpr = (iou > thr).float().mean()  # best possible recall
        return iou.mean() * 0.80 + bpr * 0.20  # weighted combination

    # Get label wh
    wh = []
    dataset = LoadImagesAndLabels(path, augment=True, rect=True, cache_labels=True)
    nr = 1 if img_size[0] == img_size[1] else 10  # number augmentation repetitions
    for s, l in zip(dataset.shapes, dataset.labels):
        wh.append(l[:, 3:5] * (s / s.max()))  # image normalized to letterbox normalized wh
    wh = np.concatenate(wh, 0).repeat(nr, axis=0)  # augment 10x
    wh *= np.random.uniform(img_size[0], img_size[1], size=(wh.shape[0], 1))  # normalized to pixels (multi-scale)

    # Darknet yolov3.cfg anchors
    if n == 9:
        k = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
        k = print_results(thr, wh, k)
    else:
        # Kmeans calculation
        from scipy.cluster.vq import kmeans
        print('Running kmeans on %g points...' % len(wh))
        s = wh.std(0)  # sigmas for whitening
        k, dist = kmeans(wh / s, n, iter=20)  # points, mean distance
        k *= s
        k = print_results(thr, wh, k)

    # # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')

    # Evolve
    wh = torch.Tensor(wh)
    f, ng = fitness(thr, wh, k), 1000  # fitness, generations
    for _ in tqdm(range(ng), desc='Evolving anchors'):
        kg = (k.copy() * (1 + np.random.random() * np.random.randn(*k.shape) * 0.20)).clip(min=2.0)
        fg = fitness(thr, wh, kg)
        if fg > f:
            f, k = fg, kg.copy()
            print(fg, list(k.round().reshape(-1)))
    k = print_results(thr, wh, k)

    return k


def print_mutation(hyp, results, bucket=''):
    # Print mutation results to evolve.txt (for use with train.py --evolve)
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.3g' * len(results) % results  # results (P, R, mAP, F1, test_loss)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if bucket:
        os.system('rm evolve.txt && gsutil cp gs://%s/evolve.txt .' % bucket)  # download evolve.txt

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    np.savetxt('evolve.txt', x[np.argsort(-fitness(x))], '%10.3g')  # save sort by fitness

    if bucket:
        os.system('gsutil cp evolve.txt gs://%s' % bucket)  # upload evolve.txt


def apply_classifier(x, model, img, im0):
    # applies a second stage classifier to yolo outputs

    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0.shape)

            # Classes
            pred_cls1 = d[:, 6].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x

#fixme --yang.xu
# def fitness(x):
#     # Returns fitness (for use with results.txt or evolve.txt)
#     return x[:, 2] * 0.3 + x[:, 3] * 0.7  # weighted combination of x=[p, r, mAP@0.5, F1 or mAP@0.5:0.95]

def fitness(x):
    # Returns fitness (for use with results.txt or evolve.txt)
    w = [0.0, 0.01, 0.99, 0.00]  # weights for [P, R, mAP, F1]@0.5 or [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)

# Plotting functions ---------------------------------------------------------------------------------------------------
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_wh_methods():  # from utils.utils import *; plot_wh_methods()
    # Compares the two methods for width-height anchor multiplication
    # https://github.com/ultralytics/yolov3/issues/168
    x = np.arange(-4.0, 4.0, .1)
    ya = np.exp(x)
    yb = torch.sigmoid(torch.from_numpy(x)).numpy() * 2

    fig = plt.figure(figsize=(6, 3), dpi=150)
    plt.plot(x, ya, '.-', label='yolo method')
    plt.plot(x, yb ** 2, '.-', label='^2 power method')
    plt.plot(x, yb ** 2.5, '.-', label='^2.5 power method')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.legend()
    fig.tight_layout()
    fig.savefig('comparison.png', dpi=200)


def plot_images(imgs, targets, paths=None, fname='images.jpg'):
    # Plots training images overlaid with targets
    imgs = imgs.cpu().numpy()
    targets = targets.cpu().numpy()

    #fixme
    # targets = targets[targets[:, 1] == 2]  # plot only one class

    fig = plt.figure(figsize=(10, 10))
    bs, _, h, w = imgs.shape  # batch size, _, height, width
    bs = min(bs, 16)  # limit plot to 16 images
    ns = np.ceil(bs ** 0.5)  # number of subplots
    for i in range(bs):
        #fixme skip target is null
        if i >= targets.shape[0] or not targets[i].shape[0]:
            continue
        boxes = xywh2xyxy(targets[targets[:, 0]==i, 2:6]).T
        boxes[[0, 2]] *= w
        boxes[[1, 3]] *= h
        plt.subplot(ns, ns, i + 1).imshow(imgs[i].transpose(1, 2, 0))
        plt.plot(boxes[[0, 2, 2, 0, 0]], boxes[[1, 1, 3, 3, 1]], '.-')
        plt.axis('off')
        if paths is not None:
            s = Path(paths[i]).name
            plt.title(s[:min(len(s), 40)], fontdict={'size': 8})  # limit to 40 characters
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close()


def plot_test_txt():  # from utils.utils import *; plot_test()
    # Plot test.txt histograms
    x = np.loadtxt('test.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig('hist2d.jpg', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    fig.tight_layout()
    plt.savefig('hist1d.jpg', dpi=200)


def plot_targets_txt():  # from utils.utils import *; plot_targets_txt()
    # Plot test.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32)
    x = x.T

    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    fig.tight_layout()
    plt.savefig('targets.jpg', dpi=200)


def plot_evolution_results(hyp):  # from utils.utils import *; plot_evolution_results(hyp)
    # Plot hyperparameter evolution results in evolve.txt
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = fitness(x)
    weights = (f - f.min()) ** 2  # for weighted results
    fig = plt.figure(figsize=(12, 10))
    matplotlib.rc('font', **{'size': 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 7]
        # mu = (y * weights).sum() / weights.sum()  # best weighted result
        mu = y[f.argmax()]  # best single result
        plt.subplot(4, 5, i + 1)
        plt.plot(mu, f.max(), 'o', markersize=10)
        plt.plot(y, f, '.')
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})  # limit to 40 characters
        print('%15s: %.3g' % (k, mu))
    fig.tight_layout()
    plt.savefig('evolve.png', dpi=200)


def plot_results_overlay(start=0, stop=0):  # from utils.utils import *; plot_results_overlay()
    # Plot training results files 'results*.txt', overlaying train and val losses
    s = ['train', 'train', 'train', 'Precision', 'mAP@0.5', 'val', 'val', 'val', 'Recall', 'F1']  # legends
    t = ['GIoU', 'Objectness', 'Classification', 'P-R', 'mAP-F1']  # titles
    for f in sorted(glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        fig, ax = plt.subplots(1, 5, figsize=(14, 3.5))
        ax = ax.ravel()
        for i in range(5):
            for j in [i, i + 5]:
                y = results[j, x]
                if i in [0, 1, 2]:
                    y[y == 0] = np.nan  # dont show zero loss values
                ax[i].plot(x, y, marker='.', label=s[j])
            ax[i].set_title(t[i])
            ax[i].legend()
            ax[i].set_ylabel(f) if i == 0 else None  # add filename
        fig.tight_layout()
        fig.savefig(f.replace('.txt', '.png'), dpi=200)


def plot_results(start=0, stop=0, bucket='', id=(), class_num=1, result_dir=None, png_name='result.png', title=''):
    # from utils.utils import *; plot_results()
    # Plot training results files 'results*.txt'
    fig, ax = plt.subplots(2, 5, figsize=(14, 7), constrained_layout=True)
    ax = ax.ravel()
    s = ['GIoU', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val GIoU', 'val Objectness', 'val Classification', 'mAP@0.5', 'F1']
    if not result_dir:
        result_dir = '../result_output/{}_cls/'.format(class_num)
    if bucket:
        files = ['https://storage.googleapis.com/%s/results%g.txt' % (bucket, x) for x in id]
    else:
        # files = glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')
        files = glob.glob(result_dir + 'results*.txt')
    for f in sorted(files):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        for i in range(10):
            y = results[i, x]
            if i in [0, 1, 2, 5, 6, 7]:
                y[y == 0] = np.nan  # dont show zero loss values
            ax[i].plot(x, y, marker='.', label=Path(f).stem)
            ax[i].set_title(s[i])
            ax[i].grid(True)
            if i in [5, 6, 7]:  # share train and val loss y axes
                ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])

    fig.suptitle(title)
    fig.savefig(result_dir + png_name, dpi=200)



if __name__ == '__main__':
    # img_file = '/media/lab/Yang/data/xView_YOLO/images/2230.tif'
    # lbl_file = '/media/lab/Yang/data/xView_YOLO/labels/60_cls/2230.txt'
    # from PIL import Image
    # img = np.array(Image.open(img_file))
    # lbl = np.loadtxt(lbl_file, delimiter=' ')
    # # plot_images(img, lbl)
    # h, w = img.shape[:2]
    # boxes = xywh2xyxy(lbl[:, 1:5]).T
    # boxes[[0, 2]] *= w
    # boxes[[1, 3]] *= h
    # plt.imshow(img)
    # plt.plot(boxes[[0, 2, 2, 0, 0]], boxes[[1, 1, 3, 3, 1]], '.-')


    '''
    plot  results
    '''
    class_num = 1
    plot_results(class_num=class_num)

