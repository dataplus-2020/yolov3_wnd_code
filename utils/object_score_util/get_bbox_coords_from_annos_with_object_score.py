from utils.object_score_util import misc_utils, eval_utils
import glob
import os
import numpy as np
import cv2
import pandas as pd

IMG_FORMAT = 'png'
TEX_FORMAT = 'txt'


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def get_object_bbox_after_group(label_path, save_path, class_label=0, min_region=20, link_r=30, px_thresh=6, whr_thres=4):
    '''
    get cat id and bbox ratio based on the label file
    group all the black pixels, and each group is assigned an id (start from 1)
    :param label_path:
    :param save_path:
    :param class_label:
    :param min_region: the smallest #pixels (area) to form an object
    :param link_r: the #pixels between two connected components to be grouped
    :param whr_thres:
    :return: (catid, xcenter, ycenter, w, h) the bbox is propotional to the image size
    '''
    lbl_files = np.sort(glob.glob(os.path.join(label_path, '*.{}'.format(IMG_FORMAT))))

    lbl_files = [os.path.join(label_path, f) for f in lbl_files if os.path.isfile(os.path.join(label_path, f))]
    lbl_names = [os.path.basename(f) for f in lbl_files]

    osc = eval_utils.ObjectScorer(min_region=min_region, min_th=0.5, link_r=link_r, eps=2) #  link_r=10
    for i, f in enumerate(lbl_files):
        lbl = 1 - misc_utils.load_file(f) / 255 # h, w, c
        lbl_groups = osc.get_object_groups(lbl)
        lbl_group_map = eval_utils.display_group(lbl_groups, lbl.shape[:2], need_return=True)
        group_ids = np.sort(np.unique(lbl_group_map))
        f_txt = open(os.path.join(save_path, lbl_names[i].replace(lbl_names[i][-3:], TEX_FORMAT)), 'w')
        for id in group_ids[1:]: # exclude id==0
            min_w = np.min(np.where(lbl_group_map == id)[1])
            min_h = np.min(np.where(lbl_group_map == id)[0])
            max_w = np.max(np.where(lbl_group_map == id)[1])
            max_h = np.max(np.where(lbl_group_map == id)[0])

            w = max_w - min_w
            h = max_h - min_h
            whr = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            if whr > whr_thres:
                continue
            elif min_w <= 0 and (w <= px_thresh or h <= px_thresh):
                continue
            elif min_h <= 0 and (w <= px_thresh or h <= px_thresh):
                continue
            elif max_w >= lbl.shape[1] -1  and (w <= px_thresh or h <= px_thresh):
                continue
            elif max_h >= lbl.shape[0] -1  and (w <= px_thresh or h <= px_thresh):
                continue
            min_w = min_w / lbl.shape[1]
            min_h = min_h / lbl.shape[0]
            w = w / lbl.shape[1]
            h = h / lbl.shape[0]
            xc = min_w + w/2.
            yc = min_h + h/2.


            f_txt.write("%s %s %s %s %s\n" % (class_label, xc, yc, w, h))
        f_txt.close()


def plot_img_with_bbx(img_file, lbl_file, save_path, label_index=False, Model_id=False):
    if not is_non_zero_file(lbl_file):
        # print(is_non_zero_file(lbl_file))
        return
    # print(img_file)
    img = cv2.imread(img_file) # h, w, c
    h, w = img.shape[:2]

    df_lbl = pd.read_csv(lbl_file, header=None, delimiter=' ').to_numpy() # delimiter , error_bad_lines=False
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
        gt_bbx = df_lbl[ix, 1:].astype(np.int64)
        img = cv2.rectangle(img, (gt_bbx[0], gt_bbx[1]), (gt_bbx[2], gt_bbx[3]), (255, 0, 0), 2)
        pl = ''
        if label_index:
            pl = '{}'.format(ix)
        elif Model_id:
            mid = int(df_lbl[ix, 5])
            pl = '{}'.format(mid)
        else:
             pl = '{}'.format(cat_id)
        cv2.putText(img, text=pl, org=(gt_bbx[0] + 10, gt_bbx[1] + 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))
    cv2.imwrite(os.path.join(save_path, img_file.split('/')[-1]), img)


if __name__ == "__main__":

    min_region =100
    link_r = 15
    label_path = '/object_score_utils/test/'
    txt_path = '/object_score_utils/txt_xcycwh/'
    if not os.path.exists(txt_path):
        os.mkdir(txt_path)
    get_object_bbox_after_group(label_path, txt_path, class_label=0, min_region=min_region, link_r=link_r)

    img_file = '/object_score_utils/test/airplanes_berlin_200_76_GT.jpg'

    lbl_file = '/media/lab/Yang/code/yolov3/utils_object_score/txt_xcycwh/' + 'minr{}_linkr{}_'.format(min_region, link_r)+ 'px6whr4_ng0_' + 'airplanes_berlin_200_76_GT.txt'
    save_path = '/object_score_utils/bbx_label/'
    plot_img_with_bbx(img_file, lbl_file, save_path)


