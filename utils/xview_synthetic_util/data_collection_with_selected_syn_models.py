import glob
import numpy as np
import os
import pandas as pd
import shutil
from utils.data_process_distribution_vis_util import process_wv_coco_for_yolo_patches_no_trnval as pwv
from utils.xview_synthetic_util import preprocess_xview_syn_data_distribution as pps


def generate_new_xview_lbl_with_model_id(type='val', nonmatch=3, comments='', px_thres=None, whr_thres=None):
    '''
    generate new xview train val annotations (with model id)
    generate new xview train val list
    :return:
    '''
    args = pwv.get_args(px_thres, whr_thres)
    if comments:
        lbl_name = 'xview{}_lbl_{}.txt'.format(type, comments)
        xview_val_lbl_with_model_name = 'xview{}_lbl_{}_with_model.txt'.format(type, comments)
        data_save_dir = os.path.join(args.data_save_dir, comments)
        data_backup_dir = os.path.join(args.data_list_save_dir, comments)
        if not os.path.exists(data_backup_dir):
            os.mkdir(data_backup_dir)
    else:
        lbl_name = 'xview{}_lbl.txt'.format(type)
        xview_val_lbl_with_model_name = 'xview{}_lbl_with_model.txt'.format(type)
        data_save_dir = args.data_save_dir
        data_backup_dir = args.data_list_save_dir

    ori_val_lbl_txt = os.path.join(data_save_dir, lbl_name)
    df_ori_val = pd.read_csv(ori_val_lbl_txt, header=None)
    ori_val_names = [os.path.basename(f) for f in df_ori_val.iloc[:, 0]]

    xview_val_lbl_with_model_txt = open(os.path.join(args.data_save_dir, xview_val_lbl_with_model_name), 'w')
    if px_thres:
        des_val_lbl_path = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px{}whr{}_only_model/'.format(px_thres, whr_thres)
        src_lbl_path = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px{}whr{}_model/'.format(px_thres, whr_thres)
    else:
        des_val_lbl_path = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_only_model/'
        src_lbl_path = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_model/'

    if not os.path.exists(des_val_lbl_path):
        os.mkdir(des_val_lbl_path)

    for i in range(len(ori_val_names)):
        f = os.path.join(src_lbl_path, ori_val_names[i])
        name = os.path.basename(f)
        if not pps.is_non_zero_file(f):
            shutil.copy(f, os.path.join(des_val_lbl_path, name))
            xview_val_lbl_with_model_txt.write('%s\n' % os.path.join(des_val_lbl_path, name))
            continue

        colomns = np.arange(0, 6)
        src_lbl = pd.read_csv(f, header=None, sep=' ', index_col=False, names=colomns)
        des_lbl_txt = open(os.path.join(des_val_lbl_path, name), 'w')

        for i in range(src_lbl.shape[0]):
            if np.isnan(src_lbl.iloc[i, 5]):
                modelid = nonmatch
            else:
                modelid = src_lbl.iloc[i, 5]
            des_lbl_txt.write("%d %.8f %.8f %.8f %.8f %d\n" % (src_lbl.iloc[i, 0], src_lbl.iloc[i, 1], src_lbl.iloc[i, 2], src_lbl.iloc[i, 3], src_lbl.iloc[i, 4], modelid))
        des_lbl_txt.close()
        xview_val_lbl_with_model_txt.write('%s\n' % os.path.join(des_val_lbl_path, name))
    xview_val_lbl_with_model_txt.close()
    shutil.copy(os.path.join(data_save_dir, xview_val_lbl_with_model_name), os.path.join(data_backup_dir, xview_val_lbl_with_model_name))


def generate_new_syn_lbl_with_model_id(comments='', nonmatch=3):
    '''
    generate new syn annotations (with model id)
    generate new syn  list
    :return:
    '''
    syn_args = pps.get_syn_args()
    ori_val_lbl_txt = os.path.join(syn_args.syn_data_list_dir, '{}_{}_lbl.txt'.format(syn_args.syn_display_type, syn_args.class_num))
    df_ori_val = pd.read_csv(ori_val_lbl_txt, header=None)
    ori_val_names = [os.path.basename(f) for f in df_ori_val.loc[:, 0]]

    xview_val_lbl_with_model_name = '{}_{}_lbl{}.txt'.format(syn_args.syn_display_type, syn_args.class_num, comments)
    xview_val_lbl_with_model_txt = open(os.path.join(syn_args.syn_data_list_dir, xview_val_lbl_with_model_name), 'w')

    des_val_lbl_path = '/media/lab/Yang/data/xView_YOLO/labels/608/{}_{}_cls_xcycwh_only_model/'.format(syn_args.syn_display_type, syn_args.class_num)
    if not os.path.exists(des_val_lbl_path):
        os.mkdir(des_val_lbl_path)

    src_lbl_path = '/media/lab/Yang/data/xView_YOLO/labels/608/{}_{}_cls_xcycwh_model/'.format(syn_args.syn_display_type, syn_args.class_num)

    for name in ori_val_names:
        src_lbl_file = os.path.join(src_lbl_path, name)
        if not pps.is_non_zero_file(src_lbl_file):
            shutil.copy(src_lbl_file, os.path.join(des_val_lbl_path, name))
            xview_val_lbl_with_model_txt.write('%s\n' % os.path.join(des_val_lbl_path, name))
            continue

        colomns = np.arange(0, 6)
        src_lbl = pd.read_csv(src_lbl_file, header=None, sep=' ', index_col=False, names=colomns) #fixme --- **** index_col=False ****
        des_lbl_txt = open(os.path.join(des_val_lbl_path, name), 'w')

        for i in range(src_lbl.shape[0]):
            if np.isnan(src_lbl.iloc[i, -1]):
                des_lbl_txt.write("%d %.8f %.8f %.8f %.8f %d\n" % (src_lbl.iloc[i, 0], src_lbl.iloc[i, 1], src_lbl.iloc[i, 2], src_lbl.iloc[i, 3], src_lbl.iloc[i, 4], nonmatch))
            else:
                des_lbl_txt.write("%d %.8f %.8f %.8f %.8f %d\n" % (src_lbl.iloc[i, 0], src_lbl.iloc[i, 1], src_lbl.iloc[i, 2], src_lbl.iloc[i, 3], src_lbl.iloc[i, 4], src_lbl.iloc[i, 5]))
        des_lbl_txt.close()
        xview_val_lbl_with_model_txt.write('%s\n' % os.path.join(des_val_lbl_path, name))
    xview_val_lbl_with_model_txt.close()
    # shutil.copy(os.path.join(args.data_save_dir, xview_val_lbl_with_model_name), os.path.join(args.data_list_save_dir, xview_val_lbl_with_model_name))



def change_labels_from_px6whr4_to_px20whr4():
    '''
    drop b-boxes whose length is larger than px_thres
    or drop b-boxes whose wh_ratio is larger than whr_thres
    :return:
    '''
    syn_args = pps.get_syn_args()
    px_thres = 20
    whr_thres = 4
    label_path = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px20whr4_only_model/'

    if not os.path.exists(label_path):
        os.mkdir(label_path)
    else:
        shutil.rmtree(label_path)
        os.mkdir(label_path)
    src_files = glob.glob('/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_only_model/*.txt')
    for f in src_files:
        shutil.copy(f, label_path)
    label_files = np.sort(glob.glob(os.path.join(label_path, '*.txt')))
    for lf in label_files:
        if not pps.is_non_zero_file(lf):
            continue
        df_txt = pd.read_csv(lf, header=None, delimiter=' ')
        # print('before ', df_txt.shape)
        # if '1945_5.txt' in lf:
        #     print(lf)
        for ix in df_txt.index:
            df_txt.loc[ix, 1:4] = df_txt.loc[ix, 1:4] * syn_args.tile_size
            # whr = np.max(df_txt.loc[ix, 3] / df_txt.loc[ix, 4], df_txt.loc[ix, 4] / df_txt.loc[ix, 3])
            whr = np.maximum(df_txt.loc[ix, 3] / df_txt.loc[ix, 4], df_txt.loc[ix, 4] / df_txt.loc[ix, 3])
            if int(df_txt.loc[ix, 3]) <= px_thres or int(df_txt.loc[ix, 4]) <= px_thres or whr > whr_thres:
                df_txt = df_txt.drop(ix)
            else:
                df_txt.loc[ix, 1:4] = df_txt.loc[ix, 1:4]/syn_args.tile_size
        df_txt.to_csv(lf, header=False, index=False, sep=' ')

    a = glob.glob('/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px20whr4/*.txt')
    b = glob.glob('/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px20whr4_only_model/*.txt')

    aa = []
    bb = []
    for i in range(len(a)):
        if not pps.is_non_zero_file(a[i]):
            aa.append(os.path.basename(a[i]))
        if not pps.is_non_zero_file(b[i]):
            bb.append(os.path.basename(b[i]))
    ax = [x for x in aa if x not in bb]
    bx = [x for x in bb if x not in aa]
    print(ax)
    print(bx)


def change_labels_from_px6whr4_to_px23whr4():
    '''
    only drop b-boxes those at the edge of the image, and the length of bbox is larger than px_thres
    or drop b-boxes whose wh_ratio is larger than whr_thres
    :return:
    '''
    syn_args = pps.get_syn_args()
    px_thres = 23
    whr_thres = 4
    label_path = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px{}whr{}_only_model/'.format(px_thres, whr_thres)

    if not os.path.exists(label_path):
        os.mkdir(label_path)
    else:
        shutil.rmtree(label_path)
        os.mkdir(label_path)

    src_files = glob.glob('/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_only_model/*.txt')
    for f in src_files:
        shutil.copy(f, label_path)
    label_files = np.sort(glob.glob(os.path.join(label_path, '*.txt')))
    for lf in label_files:
        if not pps.is_non_zero_file(lf):
            continue
        df_txt = pd.read_csv(lf, header=None, delimiter=' ')
        # print('before ', df_txt.shape)
        # if '1044_1.txt' in lf:
        #     print(lf)
        for ix in df_txt.index:
            #fixme
            # df_txt.loc[ix, 1:4] = df_txt.loc[ix, 1:4] * syn_args.tile_size
            # # whr = np.max(df_txt.loc[ix, 3] / df_txt.loc[ix, 4], df_txt.loc[ix, 4] / df_txt.loc[ix, 3])
            # whr = np.maximum(df_txt.loc[ix, 3] / df_txt.loc[ix, 4], df_txt.loc[ix, 4] / df_txt.loc[ix, 3])
            # if int(df_txt.loc[ix, 3]) <= px_thres or int(df_txt.loc[ix, 4]) <= px_thres or whr > whr_thres:
            #     df_txt = df_txt.drop(ix)
            # else:
            #     df_txt.loc[ix, 1:4] = df_txt.loc[ix, 1:4]/syn_args.tile_size
            bbx = df_txt.loc[ix, 1:4] * syn_args.tile_size
            xl = round(bbx.loc[1] - bbx.loc[3]/2)
            yl = round(bbx.loc[2] - bbx.loc[4]/2)
            xr = round(bbx.loc[1] + bbx.loc[3]/2)
            yr = round(bbx.loc[2] + bbx.loc[4]/2)
            bbx_wh = max(bbx.loc[3]/bbx.loc[4], bbx.loc[4]/bbx.loc[3])
            w = round(bbx.loc[3])
            h = round(bbx.loc[4])
            if bbx_wh >= whr_thres:
                df_txt = df_txt.drop(ix)
            elif (xl<=0 or xr>=syn_args.tile_size-1) and (w <= px_thres or h <= px_thres):
                df_txt = df_txt.drop(ix)
            elif (yl<=0 or yr>=syn_args.tile_size-1) and (w <= px_thres or h <= px_thres):
                df_txt = df_txt.drop(ix)
        df_txt.to_csv(lf, header=False, index=False, sep=' ')

    a = glob.glob('/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px{}whr{}/*.txt'.format(px_thres, whr_thres))
    b = glob.glob('/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px{}whr{}_only_model/*.txt'.format(px_thres, whr_thres))

    aa = []
    bb = []
    for i in range(len(a)):
        if not pps.is_non_zero_file(a[i]):
            aa.append(os.path.basename(a[i]))
        if not pps.is_non_zero_file(b[i]):
            bb.append(os.path.basename(b[i]))
    ax = [x for x in aa if x not in bb]
    bx = [x for x in bb if x not in aa]
    print(ax)
    print(bx)

def check_regenerate_labels_with_model_id_based_on_previous():
    new_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px23whr3_all_model/'
    old_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px23whr3_all_model_5.2/'
    new_ll = glob.glob(os.path.join(new_dir, '*.txt'))
    old_ll = glob.glob(os.path.join(old_dir, '*.txt'))
    new_names = [os.path.basename(s) for s in new_ll]
    old_names =  [os.path.basename(s) for s in old_ll]
    same_names = [s for s in old_names if s in new_names]
    for n in same_names:
        if not pps.is_non_zero_file(os.path.join(old_dir, n)):
            continue
        df_old = pd.read_csv(os.path.join(old_dir, n), header=None, sep=' ')
        df_new = pd.read_csv(os.path.join(new_dir, n), header=None, sep=' ')
        if not df_old.loc[:, :4].equals(df_new):
            print(n)
        else:
            shutil.copy(os.path.join(old_dir, n),
                        os.path.join(new_dir, n))
            # 2309_5.txt
            # 2122_5.txt
            # 2309_0.txt
            # 1817_2.txt

def change_labels_from_old_px23whr3(px_thres=23, whr_thres=3):
    args = pwv.get_args(px_thres, whr_thres)
    old_model_lbl_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px23whr3_all_model_4.26/'
    part_added_model_lbl_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px23whr3_part_model_added/'
    new_lbl = glob.glob(os.path.join(args.annos_save_dir, '*.txt'))
    new_model_lbl_dir = args.annos_save_dir[:-1] + '_all_model/'
    if not os.path.exists(new_model_lbl_dir):
        os.mkdir(new_model_lbl_dir)
    else:
        shutil.rmtree(new_model_lbl_dir)
        os.mkdir(new_model_lbl_dir)

    for f in new_lbl:
        name = os.path.basename(f)
        omf = os.path.join(old_model_lbl_dir, name)
        pamf = os.path.join(part_added_model_lbl_dir, name)
        if os.path.exists(omf):
            shutil.copy(omf, os.path.join(new_model_lbl_dir, name))
        elif os.path.exists(pamf):
            shutil.copy(pamf, os.path.join(new_model_lbl_dir, name))


def change_end_model_id(px_thres=23, whr_thres=3, end_mid=6):
    args = pwv.get_args(px_thres, whr_thres)
    old_model_lbl_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px23whr3_all_model/'
    lbl_files = glob.glob(os.path.join(old_model_lbl_dir, '*.txt'))

    backup_lbl_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px23whr3_all_model_backup/'
    if not os.path.exists(backup_lbl_dir):
        os.mkdir(backup_lbl_dir)
        for f in lbl_files:
            name = os.path.basename(f)
            shutil.copy(f, os.path.join(backup_lbl_dir, name))

    for f in lbl_files:
        if not pps.is_non_zero_file(f):
            continue
        df_txt = pd.read_csv(f, header=None, sep=' ')
        df_txt.loc[df_txt.loc[:, 5]==end_mid, 5] = end_mid+1
        if df_txt.empty:
           continue
        df_txt.to_csv(f, sep=' ', header=False, index=False)


def backup_val_rgb_bbx_indices_lbl(cmt='', typestr='', px_thres=None, whr_thres=None):
    bbox_folder_name = '{}_images_with_bbox_with_indices'.format(cmt)
    src_bbx_path = os.path.join(args.cat_sample_dir,'image_with_bbox_indices', bbox_folder_name)

    val_bbox_folder_name = '{}_{}_images_with_bbox_with_indices'.format(cmt, typestr)
    dst_bbx_path = os.path.join(args.cat_sample_dir,'image_with_bbox_indices', val_bbox_folder_name)

    if not os.path.exists(dst_bbx_path):
        os.makedirs(dst_bbx_path)

    val_lbl_file = os.path.join(args.data_save_dir, cmt, 'xview{}_lbl_{}.txt'.format(typestr, cmt))
    save_lbl_path = os.path.join(args.txt_save_dir, 'lbl_with_model_id', '{}_{}_model'.format(cmt, typestr))
    print(os.path.exists(save_lbl_path), save_lbl_path)
    if not os.path.exists(save_lbl_path):
        os.makedirs(save_lbl_path)

    val_lbl = pd.read_csv(val_lbl_file, header=None)
    for vl in val_lbl.loc[:, 0]:
        shutil.copyfile(vl, os.path.join(save_lbl_path, os.path.basename(vl)))
        if pps.is_non_zero_file(vl):
            img_name = os.path.basename(vl).replace('.txt', '.jpg')
            shutil.copyfile(os.path.join(src_bbx_path, img_name),
                            os.path.join(dst_bbx_path, img_name))


def bakcup_all_bbox_with_model_id(cmt='', typestr='', px_thres=None, whr_thres=None):
    src_lbl_path = os.path.join(args.txt_save_dir, 'lbl_with_model_id', '{}_{}_model'.format(cmt, typestr))
    lbl_with_mid_files = np.sort(glob.glob(os.path.join(src_lbl_path, '*.txt')))

    des_lbl_path = args.annos_save_dir[:-1] + '_all_model/'
    if not os.path.exists(des_lbl_path):
        os.mkdir(des_lbl_path)

    for f in lbl_with_mid_files:
        shutil.copy(f, des_lbl_path)


if __name__ == '__main__':
    '''
    draw bbox on rgb images with label indices
    '''
    # syn = False
    # px_thres = 23
    # whr_thres = 3
    # pxwhr = 'px{}whr{}_seed17'.format(px_thres, whr_thres)
    # pps.draw_bbx_on_rgb_images_with_indices(syn, pxwhr=pxwhr, px_thres=px_thres, whr_thres=whr_thres)

    '''
    get val rgb images with bbox indices 
    get val lbl for label modelid
    '''
    # px_thres = 23
    # whr_thres = 3
    # seed = 17
    # args = pwv.get_args(px_thres, whr_thres)
    # cmt = 'px{}whr{}_seed{}'.format(px_thres, whr_thres, seed)
    # # typestr = 'val'
    # typestr = 'train'
    # backup_val_rgb_bbx_indices_lbl(cmt, typestr, px_thres, whr_thres)

    '''
    after label bbox with model id
    backup all label_with_model_id 
    '''
    # px_thres = 23
    # whr_thres = 3
    # seed = 17
    # args = pwv.get_args(px_thres, whr_thres)
    # cmt = 'px{}whr{}_seed{}'.format(px_thres, whr_thres, seed)
    # # typestr = 'val'
    # typestr = 'train'
    # bakcup_all_bbox_with_model_id(cmt, typestr, px_thres, whr_thres)

    '''
    generate new xviewval_lbl_with_model.txt
    '''
    # type = 'train'
    # type = 'val'
    # nonmatch = 3
    # generate_new_xview_lbl_with_model_id(type, nonmatch)

    # type = 'val'
    # nonmatch = 5
    # px_thres = 23
    # whr_thres = 3
    # comments = 'px{}whr{}_seed17'.format(px_thres, whr_thres)
    # generate_new_xview_lbl_with_model_id(type, nonmatch, comments, px_thres, whr_thres)

    '''
    generate new syn_*_lbl_with_model.txt
    '''
    # comments = '_with_model'
    # nonmatch = 3
    # generate_new_syn_lbl_with_model_id(comments, nonmatch)


    '''
    change labels from px6whr4 to px20whr4
    '''
    # change_labels_from_px6whr4_to_px20whr4()

    # ['1585_7.txt', '1945_5.txt']
    # for bbx in ax:
    #     df_a = pd.read_csv(os.path.join('/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px20whr4_only_model/', bbx), header=None, delimiter=' ').to_numpy()
    #     df_a[:, 1:5] = df_a[:,1:5]*syn_args.tile_size
    #     print(df_a)

    # df_txt = pd.read_csv('/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_only_model/1945_5.txt', header=None, sep=' ')
    # df_txt.iloc[:, 1:5] = df_txt.iloc[:, 1:5] * syn_args.tile_size
    # df_txt = df_txt[df_txt[:, 4]>20]

    '''
    change labels from px6whr4 to px23whr4
    '''
    # change_labels_from_px6whr4_to_px23whr4()
    '''
    check how many new labels are the same as previous
    * manually change the different bbox**********
    '''
    # check_regenerate_labels_with_model_id_based_on_previous()

    '''
    change labels from px23whr3_4.26 to px23whr3
    '''
    # px_thres = 23
    # whr_thres = 3
    # change_labels_from_old_px23whr3(px_thres, whr_thres)

    '''
    change end model id 
    '''
    # px_thres = 23
    # whr_thres = 3
    # end_id = 6
    # change_end_model_id(px_thres, whr_thres)

    '''
    draw bbox on rgb images with model_id
    '''
    # syn = False
    # px_thres = 23
    # whr_thres = 3
    # pxwhr = 'px{}whr{}_seed17'.format(px_thres, whr_thres)
    # pps.draw_bbx_on_rgb_images_with_model_id(syn, pxwhr=pxwhr, px_thres=px_thres, whr_thres=whr_thres)
