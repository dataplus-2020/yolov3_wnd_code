import glob
import numpy as np
import os
# from utils.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc
import pandas as pd
import cv2
import shutil
import argparse

def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def convert_norm(size, box):
    '''
    https://blog.csdn.net/xuanlang39/article/details/88642010
    :param size:  w h
    :param box: y1 x1 y2 x2
    :return: xc yc w h  (relative values)
    '''
    dh = 1. / (size[1])  # h--1--y
    dw = 1. / (size[0])  # w--0--x

    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0  # (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


def resize_crop_windturbine(syn_args, px_thres = 10):
    """
    divides images and labels from syn_data_dir into 4 patches of size tile_size, writes .jpg and .txt files to save_img_dir and save_lbl_dir
    :param syn_args: var set to get_args()
    :param px_thres: min distance a bounding box can be from patch boundary to be counted
    """
    wt_dir = syn_args.syn_data_dir
    wt_images = np.sort(glob.glob(os.path.join(wt_dir, '*.jpg')))
    wt_lbls = np.sort(glob.glob(os.path.join(wt_dir, '*.txt')))
    save_img_dir = syn_args.syn_img_dir
    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)
    save_lbl_dir = syn_args.syn_annos_dir
    if not os.path.exists(save_lbl_dir):
        os.mkdir(save_lbl_dir)
    w = syn_args.tile_size
    h = syn_args.tile_size

    for i in range(len(wt_images)):
        img = cv2.imread(wt_images[i])
        name = os.path.basename(wt_images[i])
        # if name == 'naip_324_AZ_WND.jpg':
        #     print('name', name)
        height, width = img.shape[:2]
        rnum = np.ceil(height/h).astype(np.int)
        cnum = np.ceil(width/w).astype(np.int)
        # print('label file', wt_lbls[i])
        tl_w, tl_h = 0, 0
        br_w, br_h = width, height
        c_w, c_h = br_w-w, br_h-h

        for r in range(rnum):
            for c in range(cnum):
                wmin = c*w
                wmax = (c+1)*w
                hmin = r*h
                hmax = (r+1)*h
                if hmax >= height and wmax >= width:
                    hmin = height - h
                    wmin = width - w
                    chip = img[hmin : height, wmin : width, :3]
                elif hmax >= height and wmax < width:
                    hmin = height - h
                    chip = img[hmin : height, wmin : wmax, :3]
                elif hmax < height and wmax >= width:
                    wmin = width - w
                    chip = img[hmin : hmax, wmin : width, :3]
                else:
                    chip = img[hmin : hmax, wmin : wmax, :3]
                cv2.imwrite(os.path.join(save_img_dir, name.split('.')[0] + '_i{}j{}.jpg'.format(r, c)), chip)

        
        if not is_non_zero_file(wt_lbls[i]):
            shutil.copy(wt_lbls[i], os.path.join(save_lbl_dir, name.replace('.jpg', '.txt')))
        else:
                lbl = pd.read_csv(wt_lbls[i], header=None, sep=' ').to_numpy()
                b0_list = []
                b1_list = []
                b2_list = []
                b3_list = []
                bboxes = lbl[:, 1:]
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * width
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * height
                bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2]/2
                bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3]/2
                bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
                bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
                for ti in range(bboxes.shape[0]):
                    bbox = bboxes[ti, :] # xc, yc, w, h
                    b0 = np.clip(bbox, [tl_w, tl_h, tl_w, tl_h], [w-1, h-1, w-1, h-1])
                    b1 = bbox.copy()
                    # print('b1', b1.shape)
                    b1[[0, 2]] = b1[[0, 2]] - c_w
                    b1 = np.clip(b1, [tl_w, tl_h, tl_w, tl_h], [w-1, h-1, w-1, h-1])
                    b2 = bbox.copy()
                    b2[[1, 3]] = b2[[1, 3]] - c_h
                    b2 = np.clip(b2, [tl_w, tl_h, tl_w, tl_h], [w-1, h-1, w-1, h-1])
                    b3 = bbox.copy()
                    b3[[0, 2]] = b3[[0, 2]] - c_w
                    b3[[1, 3]] = b3[[1, 3]] - c_h
                    b3 = np.clip(b3, [tl_w, tl_h, tl_w, tl_h], [w-1, h-1, w-1, h-1])
                    if b0[2]-b0[0] > px_thres and b0[3]-b0[1] > px_thres:
                        b0_list.append(convert_norm((w, h), b0))
                    if b1[2]-b1[0] > px_thres and b1[3]-b1[1] > px_thres:
                        # b1[[0, 2]] = b1[[0, 2]] - (br_w-w)
                        b1_list.append(convert_norm((w, h), b1))
                    if b2[2]-b2[0] > px_thres and b2[3]-b2[1] > px_thres:
                        # b2[[1, 3]]= b2[[1, 3]] - (br_h-h)
                        b2_list.append(convert_norm((w, h), b2))
                    if b3[2]-b3[0] > px_thres and b3[3]-b3[1] > px_thres:
                        # b3[[0, 2]] = b3[[0, 2]] - c_w
                        # b3[[1, 3]] = b3[[1, 3]] - c_h
                        b3_list.append(convert_norm((w, h), b3)) #cid, xc, yc, w, h, mid
                if len(b0_list):
                    f_txt = open(os.path.join(save_lbl_dir, name.split('.')[0] + '_i0j0.txt'), 'w')
                    for i0 in b0_list:
                        f_txt.write( "%s %s %s %s %s\n" % (0, i0[0], i0[1], i0[2], i0[3]))
                    f_txt.close()
                if len(b1_list):
                    f_txt = open(os.path.join(save_lbl_dir, name.split('.')[0] + '_i0j1.txt'), 'w')
                    for i1 in b1_list:
                        # print('i1', i1)
                        f_txt.write( "%s %s %s %s %s\n" % (0, i1[0], i1[1], i1[2], i1[3]))
                    f_txt.close()
                if len(b2_list):
                    f_txt = open(os.path.join(save_lbl_dir, name.split('.')[0] + '_i1j0.txt'), 'w')
                    for i2 in b2_list:
                        f_txt.write( "%s %s %s %s %s\n" % (0, i2[0], i2[1], i2[2], i2[3]))
                    f_txt.close()
                if len(b3_list):
                    f_txt = open(os.path.join(save_lbl_dir, name.split('.')[0] + '_i1j1.txt'), 'w')
                    for i3 in b3_list: #cid, xc, yc, w, h, mid
                        f_txt.write( "%s %s %s %s %s\n" % (0, i3[0], i3[1], i3[2], i3[3]))
                    f_txt.close()
        '''
        for i in range(2):
            for j in range(2):
                f_txt = open(os.path.join(save_lbl_dir, name.split('.')[0] + '_i{}j{}.txt'.format(i, j)), 'w')
                f_txt.write("")
                f_txt.close()
        '''


def split_syn_wnd_trn_val(syn_args, comment='wnd', seed=17, pxs='px10_seed17'):
    """
    splits data into training and validation sets and creates lists of paths to images and labels used for training/testing
    :param syn_args: var set to get_args()
    :param comment: determines names of .txt files
    :param seed: sets random seed
    :param pxs: determines names of .txt files
    """

    all_imgs = np.sort(glob.glob(os.path.join(syn_args.syn_img_dir, '*.jpg')))
    num_files = len(all_imgs)

    np.random.seed(seed)
    all_indices = np.random.permutation(num_files)
    # data_txt_dir = os.path.join(syn_args.syn_txt_dir, pxs)
    data_txt_dir = syn_args.syn_txt_dir
    if not os.path.exists(data_txt_dir):
        os.mkdir(data_txt_dir)

    trn_img_txt = open(os.path.join(data_txt_dir, '{}_train_img_{}.txt'.format(comment, pxs)), 'w')
    trn_lbl_txt = open(os.path.join(data_txt_dir, '{}_train_lbl_{}.txt'.format(comment, pxs)), 'w')

    val_img_txt = open(os.path.join(data_txt_dir, '{}_val_img_{}.txt'.format(comment, pxs)), 'w')
    val_lbl_txt = open(os.path.join(data_txt_dir, '{}_val_lbl_{}.txt'.format(comment, pxs)), 'w')

    num_val = int(num_files*syn_args.val_percent)
    lbl_dir = syn_args.syn_annos_dir

    for i in all_indices[:num_val]:
        val_img_txt.write('%s\n' % all_imgs[i])
        val_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(all_imgs[i]).replace('.jpg', '.txt')))
    val_img_txt.close()
    val_lbl_txt.close()
    for j in all_indices[num_val:]:
        trn_img_txt.write('%s\n' % all_imgs[j])
        trn_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(all_imgs[j]).replace('.jpg', '.txt')))
    trn_img_txt.close()
    trn_lbl_txt.close()
    
    
def all_to_trn(syn_args, comment='wnd', seed=17, pxs='px10_seed17'):

    all_imgs = np.sort(glob.glob(os.path.join(syn_args.syn_img_dir, '*.jpg')))
    num_files = len(all_imgs)

    # data_txt_dir = os.path.join(syn_args.syn_txt_dir, pxs)
    data_txt_dir = syn_args.syn_txt_dir
    if not os.path.exists(data_txt_dir):
        os.mkdir(data_txt_dir)

    trn_img_txt = open(os.path.join(data_txt_dir, '{}_train_img_{}.txt'.format(comment, pxs)), 'w')
    trn_lbl_txt = open(os.path.join(data_txt_dir, '{}_train_lbl_{}.txt'.format(comment, pxs)), 'w')

    lbl_dir = syn_args.syn_annos_dir

    for i in range(num_files):
        trn_img_txt.write('%s\n' % all_imgs[i])
        trn_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(all_imgs[i]).replace('.jpg', '.txt')))
    trn_img_txt.close()
    trn_lbl_txt.close()
    
    
def all_to_val(syn_args, comment='wnd', seed=17, pxs='px10_seed17'):

    all_imgs = np.sort(glob.glob(os.path.join(syn_args.syn_img_dir, '*.jpg')))
    num_files = len(all_imgs)

    # data_txt_dir = os.path.join(syn_args.syn_txt_dir, pxs)
    data_txt_dir = syn_args.syn_txt_dir
    if not os.path.exists(data_txt_dir):
        os.mkdir(data_txt_dir)

    val_img_txt = open(os.path.join(data_txt_dir, '{}_val_img_{}.txt'.format(comment, pxs)), 'w')
    val_lbl_txt = open(os.path.join(data_txt_dir, '{}_val_lbl_{}.txt'.format(comment, pxs)), 'w')

    lbl_dir = syn_args.syn_annos_dir

    for i in range(num_files):
        val_img_txt.write('%s\n' % all_imgs[i])
        val_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(all_imgs[i]).replace('.jpg', '.txt')))
    val_img_txt.close()
    val_lbl_txt.close()


def create_syn_data(syn_args, comment='wnd', pxs='px10_seed17'):
    """
    creates a .data file that stores metadata and the paths to train and val img/lbl lists
    :param syn_args: var set to get_args()
    :param comment: determines names of .data and .txt files
    :param pxs: determines names of .data and .txt files
    """
    data_txt_dir = syn_args.syn_txt_dir

    data_txt = open(os.path.join(data_txt_dir, '{}_{}.data'.format(comment, pxs)), 'w')
    data_txt.write('train={}/{}_train_img_{}.txt\n'.format(data_txt_dir, comment, pxs))
    data_txt.write('train_label={}/{}_train_lbl_{}.txt\n'.format(data_txt_dir, comment, pxs))

     #********** syn_0_xview_number corresponds to train*.py the number of train files
    #df = pd.read_csv(os.path.join(syn_args.syn_txt_dir, '{}_train_img_{}.txt'.format(comment, pxs)), header=None)
    #data_txt.write('syn_0_xview_number={}\n'.format(df.shape[0]))
    data_txt.write('classes=%s\n' % str(syn_args.class_num))

    data_txt.write('valid={}/{}_val_img_{}.txt\n'.format(data_txt_dir, comment, pxs))
    data_txt.write('valid_label={}/{}_val_lbl_{}.txt\n'.format(data_txt_dir, comment, pxs))
    data_txt.write('names={}/{}.names\n'.format(data_txt_dir, comment))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval={}'.format(comment))
    data_txt.close()


def get_args():
    """
    sets paths to various relavant folders for accessing imgs/lbls
    """
    parser = argparse.ArgumentParser()
    '''
    parser.add_argument("--syn_data_dir", type=str,
                        help="Path to folder containing raw synthetic images and annos ",
                        default='/home/jovyan/work/data/background_images_for_testing_original/')

    parser.add_argument("--syn_img_dir", type=str,
                        help="Path to folder containing cropped synthetic images ",
                        default='/home/jovyan/work/data/syn_uspp_bkg_shdw_scatter_uniform_60_wnd_v1/color_all_images_step182.4/')
    parser.add_argument("--syn_annos_dir", type=str, default='/home/jovyan/work/data/syn_uspp_bkg_shdw_scatter_uniform_60_wnd_v1_txt_xcycwh/minr100_linkr15_px15whr6_color_all_annos_txt_step182.4/',
                        help="syn label.txt")
    '''
    parser.add_argument("--syn_data_dir", type=str,
                        help="Path to folder containing raw synthetic images and annos ",
                        default='/home/jovyan/work/data/background_images_for_testing_original/')

    parser.add_argument("--syn_img_dir", type=str,
                        help="Path to folder containing cropped synthetic images",
                        default='/home/jovyan/work/data/cropped_bg_imgs/')
    parser.add_argument("--syn_annos_dir", type=str, default='/home/jovyan/work/data/cropped_bg_labels/',
                        help="Path to folder containing cropped synthetic labels")
    
    parser.add_argument("--syn_box_dir", type=str, default='/home/jovyan/work/data/wind_turbine_bbox/',
                        help="syn related txt files")

    parser.add_argument("--syn_txt_dir", type=str, default='/home/jovyan/work/code/yxu-yolov3-xview/data_bg/',
                        help="Path to folder where img/lbl paths are stored; used for training/testing")

    parser.add_argument("--syn_display_type", type=str, default='color',
                        help="texture, color, mixed")  # syn_color0, syn_texture0,
    #fixme ---***** min_region ***** change
    parser.add_argument("--min_region", type=int, default=100, help="300 100 the smallest #pixels (area) to form an object")
    parser.add_argument("--link_r", type=int, default=15,
                        help="the #pixels between two connected components to be grouped")
    parser.add_argument("--resolution", type=float, default=0.3, help="resolution of synthetic data")
    parser.add_argument("--tile_size", type=int, default=608, help="image size")
    parser.add_argument("--class_num", type=int, default=1, help="class number")
    parser.add_argument("--val_percent", type=float, default=0.25, help="train:val=0.75:0.25")


    args = parser.parse_args()
    if not os.path.exists(args.syn_annos_dir):
        os.makedirs(args.syn_annos_dir)
    if not os.path.exists(args.syn_txt_dir):
        os.makedirs(args.syn_txt_dir)

    return args




if __name__ == "__main__":
    syn_args = get_args()

    '''
    crop windturbin images
    recreate corresponging labels
    '''
    resize_crop_windturbine(syn_args, px_thres=5)

    '''
    check new lables
    plot bbox on images
    '''
    # lbl_file =  os.path.join(syn_args.syn_annos_dir, 'naip_324_AZ_WND_i1j1.txt')
    # lbl_file = os.path.join(syn_args.syn_annos_dir, 'naip_1101_CA_WND_i0j0.txt')
    # img_dir = syn_args.syn_img_dir

    # lbl_file = os.path.join(syn_args.syn_data_dir, 'naip_1101_CA_WND.txt')
    # # lbl_file = os.path.join(syn_args.syn_data_dir, 'naip_324_AZ_WND.txt')
    # img_dir = syn_args.syn_data_dir

    # save_dir = syn_args.syn_box_dir
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # name = os.path.basename(lbl_file)
    # gbc.plot_img_with_bbx(os.path.join(img_dir, name.replace('.txt', '.jpg')), lbl_file, save_dir)


    '''
    split train val
    '''
    split_syn_wnd_trn_val(syn_args, comment='wnd', seed=17, pxs='px5_seed17')

    '''
    create *.data
    '''
    comment = 'wnd'
    # create_syn_data(comment, seed=17)
    create_syn_data(syn_args, comment, pxs='px5_seed17')





