
import preprocess_wv_for_windturbine as pp

"""
first, update the get_args() function in pp to desired paths.
this script splits the data into training and validation sets and creates folder yolov3_wnd_code/{syn_txt_dir} containing
lists of paths to the specific training and validation imgs/lbls.
it also creates a .data file containing metadata and paths to these lists.
"""

syn_args = pp.get_args()

# crop images into patches if necessary
pp.resize_crop_windturbine(syn_args, px_thres=5)

comment = 'wnd'

pp.split_syn_wnd_trn_val(syn_args, seed=17, pxs='px5_seed17')

pp.create_syn_data(syn_args, comment, pxs='px5_seed17')
