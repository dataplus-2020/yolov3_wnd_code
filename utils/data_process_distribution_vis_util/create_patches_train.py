
import preprocess_wv_for_windturbine as pp

"""
first, update the get_args() function in pp to desired paths (make sure to change syn_txt_data).
this script creates folder yolov3_wnd_code/{syn_txt_dir} containing lists of paths to training imgs/lbls.
copy/paste these lists into training lists in the original syn_txt_dir folder, then update
syn_0_xview_number in the main .data file to the new number of training examples.
"""

syn_args = pp.get_args()

# crop images into patches if necessary
# pp.resize_crop_windturbine(syn_args, px_thres=5)

comment = 'synth'

pp.all_to_trn(syn_args, comment, seed=17, pxs='px5_seed17')

pp.create_syn_data(syn_args, comment, pxs='px5_seed17')
