
import preprocess_wv_for_windturbine as pp

syn_args = pp.get_args()
# pp.resize_crop_windturbine(syn_args)

'''
pp.split_syn_wnd_trn_val(syn_args, comment='wnd', seed=17, pxs='px5_seed17')

comment = 'bg'
# create_syn_data(comment, seed=17)
pp.create_syn_data(syn_args, comment, pxs='px5_seed17')
'''

pp.resize_crop_windturbine(syn_args, px_thres=5)