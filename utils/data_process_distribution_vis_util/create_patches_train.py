
import preprocess_wv_for_windturbine as pp

syn_args = pp.get_args()

# pp.resize_crop_windturbine(syn_args, px_thres=5)

comment = 'synth'

pp.all_to_trn(syn_args, comment, seed=17, pxs='px5_seed17')

# create_syn_data(comment, seed=17)
pp.create_syn_data(syn_args, comment, pxs='px5_seed17')