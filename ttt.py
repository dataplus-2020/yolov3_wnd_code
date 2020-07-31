import numpy as np
import pandas as pd
from utils.parse_config_xview import *

# with open('/media/lab/Yang/data/xView_YOLO/labels/60_cls/1889.txt', 'r') as f:
#     print(f.readline())


# df_txt = pd.read_csv('/media/lab/Yang/data/xView_YOLO/labels/60_cls/1889.txt', header=None, delimiter=' ')
# txt_arr = df_txt.to_numpy()


opt = parse_data_cfg('data_xview/60_cls/xview.data')
