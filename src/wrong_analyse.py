import pandas as pd
pd.set_option('max_colwidth',150)
pd.options.mode.chained_assignment = None
import numpy as np

WRONG_PRED_DIR = './wrong_prediction.csv'

raw_data = pd.read_csv(WRONG_PRED_DIR)[['ori_label', 'our_label_true', 'our_label_pred']]

label_gb = raw_data.groupby(by=['ori_label', 'our_label_pred']).size().reset_index(name='freq')
label_gb.to_csv('group_by_res.csv', index=False)