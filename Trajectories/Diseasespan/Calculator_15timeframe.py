
import glob
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from tqdm import tqdm
import os
import re
import warnings
from scipy import stats
warnings.filterwarnings('error')

dpath = '/Volumes/JasonWork/Projects/MetaAtlas/'
sig_df = pd.read_csv(dpath + 'Results/Associations/Discovery/Incident/Summary/Combined_Results_All.csv')
sig_df = sig_df.loc[sig_df.P_value<0.05/313/859]
sig_df.reset_index(inplace = True, drop = True)
nmr_df = pd.read_csv(dpath + 'Data/MetaData/NMR_Preprocessed.csv')

my_bin_lst = [-20, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0]
my_label_lst = ['15_14', '14_13', '13_12', '12_11', '11_10',
                '10_9', '9_8', '8_7', '7_6', '6_5', '5_4', '4_3', '3_2', '2_1', '1_0']


def cal_diff_p(tmp_df, my_bin_lst, my_label_lst):
    tmp_df['tmp_bins'] = pd.cut(tmp_df['BL2Target_yrs_matched'], bins = my_bin_lst, labels = my_label_lst)
    pval_lst = []
    for my_label in my_label_lst:
        try:
            case_df = tmp_df.loc[(tmp_df.tmp_bins == my_label) & (tmp_df.target_y == 1)]
            ctrl_df = tmp_df.loc[(tmp_df.tmp_bins == my_label) & (tmp_df.target_y == 0)]
            _, pval = stats.ttest_ind(case_df[tmp_nmr], ctrl_df[tmp_nmr], nan_policy='omit')
            pval_lst.append(pval)
        except:
            pval_lst.append(np.nan)
    return pval_lst

tmp_out_lst = []
for i in range(len(sig_df)):
    tmp_nmr = sig_df.NMR_code.iloc[i]
    tmp_dis = sig_df.NAME.iloc[i]
    tmp_nmr_df = nmr_df[['eid', tmp_nmr]]
    tmp_tgt_df = pd.read_csv(dpath + 'Results/Trajectory/DiseaseSpan/matched_data_short/' + tmp_dis + '.csv')
    tmp_df = pd.merge(tmp_tgt_df, tmp_nmr_df)
    pval_lst = cal_diff_p(tmp_df, my_bin_lst, my_label_lst)
    tmp_out_lst.append([tmp_nmr, tmp_dis] + pval_lst)
    print(i)

outdf = pd.DataFrame(tmp_out_lst)
outdf.columns = ['NMR', 'NAME'] + ['Pval_' + ele for ele in my_label_lst]
outdf.to_csv(dpath + 'Results/Trajectory/DiseaseSpan/temporal_pval_15frames.csv')



