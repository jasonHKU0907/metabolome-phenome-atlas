
import glob
import numpy as np
import pandas as pd
import os
import re
import warnings
warnings.filterwarnings('error')


def get_sig_yrs2(p_lst):
    try:
        i = 0
        while ((p_lst[i] < 0.05) | (p_lst[i + 1] < 0.05)):
            i += 1
        out = i-1
    except:
        if (p_lst[13]<0.05) & (p_lst[14]>=0.05):
            out = 13
        elif (p_lst[13]<0.05) & (p_lst[14]<0.05):
            out = 14
        elif (p_lst[13]>=0.05) & (p_lst[14]<0.05):
            out = 12
        else:
            out = 'bad'
    return out

#numbers = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.09]
#numbers = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
#get_sig_yrs2(numbers)


dpath = '/Volumes/JasonWork/Projects/MetaAtlas/'

my_label_lst = ['15_14', '14_13', '13_12', '12_11', '11_10',
                '10_9', '9_8', '8_7', '7_6', '6_5', '5_4', '4_3', '3_2', '2_1', '1_0']

mydf = pd.read_csv(dpath + 'Results/Trajectory/DiseaseSpan/temporal_pval_15frames.csv', low_memory=False)

sig_yr2_lst = []
for i in range(len(mydf)):
    pval_lst = list(mydf.loc[i, ['Pval_' + ele for ele in my_label_lst]])
    pval_lst.reverse()
    try:
        pval_lst = [float(ele) for ele in pval_lst]
        sig_yr2_lst.append(get_sig_yrs2(pval_lst)+1)
    except:
        sig_yr2_lst.append(np.nan)

mydf.to_csv(dpath + 'Results/Trajectory/DiseaseSpan/temporal_pval_15frames.csv', index=False)
