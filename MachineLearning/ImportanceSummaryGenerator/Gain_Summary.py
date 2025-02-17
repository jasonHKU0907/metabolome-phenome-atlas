
import glob
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd
import seaborn as sns

def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c.replace("_","")) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l

type = 'Incident'
type = 'Prevalent'

dpath = '/Volumes/JasonWork/Projects/MetaAtlas/'
tgt_dir_lst = sort_nicely(glob.glob(dpath + 'Revision/Results/Prediction/'+type+'/NMR_Importance/*.csv'))
tgt_dict = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv', encoding='latin-1')
tgt_dict = tgt_dict.loc[tgt_dict[type+'_analysis'] == 1]
root_lst = list(set(tgt_dict.Root.tolist()))
root_lst.sort()
nmr_code_df = pd.read_csv(tgt_dir_lst[0], usecols = ['NMR_code'])
nmr_code_df.sort_values(by = 'NMR_code', inplace = True)
root_imp_df = nmr_code_df.copy()


for root in tqdm(root_lst):
    root_name = root.split(' ')[0] + '_' + root.split(' ')[1]
    name_lst = tgt_dict.loc[tgt_dict.Root == root].NAME.tolist()
    name_imp_df = nmr_code_df.copy()
    for name in name_lst:
        tgt_dir = dpath + 'Revision/Results/Prediction/'+type+'/NMR_Importance/'+name+'.csv'
        tgt_imp_df = pd.read_csv(tgt_dir, usecols = ['NMR_code', 'Importance'])
        tgt_imp_df.rename(columns = {'Importance':name}, inplace = True)
        name_imp_df = pd.merge(name_imp_df, tgt_imp_df, how = 'left', on = ['NMR_code'])
    mean_imp = name_imp_df.iloc[:, 1:].mean(axis=1).tolist()
    name_imp_df[root_name] = mean_imp
    name_imp_df.to_csv(dpath + 'Revision/Results/Prediction/'+type+'/ImportanceSummary/GainSummary/'+root_name+'.csv', index = False)
    root_imp_df = pd.merge(root_imp_df, name_imp_df[['NMR_code', root_name]], how = 'left', on = ['NMR_code'])

root_imp_df.to_csv(dpath + 'Revision/Results/Prediction/'+type+'/ImportanceSummary/GainSummary/Importance_by_root.csv', index = False)


