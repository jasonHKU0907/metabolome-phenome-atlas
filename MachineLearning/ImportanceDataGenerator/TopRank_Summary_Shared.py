
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

top_nb = 1
dpath = '/Volumes/JasonWork/Projects/MetaAtlas/'
tgt_dict = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv', encoding='latin-1')
tgt_dict = tgt_dict.loc[(tgt_dict.Incident_analysis == 1)&(tgt_dict.Prevalent_analysis == 1)]
tgt_name_lst = tgt_dict.NAME.tolist()
root_lst = list(set(tgt_dict.Root.tolist()))
root_lst.sort()
nmr_code_df = pd.read_csv(dpath + 'Results/Prediction/Incident/NMR_Importance/E4_DM2.csv', usecols = ['NMR_code'])
nmr_code_df.sort_values(by = 'NMR_code', inplace = True)
root_imp_df = nmr_code_df.copy()

for root in tqdm(root_lst):
    root_name = root.split(' ')[0] + '_' + root.split(' ')[1]
    name_lst = tgt_dict.loc[tgt_dict.Root == root].NAME.tolist()
    name_imp_df = nmr_code_df.copy()
    for name in name_lst:
        tgt_dir0 = dpath + 'Results/Prediction/Incident/NMR_Importance/'+name+'.csv'
        tgt_imp_df0 = pd.read_csv(tgt_dir0, usecols = ['NMR_code', 'TotalGain_cv'])
        tgt_imp_df0.sort_values(by = 'TotalGain_cv', ascending=False, inplace = True)
        tgt_imp_df0['rank_incident'] = [1]*top_nb + [0]*(len(tgt_imp_df0)-top_nb)
        tgt_imp_df0.drop(['TotalGain_cv'], axis = 1, inplace = True)
        tgt_dir1 = dpath + 'Results/Prediction/Prevalent/NMR_Importance/' + name + '.csv'
        tgt_imp_df1 = pd.read_csv(tgt_dir1, usecols=['NMR_code', 'TotalGain_cv'])
        tgt_imp_df1.sort_values(by='TotalGain_cv', ascending=False, inplace=True)
        tgt_imp_df1['rank_prevalent'] = [1] * top_nb + [0] * (len(tgt_imp_df1) - top_nb)
        name_imp_df = pd.merge(name_imp_df, tgt_imp_df0[['NMR_code', 'rank_incident']], how = 'left', on = ['NMR_code'])
        name_imp_df = pd.merge(name_imp_df, tgt_imp_df1[['NMR_code', 'rank_prevalent']], how = 'left', on = ['NMR_code'])
        name_imp_df[name] = name_imp_df['rank_incident'] + name_imp_df['rank_prevalent']
        name_imp_df[name].replace([0,1,2], [0,0,1], inplace = True)
        name_imp_df.drop(['rank_incident', 'rank_prevalent'], axis = 1, inplace = True)
    name_imp_df[root_name] = name_imp_df.iloc[:, 1:].sum(axis=1).tolist()
    name_imp_df.to_csv(dpath + 'Results/Prediction/SharedImportanceSummary/Top'+str(top_nb)+'_summary/'+root_name+'.csv', index = False)
    root_imp_df = pd.merge(root_imp_df, name_imp_df[['NMR_code', root_name]], how = 'left', on = ['NMR_code'])

root_imp_df['Total'] = root_imp_df.iloc[:, 1:].sum(axis=1).tolist()
root_imp_df.to_csv(dpath + 'Results/Prediction/SharedImportanceSummary/Top'+str(top_nb)+'_summary/Importance_by_root.csv', index = False)


