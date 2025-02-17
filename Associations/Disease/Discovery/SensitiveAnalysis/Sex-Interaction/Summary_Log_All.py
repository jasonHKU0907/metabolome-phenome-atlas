
import glob
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from statsmodels.stats.multitest import fdrcorrection
warnings.filterwarnings('error')

def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c.replace("_","")) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l

dpath = '/Volumes/JasonWork/Projects/MetaAtlas/'

group = 'All-SexInteraction'
tgt_info_df = pd.read_csv(dpath + 'Data/TargetData/PrevalentDiseaseTable.csv')
tgt_info_df = tgt_info_df[['NAME', 'LONGNAME', 'Root', 'ICD_10', 'SEX', 'PRE_CONDITIONS', 'CONDITIONS',
                 'CONTROL_PRECONDITIONS', 'CONTROL_CONDITIONS', 'CONTROL_EXCLUDE', 'INCLUDE']]
results_dir_lst = sort_nicely(glob.glob(dpath + 'Revision/Results/Associations/Discovery/Prevalent/'+group+'/*.csv'))

meta_df = pd.read_csv(dpath + 'Data/MetaData/NMR_Preprocessed.csv', usecols = ['eid'])
cov_df = pd.read_csv(dpath + 'Data/Covariates/Covariates.csv', usecols = ['eid', 'Age', 'Sex', 'Split'])
meta_df = pd.merge(meta_df, cov_df, how = 'inner', on = ['eid'])
meta_df = meta_df.loc[meta_df.Split == 1]
meta_df.reset_index(inplace=True, drop=True)

tgt_lst, nb_all_lst, nb_case_lst, nb_ctrl_lst, prop_case_lst = [], [], [], [], []
nb_sig_raw, nb_sig_fdr, nb_sig_bfi1, nb_sig_bfi2 = [], [], [], []

for results_dir in tqdm(results_dir_lst):
    try:
        tgt = os.path.basename(results_dir)[:-4]
        tgt_dir = dpath + 'Data/TargetData/TargetData/' + tgt + '.csv'
        tgt_df = pd.read_csv(tgt_dir, usecols=['eid', 'target_y', 'BL2Target_yrs'])
        tgt_df = pd.merge(tgt_df, meta_df, how='inner', on=['eid'])
        rm_bl_idx = tgt_df.index[(tgt_df.target_y == 1) & (tgt_df.BL2Target_yrs > 0)]
        tgt_df.drop(rm_bl_idx, axis=0, inplace=True)
        tgt_df.reset_index(inplace=True, drop=True)
        tgt_lst.append(tgt)
        nb_all_lst.append(len(tgt_df))
        nb_case_lst.append(tgt_df.target_y.sum())
        nb_ctrl_lst.append(len(tgt_df) - tgt_df.target_y.sum())
        prop_case_lst.append(np.round(tgt_df.target_y.sum() / len(tgt_df), 3))
        results_df = pd.read_csv(results_dir, usecols=['NMR_code', 'pval_raw'])
        _, p_f_fdr = fdrcorrection(results_df.pval_raw.fillna(1))
        results_df['p_fdr'] = p_f_fdr
        results_df1, results_df2, results_df3, results_df4 = results_df.copy(), results_df.copy(), results_df.copy(), results_df.copy()
        results_df1 = results_df.loc[results_df.pval_raw<0.05]
        results_df2 = results_df.loc[results_df.p_fdr<0.05]
        results_df3 = results_df.loc[results_df.pval_raw<0.05/313]
        results_df4 = results_df.loc[results_df.pval_raw<0.05/313/len(results_dir_lst)]
        nb_sig_raw.append(len(results_df1))
        nb_sig_fdr.append(len(results_df2))
        nb_sig_bfi1.append(len(results_df3))
        nb_sig_bfi2.append(len(results_df4))
    except:
        pass

mydf = pd.DataFrame([tgt_lst, nb_all_lst, nb_case_lst, nb_ctrl_lst, prop_case_lst, nb_sig_raw, nb_sig_fdr, nb_sig_bfi1, nb_sig_bfi2])
mydf = mydf.T
mydf.columns = ['NAME', 'NB_all', 'NB_case', 'NB_control', 'Prop_case', 'NB_sig_nmr_raw', 'NB_sig_nmr_fdr', 'NB_sig_nmr_bfi1', 'NB_sig_nmr_bfi2']
mydf = pd.merge(mydf, tgt_info_df, how = 'inner', on = ['NAME'])

mydf.to_csv(dpath + 'Revision/Results/Associations/Discovery/Prevalent/Summary/Data_Summary_'+group+'.csv', index = False)

