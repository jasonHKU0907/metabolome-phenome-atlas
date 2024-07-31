
import os
import re
import glob
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
from joblib import Parallel, delayed
from mne.stats import bonferroni_correction

nb_cpus = 10

def process(nmr_f, mydf, cov_f_lst):
    tmp_df = mydf[['target_y_ordinal', nmr_f] + cov_f_lst]
    tmp_df.rename(columns={nmr_f: 'x_nmr'}, inplace=True)
    rm_eid_idx = tmp_df.index[tmp_df.x_nmr.isnull() == True]
    tmp_df.drop(rm_eid_idx, axis=0, inplace=True)
    tmp_df.reset_index(inplace=True, drop=True)
    nb_all= len(tmp_df)
    nb_case1 = len(tmp_df.loc[tmp_df.target_y_ordinal == 1])
    nb_case2 = len(tmp_df.loc[tmp_df.target_y_ordinal == 2])
    nb_case3 = len(tmp_df.loc[tmp_df.target_y_ordinal == 3])
    try:
        mod_logit = OrderedModel(tmp_df['target_y_ordinal'], tmp_df[['x_nmr'] + cov_f_lst], distr='logit')
        mod_log = mod_logit.fit(method='bfgs', disp=False)
        oratio = np.round(np.exp(mod_log.params.x_nmr), 5)
        pval = mod_log.pvalues.x_nmr
        ci_mod = mod_log.conf_int(alpha=0.05, cols=None)
        lbd, ubd = np.round(np.exp(ci_mod.loc['x_nmr'][0]), 5), np.round(np.exp(ci_mod.loc['x_nmr'][1]), 5)
        tmpout = [nmr_f, nb_all, nb_case1, nb_case2, nb_case3, oratio, lbd, ubd, pval]
    except:
        tmpout = [nmr_f, nb_all, nb_case1, nb_case2, nb_case3, np.nan, np.nan, np.nan, np.nan]
    return tmpout

def results_summary(tgt_out_df):
    oratio_out_lst, p_out_lst = [], []
    for i in range(len(tgt_out_df)):
        oratio = f'{tgt_out_df.oratio.iloc[i]:.2f}'
        lbd = f'{tgt_out_df.or_lbd.iloc[i]:.2f}'
        ubd = f'{tgt_out_df.or_ubd.iloc[i]:.2f}'
        oratio_out_lst.append(oratio + ' [' + lbd + '-' + ubd + ']')
        if tgt_out_df.pval_bfi.iloc[i] < 0.001:
            p_out_lst.append('***')
        elif tgt_out_df.pval_bfi.iloc[i] < 0.01:
            p_out_lst.append('**')
        elif tgt_out_df.pval_bfi.iloc[i] < 0.05:
            p_out_lst.append('*')
        else:
            p_out_lst.append('')
    return (oratio_out_lst, p_out_lst)

dpath = '/Volumes/JasonWork/Projects/MetaAtlas/'
cls_tgt_df = pd.read_csv(dpath + 'Results/Trajectory/DiseaseSpan/Multimorbidity/ClusterData_final/Cluster_22.csv',
                         usecols = ['eid', 'target_y_ordinal'])

nmr_df = pd.read_csv(dpath + 'Data/MetaData/NMR_Preprocessed.csv')
nmr_f_lst = nmr_df.columns.tolist()[1:]
cov_df = pd.read_csv(dpath + 'Data/Covariates/Covariates.csv')
mydf = pd.merge(nmr_df, cov_df, how = 'inner', on = ['eid'])
mydf = mydf.loc[mydf.Split == 1]
mydf.reset_index(inplace = True, drop = True)

mydf = pd.merge(mydf, cls_tgt_df, how = 'inner', on = ['eid'])

cov_f_lst = ["Age", "Sex", "TDI", "BMI", "Smoke", "Statin", "FastingTime"]


tgt_out_df = Parallel(n_jobs=nb_cpus)(delayed(process)(nmr_f, mydf, cov_f_lst) for nmr_f in nmr_f_lst)
tgt_out_df = pd.DataFrame(tgt_out_df)
tgt_out_df.columns = ['NMR_code', 'nb_individuals', 'nb_case_level1', 'nb_case_level2', 'nb_case_level3',
                      'oratio', 'or_lbd', 'or_ubd', 'pval_raw']
_, p_f_bfi = bonferroni_correction(tgt_out_df.pval_raw.fillna(1), alpha=0.05)
tgt_out_df['pval_bfi'] = p_f_bfi
tgt_out_df.loc[tgt_out_df['pval_bfi'] >= 1, 'pval_bfi'] = 1
tgt_out_df['or_output'], tgt_out_df['pval_significant'] = results_summary(tgt_out_df)
tgt_out_df = tgt_out_df[['NMR_code', 'nb_individuals', 'nb_case_level1', 'nb_case_level2', 'nb_case_level3', 'oratio',
                         'or_lbd', 'or_ubd', 'pval_raw', 'pval_bfi', 'or_output', 'pval_significant']]
tgt_out_df.to_csv(dpath + 'Results/Trajectory/DiseaseSpan/Multimorbidity/Cluster_22.csv', index=False)

