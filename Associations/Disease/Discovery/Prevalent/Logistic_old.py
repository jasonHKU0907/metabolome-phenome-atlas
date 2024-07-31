
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
import os
import re
from statsmodels.stats.multitest import fdrcorrection
from mne.stats import bonferroni_correction
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('error')

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


def process(nmr_f, tmp_df, cov_f_lst):
    tmp_df.rename(columns={nmr_f: 'x_nmr'}, inplace=True)
    rm_eid_idx = tmp_df.index[tmp_df.x_nmr.isnull() == True]
    tmp_df.drop(rm_eid_idx, axis=0, inplace=True)
    tmp_df.reset_index(inplace=True, drop=True)
    nb_all, nb_case = len(tmp_df), tmp_df.target_y.sum()
    prop_case = np.round(nb_case / nb_all * 100, 3)
    Y = tmp_df.target_y
    X = tmp_df[cov_f_lst + ['x_nmr']]
    try:
        log_mod = sm.Logit(Y, sm.add_constant(X)).fit()
        oratio = np.round(np.exp(log_mod.params).loc['x_nmr'], 5)
        pval = log_mod.pvalues.loc['x_nmr']
        ci_mod = log_mod.conf_int(alpha=0.05)
        lbd, ubd = np.round(np.exp(ci_mod.loc['x_nmr'][0]), 5), np.round(np.exp(ci_mod.loc['x_nmr'][1]), 5)
        tmpout = [nmr_f, nb_all, nb_case, prop_case, oratio, lbd, ubd, pval]
    except:
        tmpout = [nmr_f, nb_all, nb_case, prop_case, oratio, lbd, ubd, pval]
    return tmpout

def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c.replace("_","")) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l

nb_cpus = 20

dpath = '/home1/jiayou/Documents/Projects/MetaAtlas/'
#dpath = '/Volumes/JasonWork/Projects/MetaAtlas/'
badoutfile = dpath + 'Results/Associations/Discovery/Prevalent/bad_tgt_old.csv'

tgt_df = pd.read_csv(dpath + 'Data/TargetData/PrevalentDiseaseTable.csv')
tgt_lst = tgt_df.NAME.tolist()

nmr_df = pd.read_csv(dpath + 'Data/MetaData/NMR_Preprocessed.csv')
nmr_f_lst = nmr_df.columns.tolist()[1:]
cov_df = pd.read_csv(dpath + 'Data/Covariates/Covariates.csv')
mydf = pd.merge(nmr_df, cov_df, how = 'inner', on = ['eid'])
mydf = mydf.loc[mydf.Age>=60]
mydf.reset_index(inplace=True, drop=True)
mydf = mydf.loc[mydf.Split == 1]
mydf.reset_index(inplace = True, drop = True)

cov_f_lst_full = ["Age", "Sex", "TDI", "BMI", "Smoke", "Statin", "FastingTime"]
cov_f_lst_sex = ["Age", "TDI", "BMI", "Smoke", "Statin", "FastingTime"]

bad_tgt = []

for tgt_name in tqdm(tgt_lst):
    tgt_file = dpath + 'Data/TargetData/TargetData/'+tgt_name+'.csv'
    sex_id = tgt_df.loc[tgt_df.NAME == tgt_name].SEX.iloc[0]
    cov_f_lst = [cov_f_lst_sex if (sex_id == 1) | (sex_id == 2) else cov_f_lst_full][0]
    tmp_tgt_df = pd.read_csv(tgt_file, usecols=['eid', 'target_y', 'BL2Target_yrs'])
    rm_bl_idx = tmp_tgt_df.index[(tmp_tgt_df.target_y == 1) & (tmp_tgt_df.BL2Target_yrs > 0)]
    tmp_tgt_df.drop(rm_bl_idx, axis=0, inplace=True)
    tmp_tgt_df.reset_index(inplace=True, drop=True)
    tmp_df = pd.merge(mydf, tmp_tgt_df, how='inner', on=['eid'])
    try:
        if tmp_df.target_y.sum()>=50:
            tgt_out_df = Parallel(n_jobs=nb_cpus)(delayed(process)(nmr_f, tmp_df, cov_f_lst) for nmr_f in nmr_f_lst)
            tgt_out_df = pd.DataFrame(tgt_out_df)
            tgt_out_df.columns = ['NMR_code', 'nb_individuals', 'nb_case', 'prop_case(%)', 'oratio', 'or_lbd', 'or_ubd', 'pval_raw']
            _, p_f_bfi = bonferroni_correction(tgt_out_df.pval_raw.fillna(1), alpha=0.05)
            tgt_out_df['pval_bfi'] = p_f_bfi
            tgt_out_df.loc[tgt_out_df['pval_bfi'] >= 1, 'pval_bfi'] = 1
            tgt_out_df['or_output'], tgt_out_df['pval_significant'] = results_summary(tgt_out_df)
            tgt_out_df = tgt_out_df[['NMR_code', 'nb_individuals', 'nb_case', 'prop_case(%)', 'oratio', 'or_lbd',
                                     'or_ubd', 'pval_raw', 'pval_bfi', 'or_output', 'pval_significant']]
            tgt_out_df.to_csv(dpath + 'Results/Associations/Discovery/Prevalent/Old/' + tgt_name + '.csv', index=False)
        else:
            bad_tgt.append(tgt_name)
    except:
        bad_tgt.append(tgt_name)

bad_df = pd.DataFrame(bad_tgt)
bad_df.to_csv(badoutfile, index=False)

