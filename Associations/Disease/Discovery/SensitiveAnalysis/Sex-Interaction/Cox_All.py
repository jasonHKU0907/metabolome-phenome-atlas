
import glob
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from tqdm import tqdm
import os
import re
from statsmodels.stats.multitest import fdrcorrection
from mne.stats import bonferroni_correction
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('error')

def results_summary(tgt_out_df):
    hr_out_lst, p_out_lst = [], []
    for i in range(len(tgt_out_df)):
        hr = f'{tgt_out_df.hr.iloc[i]:.2f}'
        lbd = f'{tgt_out_df.hr_lbd.iloc[i]:.2f}'
        ubd = f'{tgt_out_df.hr_ubd.iloc[i]:.2f}'
        hr_out_lst.append(hr + ' [' + lbd + '-' + ubd + ']')
        if tgt_out_df.pval_bfi.iloc[i] < 0.001:
            p_out_lst.append('***')
        elif tgt_out_df.pval_bfi.iloc[i] < 0.01:
            p_out_lst.append('**')
        elif tgt_out_df.pval_bfi.iloc[i] < 0.05:
            p_out_lst.append('*')
        else:
            p_out_lst.append('')
    return (hr_out_lst, p_out_lst)


def process(nmr_f, tmp_df, my_formula):
    tmp_df.rename(columns={nmr_f: 'x_nmr'}, inplace=True)
    tmp_df['SexInter'] = tmp_df.Sex*tmp_df.x_nmr
    rm_eid_idx = tmp_df.index[tmp_df.x_nmr.isnull() == True]
    tmp_df.drop(rm_eid_idx, axis=0, inplace=True)
    tmp_df.reset_index(inplace=True, drop=True)
    nb_all, nb_case = len(tmp_df), tmp_df.target_y.sum()
    prop_case = np.round(nb_case / nb_all * 100, 3)
    i, tmpout = 1, []
    while ((len(tmpout) == 0) | (i > 1e7)):
        cph = CoxPHFitter(penalizer=1e-7 * i)
        i = 10 * i
        try:
            cph.fit(tmp_df, duration_col='BL2Target_yrs', event_col='target_y', formula=my_formula)
            hr = np.round(cph.hazard_ratios_.SexInter, 5)
            lbd = np.round(np.exp(cph.confidence_intervals_).loc['SexInter'][0], 5)
            ubd = np.round(np.exp(cph.confidence_intervals_).loc['SexInter'][1], 5)
            pval = cph.summary.p.SexInter
            tmpout = [nmr_f, nb_all, nb_case, prop_case, hr, lbd, ubd, pval]
        except:
            pass
    if tmpout == []:
        tmpout = [nmr_f, nb_all, nb_case, prop_case, np.nan, np.nan, np.nan, np.nan]
    else:
        pass
    return tmpout

def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c.replace("_","")) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l

nb_cpus = 10

dpath = '/home1/jiayou/Documents/Projects/MetaAtlas/'
dpath = '/Volumes/JasonWork/Projects/MetaAtlas/'
badoutfile = dpath + 'Revision/Results/Associations/Discovery/Incident/bad_tgt_all-sexiter.csv'

tgt_df = pd.read_csv(dpath + 'Data/TargetData/IncidentDiseaseTable.csv')
tgt_df = tgt_df.loc[(tgt_df.SEX != 1)&(tgt_df.SEX != 2)]
tgt_lst = tgt_df.NAME.tolist()

nmr_df = pd.read_csv(dpath + 'Data/MetaData/NMR_Preprocessed.csv')
nmr_f_lst = nmr_df.columns.tolist()[1:]
cov_df = pd.read_csv(dpath + 'Data/Covariates/Covariates.csv')
mydf = pd.merge(nmr_df, cov_df, how = 'inner', on = ['eid'])
mydf = mydf.loc[mydf.Split == 1]
mydf.reset_index(inplace = True, drop = True)

my_formula = "Age + C(Sex) + TDI + BMI + C(Smoke) + Statin + FastingTime + x_nmr + SexInter"

bad_tgt = []

tgt_name = 'E4_DM2'
nmr_f = nmr_f_lst[100]

for tgt_name in tqdm(tgt_lst):
    try:
        tmp_tgt_df = pd.read_csv(dpath + 'Data/TargetData/TargetData/'+tgt_name+'.csv', usecols=['eid', 'target_y', 'BL2Target_yrs'])
        rm_bl_idx = tmp_tgt_df.index[tmp_tgt_df.BL2Target_yrs <= 0]
        tmp_tgt_df.drop(rm_bl_idx, axis=0, inplace=True)
        tmp_tgt_df.reset_index(inplace=True, drop=True)
        tmp_df = pd.merge(mydf, tmp_tgt_df, how='inner', on=['eid'])
        if tmp_df.target_y.sum()>=50:
            tgt_out_df = Parallel(n_jobs=nb_cpus)(delayed(process)(nmr_f, tmp_df, my_formula) for nmr_f in nmr_f_lst)
            tgt_out_df = pd.DataFrame(tgt_out_df)
            tgt_out_df.columns = ['NMR_code', 'nb_individuals', 'nb_case', 'prop_case(%)', 'hr', 'hr_lbd', 'hr_ubd', 'pval_raw']
            _, p_f_bfi = bonferroni_correction(tgt_out_df.pval_raw.fillna(1), alpha=0.05)
            tgt_out_df['pval_bfi'] = p_f_bfi
            tgt_out_df.loc[tgt_out_df['pval_bfi'] >= 1, 'pval_bfi'] = 1
            tgt_out_df['hr_output'], tgt_out_df['pval_significant'] = results_summary(tgt_out_df)
            tgt_out_df = tgt_out_df[['NMR_code', 'nb_individuals', 'nb_case', 'prop_case(%)', 'hr', 'hr_lbd',
                                     'hr_ubd', 'pval_raw', 'pval_bfi', 'hr_output', 'pval_significant']]
            tgt_out_df.to_csv(dpath + 'Revision/Results/Associations/Discovery/Incident/All-SexInteraction/' + tgt_name + '.csv', index=False)
        else:
            bad_tgt.append(tgt_name)
    except:
        bad_tgt.append(tgt_name)

bad_df = pd.DataFrame({'NAME':bad_tgt})
bad_df = pd.merge(bad_df, tgt_df)
bad_df.to_csv(badoutfile, index=False)

