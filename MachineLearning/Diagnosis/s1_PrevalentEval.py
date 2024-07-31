
import glob
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import random
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve
from joblib import Parallel, delayed
warnings.filterwarnings('error')

def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c.replace("_","")) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l


def threshold(array, cutoff):
    array1 = array.copy()
    array1[array1 < cutoff] = 0
    array1[array1 >= cutoff] = 1
    return array1


def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t['threshold'])


def get_eval(y_test, pred_prob, cutoff):
    pred_binary = threshold(pred_prob, cutoff)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_binary).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    prec = tp / (tp + fp)
    Youden = sens + spec - 1
    f1 = 2 * prec * sens / (prec + sens)
    auc = roc_auc_score(y_test, pred_prob)
    evaluations = np.round((auc, acc, sens, spec, prec, Youden, f1), 5)
    evaluations = pd.DataFrame(evaluations).T
    evaluations.columns = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'Youden-index', 'F1-score']
    return evaluations


def convert_output(result_df):
    result_df = result_df.T
    result_df['Median'] = result_df.median(axis=1)
    result_df['LBD'] = result_df.quantile(0.025, axis=1)
    result_df['UBD'] = result_df.quantile(0.975, axis=1)
    output_lst = []
    for i in range(7):
        output_lst.append('{:.3f}'.format(result_df['Median'][i]) + ' [' +
                          '{:.3f}'.format(result_df['LBD'][i]) + ' - ' +
                          '{:.3f}'.format(result_df['UBD'][i]) + ']')
    result_df['output'] = output_lst
    myout = result_df.T
    return myout.iloc[-1, :]

def get_iter_output(mydf, gt_col, pred_lst, ct_lst, my_iter):
    tmp_random = np.random.RandomState(my_iter)
    bt_idx = tmp_random.choice(range(len(mydf)), size=len(mydf), replace=True)
    mydf_bt = mydf.copy()
    mydf_bt = mydf_bt.iloc[bt_idx, :]
    mydf_bt.reset_index(inplace=True, drop=True)
    y_test_bt = mydf_bt[gt_col]
    cov_iter = get_eval(y_test_bt, mydf_bt[pred_lst[0]], ct_lst[0])
    nmr_iter = get_eval(y_test_bt, mydf_bt[pred_lst[1]], ct_lst[1])
    top_nmr_iter = get_eval(y_test_bt, mydf_bt[pred_lst[2]], ct_lst[2])
    nmr_cov_iter = get_eval(y_test_bt, mydf_bt[pred_lst[3]], ct_lst[3])
    top_nmr_cov_iter = get_eval(y_test_bt, mydf_bt[pred_lst[4]], ct_lst[4])
    return (cov_iter, nmr_iter, top_nmr_iter, nmr_cov_iter, top_nmr_cov_iter)

nb_cpus = 40

dpath = '/home1/jiayou/Documents/Projects/MetaAtlas/'
#dpath = '/Volumes/JasonWork/Projects/MetaAtlas/'

tgt_dir_lst = sort_nicely(glob.glob(dpath + 'Results/Prediction/Prevalent/Predictions/*.csv'))

for tgt_dir in tqdm(tgt_dir_lst):
    tgt = os.path.basename(tgt_dir)[:-4]
    tgt_pred_df = pd.read_csv(tgt_dir)
    ct_cov = Find_Optimal_Cutoff(tgt_pred_df.target_y, tgt_pred_df.y_pred_cov)[0]
    ct_nmr = Find_Optimal_Cutoff(tgt_pred_df.target_y, tgt_pred_df.y_pred_nmr)[0]
    ct_top_nmr = Find_Optimal_Cutoff(tgt_pred_df.target_y, tgt_pred_df.y_pred_top_nmr)[0]
    ct_nmr_cov = Find_Optimal_Cutoff(tgt_pred_df.target_y, tgt_pred_df.y_pred_nmr_cov)[0]
    ct_top_nmr_cov = Find_Optimal_Cutoff(tgt_pred_df.target_y, tgt_pred_df.y_pred_top_nmr_cov)[0]
    ct_lst = [ct_cov, ct_nmr, ct_top_nmr, ct_nmr_cov, ct_top_nmr_cov]
    pred_lst = ['y_pred_cov', 'y_pred_nmr', 'y_pred_top_nmr', 'y_pred_nmr_cov', 'y_pred_top_nmr_cov']
    iter_eval_lst = Parallel(n_jobs=nb_cpus)(delayed(get_iter_output)(tgt_pred_df, 'target_y', pred_lst, ct_lst, my_iter) for my_iter in range(500))
    cov_df, nmr_df, top_nmr_df, nmr_cov_df, top_nmr_cov_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for iter_eval in iter_eval_lst:
        cov_df = pd.concat([cov_df, iter_eval[0]], axis=0)
        nmr_df = pd.concat([nmr_df, iter_eval[1]], axis=0)
        top_nmr_df = pd.concat([top_nmr_df, iter_eval[2]], axis=0)
        nmr_cov_df = pd.concat([nmr_cov_df, iter_eval[3]], axis=0)
        top_nmr_cov_df = pd.concat([top_nmr_cov_df, iter_eval[4]], axis=0)
    res_cov, res_nmr, res_top_nmr = convert_output(cov_df), convert_output(nmr_df), convert_output(top_nmr_df)
    res_nmr_cov, res_top_nmr_cov = convert_output(nmr_cov_df), convert_output(top_nmr_cov_df)
    res_df = pd.concat([res_cov, res_nmr, res_top_nmr, res_nmr_cov, res_top_nmr_cov], axis=1)
    res_df = res_df.T
    res_df.index = ['Demographic', 'NMR', 'TopNMR', 'NMR+Demographic', 'TopNMR+Demographic']
    res_df.to_csv(dpath + 'Results/Prediction/Prevalent/Evaluation/' + tgt + '.csv', index=True)












#!/bin/bash
#SBATCH -p amd         # Queue
#SBATCH -N 1          # Node count required for the job
#SBATCH -n 40           # Number of tasks to be launched
#SBATCH --mem=128G
#SBATCH -J p_eval           # Job name
#SBATCH -o p_eval.out       # Standard output
#SBATCH -w amdnode3

cd /home1/jiayou/Documents/Projects/MetaAtlas/SeverCode/Prediction/Evaluation/
python eval_prevalent.py

