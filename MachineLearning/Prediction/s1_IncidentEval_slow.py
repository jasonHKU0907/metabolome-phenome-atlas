
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


def get_avg_output(mydf, gt_col, pred_lst, ct_lst, nb_iters):
    nb_ind = len(mydf)
    out_df0, out_df1, out_df2, out_df3, out_df4 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i in tqdm(range(nb_iters)):
        tmp_random = np.random.RandomState(i)
        bt_idx = tmp_random.choice(range(nb_ind), size=nb_ind, replace=True)
        mydf_bt = mydf.copy()
        mydf_bt = mydf_bt.iloc[bt_idx, :]
        mydf_bt.reset_index(inplace = True, drop = True)
        y_test_bt = mydf_bt[gt_col]
        out_df0 = pd.concat([out_df0, get_eval(y_test_bt, mydf_bt[pred_lst[0]], ct_lst[0])], axis=0)
        out_df1 = pd.concat([out_df1, get_eval(y_test_bt, mydf_bt[pred_lst[1]], ct_lst[1])], axis=0)
        out_df2 = pd.concat([out_df2, get_eval(y_test_bt, mydf_bt[pred_lst[2]], ct_lst[2])], axis=0)
        out_df3 = pd.concat([out_df3, get_eval(y_test_bt, mydf_bt[pred_lst[3]], ct_lst[3])], axis=0)
        out_df4 = pd.concat([out_df4, get_eval(y_test_bt, mydf_bt[pred_lst[4]], ct_lst[4])], axis=0)
    my_out0, my_out1, my_out2 = convert_output(out_df0), convert_output(out_df1), convert_output(out_df2)
    my_out3, my_out4 = convert_output(out_df3), convert_output(out_df4)
    return (my_out0, my_out1, my_out2, my_out3, my_out4)



dpath = '/home1/jiayou/Documents/Projects/MetaAtlas/'
dpath = '/Volumes/JasonWork/Projects/MetaAtlas/'

tgt_dir_lst = sort_nicely(glob.glob(dpath + 'Results/Prediction/Incident/Predictions/*.csv'))


import time
start = time.time()

for tgt_dir in tqdm(tgt_dir_lst[:5]):
    tgt = os.path.basename(tgt_dir)[:-4]
    tgt_pred_df = pd.read_csv(tgt_dir)
    ct_cov = Find_Optimal_Cutoff(tgt_pred_df.target_y, tgt_pred_df.y_pred_cov)[0]
    ct_nmr = Find_Optimal_Cutoff(tgt_pred_df.target_y, tgt_pred_df.y_pred_nmr)[0]
    ct_top_nmr = Find_Optimal_Cutoff(tgt_pred_df.target_y, tgt_pred_df.y_pred_top_nmr)[0]
    ct_nmr_cov = Find_Optimal_Cutoff(tgt_pred_df.target_y, tgt_pred_df.y_pred_nmr_cov)[0]
    ct_top_nmr_cov = Find_Optimal_Cutoff(tgt_pred_df.target_y, tgt_pred_df.y_pred_top_nmr_cov)[0]
    ct_lst = [ct_cov, ct_nmr, ct_top_nmr, ct_nmr_cov, ct_top_nmr_cov]
    pred_lst = ['y_pred_cov', 'y_pred_nmr', 'y_pred_top_nmr', 'y_pred_nmr_cov', 'y_pred_top_nmr_cov']
    res_cov, res_nmr, res_top_nmr, res_nmr_cov, res_top_nmr_cov  = get_avg_output(tgt_pred_df, 'target_y', pred_lst, ct_lst, 200)
    res_df = pd.concat([res_cov, res_nmr, res_top_nmr, res_nmr_cov, res_top_nmr_cov], axis=1)
    res_df = res_df.T
    res_df.index = ['Demographic', 'NMR', 'TopNMR', 'NMR+Demographic', 'TopNMR+Demographic']
    res_df.to_csv(dpath + 'Results/Prediction/Incident/Evaluation/' + tgt + '.csv', index=True)

print(time.time()-start)


