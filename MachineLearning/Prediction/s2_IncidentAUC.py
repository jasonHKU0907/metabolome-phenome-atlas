
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
warnings.filterwarnings('error')


def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c.replace("_","")) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l

mywidth = 5
col1, col2 = 'deepskyblue', 'yellowgreen'
y_pred_col1, y_pred_col2 = 'y_pred_top_nmr', 'y_pred_top_nmr_cov'

dpath = '/Volumes/JasonWork/Projects/MetaAtlas/'
tgt_dir_lst = sort_nicely(glob.glob(dpath + 'Results/Prediction/Incident/Predictions/*.csv'))
tgt_dict = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv', encoding='latin-1')
cov_df = pd.read_csv(dpath + 'Data/Covariates/Covariates.csv', usecols = ['eid', 'Region_code'])
fold_id_lst = [i for i in range(10)]

for tgt_dir in tqdm(tgt_dir_lst):
    tgt = os.path.basename(tgt_dir)[:-4]
    tgt_name = tgt_dict.loc[tgt_dict.NAME == tgt].LONGNAME.iloc[0]
    output_img = dpath + 'Results/Prediction/Incident/AUC_plot/' + tgt + '.pdf'
    eval_df = pd.read_csv(dpath + 'Results/Prediction/Incident/Evaluation/' + tgt + '.csv')
    legend1 = 'MetRS                        : ' + eval_df.AUC.iloc[2]
    legend2 = 'MetRS+Demographic : ' + eval_df.AUC.iloc[4]
    tgt_pred_df = pd.read_csv(tgt_dir)
    tgt_pred_df = pd.merge(tgt_pred_df, cov_df, how='inner', on=['eid'])
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.rcParams["font.family"] = "Arial"
    tprs1, tprs2 = [], []
    base_fpr = np.linspace(0, 1, 101)
    for fold_id in fold_id_lst:
        try:
            test_idx = tgt_pred_df['Region_code'].index[tgt_pred_df['Region_code'] == fold_id]
            y_true = tgt_pred_df.iloc[test_idx].target_y
            y_pred1, y_pred2 = tgt_pred_df.iloc[test_idx].y_pred_top_nmr, tgt_pred_df.iloc[test_idx].y_pred_top_nmr_cov
            fpr1, tpr1, _ = roc_curve(y_true, y_pred1)
            fpr2, tpr2, _ = roc_curve(y_true, y_pred2)
            # plt.plot(fpr, tpr, col1, alpha=0.25)
            tpr1, tpr2 = np.interp(base_fpr, fpr1, tpr1), np.interp(base_fpr, fpr2, tpr2)
            tpr1[0], tpr2[0] = 0, 0
            tprs1.append(tpr1)
            tprs2.append(tpr2)
        except:
            pass
    tprs1, tprs2 = np.array(tprs1), np.array(tprs2)
    mean_tprs1, mean_tprs2 = tprs1.mean(axis=0), tprs2.mean(axis=0)
    std1, std2 = tprs1.std(axis=0), tprs2.std(axis=0)
    tprs_upper1, tprs_upper2 = np.minimum(mean_tprs1 + 2 * std1, 1), np.minimum(mean_tprs2 + 2 * std2, 1)
    tprs_lower1, tprs_lower2 = mean_tprs1 - 2 * std1, mean_tprs2 - 2 * std2,
    plt.plot(base_fpr, mean_tprs1, col1, linewidth=mywidth, label=legend1)
    plt.fill_between(base_fpr, tprs_lower1, tprs_upper1, color=col1, alpha=0.1)
    plt.plot(base_fpr, mean_tprs2, col2, linewidth=mywidth, label=legend2)
    plt.fill_between(base_fpr, tprs_lower2, tprs_upper2, color=col2, alpha=0.1)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.0, 1.0])
    plt.ylim([-0.0, 1.0])
    plt.ylabel('True Positive Rate', fontsize=32, family='Arial')
    plt.xlabel('False Positive Rate', fontsize=32, family='Arial')
    if len(tgt_name) < 45:
        plt.title(tgt_name, fontsize=32, weight='bold', family='Arial')
    elif (len(tgt_name) >= 45) & (len(tgt_name) < 60):
        plt.title(tgt_name, fontsize=24, weight='bold', family='Arial')
    else:
        plt.title(tgt_name, fontsize=16, weight='bold', family='Arial')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=28, family='Arial')
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=28, family='Arial')
    plt.grid(which='minor', alpha=0.2, linestyle=':')
    plt.grid(which='major', alpha=0.5, linestyle='--')
    plt.legend(loc=4, fontsize=26, labelspacing=1.5, facecolor='gainsboro')
    # ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    # ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    plt.tight_layout()
    plt.savefig(output_img)
    plt.close('all')


