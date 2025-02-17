
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
col1, col2, col3 = 'deepskyblue', 'yellowgreen', 'firebrick'
y_pred_col1, y_pred_col2, y_pred_col3 = 'y_pred_cov', 'y_pred_nmr', 'y_pred_nmr_cov'

dpath = '/Volumes/JasonWork/Projects/MetaAtlas/'
tgt_dir_lst = sort_nicely(glob.glob(dpath + 'Revision/Results/Prediction/Incident/Predictions/*.csv'))
tgt_dict = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv', encoding='latin-1')
cov_df = pd.read_csv(dpath + 'Data/Covariates/Covariates.csv', usecols = ['eid', 'Region_code'])
region_fold_id_lst = list(set(cov_df.Region_code))

#tgt_dir = '/Volumes/JasonWork/Projects/MetaAtlas/Revision/Results/Prediction/Incident/Predictions/E4_DM2.csv'

for tgt_dir in tqdm(tgt_dir_lst):
    tgt = os.path.basename(tgt_dir)[:-4]
    tgt_name = tgt_dict.loc[tgt_dict.NAME == tgt].LONGNAME.iloc[0]
    output_img = dpath + 'Revision/Results/Prediction/Incident/AUC_plot/' + tgt + '.pdf'
    eval_df = pd.read_csv(dpath + 'Revision/Results/Prediction/Incident/Evaluation/' + tgt + '.csv')
    legend1 = 'Demographic              : ' + eval_df.AUC.iloc[0]
    legend2 = 'MetRS                         : ' + eval_df.AUC.iloc[1]
    legend3 = 'MetRS+Demographic : ' + eval_df.AUC.iloc[2]
    tgt_pred_df = pd.read_csv(tgt_dir)
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.rcParams["font.family"] = "Arial"
    tprs1_lst, tprs2_lst, tprs3_lst = [], [], []
    base_fpr = np.linspace(0, 1, 101)
    for fold_id in region_fold_id_lst:
        try:
            test_idx = tgt_pred_df['Region_code'].index[tgt_pred_df['Region_code'] == fold_id]
            y_true = tgt_pred_df.iloc[test_idx].target_y
            y_pred1, y_pred2, y_pred3 = tgt_pred_df.iloc[test_idx].y_pred_cov, tgt_pred_df.iloc[test_idx].y_pred_nmr, \
            tgt_pred_df.iloc[test_idx].y_pred_nmr_cov
            fpr1, tpr1, _ = roc_curve(y_true, y_pred1)
            fpr2, tpr2, _ = roc_curve(y_true, y_pred2)
            fpr3, tpr3, _ = roc_curve(y_true, y_pred3)
            # plt.plot(fpr, tpr, col1, alpha=0.25)
            tpr1, tpr2, tpr3 = np.interp(base_fpr, fpr1, tpr1), np.interp(base_fpr, fpr2, tpr2), np.interp(base_fpr, fpr3, tpr3)
            tpr1[0], tpr2[0], tpr3[0] = 0, 0, 0
            tprs1_lst.append(tpr1)
            tprs2_lst.append(tpr2)
            tprs3_lst.append(tpr3)
        except:
            pass
    tprs1, tprs2, tprs3 = pd.DataFrame(tprs1_lst), pd.DataFrame(tprs2_lst), pd.DataFrame(tprs3_lst)
    mean_tprs1, mean_tprs2, mean_tprs3 = tprs1.mean(axis=0), tprs2.mean(axis=0), tprs3.mean(axis=0)
    std1, std2, std3 = tprs1.std(axis=0), tprs2.std(axis=0), tprs3.std(axis=0)
    tprs_upper1, tprs_upper2, tprs_upper3 = np.minimum(mean_tprs1 + 2*std1, 1), np.minimum(mean_tprs2 + 2*std2, 1), np.minimum(mean_tprs3 + 2*std3, 1)
    tprs_lower1, tprs_lower2, tprs_lower3 = mean_tprs1 - 2 * std1, mean_tprs2 - 2 * std2, mean_tprs3 - 2 * std3,
    plt.plot(base_fpr, mean_tprs1, col1, linewidth=mywidth, label=legend1)
    plt.fill_between(base_fpr, tprs_lower1, tprs_upper1, color=col1, alpha=0.1)
    plt.plot(base_fpr, mean_tprs2, col2, linewidth=mywidth, label=legend2)
    plt.fill_between(base_fpr, tprs_lower2, tprs_upper2, color=col2, alpha=0.1)
    plt.plot(base_fpr, mean_tprs3, col3, linewidth=mywidth, label=legend3)
    plt.fill_between(base_fpr, tprs_lower3, tprs_upper3, color=col3, alpha=0.1)
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


