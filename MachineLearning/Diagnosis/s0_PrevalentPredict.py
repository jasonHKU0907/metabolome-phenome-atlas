
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import warnings
from lightgbm import LGBMClassifier
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import random
from itertools import product
import os
warnings.filterwarnings('error')

nb_cpus = 8
nb_params = 24
my_seed = 2024

def select_params_combo(my_dict, nb_items, my_seed):
    combo_list = [dict(zip(my_dict.keys(), v)) for v in product(*my_dict.values())]
    random.seed(my_seed)
    return random.sample(combo_list, nb_items)

def normal_imp(mydict):
    mysum = sum(mydict.values())
    mykeys = mydict.keys()
    for key in mykeys:
        mydict[key] = mydict[key]/mysum
    return mydict

def get_cov_f_lst(tgt2pred_df, tgt):
    sex_id = tgt2pred_df.loc[tgt2pred_df.NAME == tgt].SEX.iloc[0]
    if (sex_id == 1) | (sex_id == 2):
        cov_f_lst = ['Age', 'Race', 'TDI', 'BMI', 'Smoke']
    else:
        cov_f_lst = ['Age', 'Sex', 'Race', 'TDI', 'BMI', 'Smoke']
    return cov_f_lst

def read_target(dpath, tmp_tgt):
    tmp_tgt_df = pd.read_csv(dpath + 'Data/TargetData/TargetData/' + tmp_tgt + '.csv',
                             usecols=['eid', 'target_y', 'BL2Target_yrs'])
    rm_bl_idx = tmp_tgt_df.index[(tmp_tgt_df.target_y == 1) & (tmp_tgt_df.BL2Target_yrs > 0)]
    tmp_tgt_df.drop(rm_bl_idx, axis=0, inplace=True)
    tmp_tgt_df.reset_index(inplace=True, drop=True)
    return tmp_tgt_df

def get_top_nmr_lst(mydf, my_f_lst, my_seed):
    X_train, y_train = mydf[my_f_lst], mydf.target_y
    lgb_nmr = LGBMClassifier(objective='binary', metric='auc', is_unbalance=False, verbosity=-1, seed=my_seed)
    lgb_nmr.fit(X_train, y_train)
    totalgain_imp = lgb_nmr.booster_.feature_importance(importance_type='gain')
    totalgain_imp = dict(zip(lgb_nmr.booster_.feature_name(), totalgain_imp.tolist()))
    tg_imp_df = pd.DataFrame({'NMR_code': list(totalgain_imp.keys()), 'TotalGain': list(totalgain_imp.values())})
    tg_imp_df.sort_values(by = 'TotalGain', inplace = True, ascending = False)
    return tg_imp_df.NMR_code.tolist()[:30], lgb_nmr

def get_best_params(mydf, my_f_lst, inner_cv_fold_id_lst, my_params_lst, my_seed):
    my_params_res_lst = []
    for my_params in my_params_lst:
        auc_cv_lst = []
        my_params0 = my_params.copy()
        for inner_cv_fold_id in inner_cv_fold_id_lst:
            in_cv_X_train = mydf.loc[mydf.in_cv_code != inner_cv_fold_id][my_f_lst]
            in_cv_y_train = mydf.loc[mydf.in_cv_code != inner_cv_fold_id].target_y
            in_cv_X_test = mydf.loc[mydf.in_cv_code == inner_cv_fold_id][my_f_lst]
            in_cv_y_test= mydf.loc[mydf.in_cv_code == inner_cv_fold_id].target_y
            my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=False, verbosity=-1, seed=my_seed)
            my_lgb.set_params(**my_params)
            my_lgb.fit(in_cv_X_train, in_cv_y_train)
            y_pred_prob = my_lgb.predict_proba(in_cv_X_test)[:, 1]
            auc_cv_lst.append(roc_auc_score(in_cv_y_test, y_pred_prob))
        my_params0['AUC_cv_MEAN'] = np.round(np.mean(auc_cv_lst), 5)
        my_params_res_lst.append(my_params0)
    my_params_res_df = pd.DataFrame(my_params_res_lst)
    my_params_res_df.sort_values(by = 'AUC_cv_MEAN', ascending = False, inplace = True)
    best_param = dict(my_params_res_df.iloc[0,:6])
    best_param['n_estimators'] = int(best_param['n_estimators'])
    best_param['max_depth'] = int(best_param['max_depth'])
    best_param['num_leaves'] = int(best_param['num_leaves'])
    return best_param

def model_train_pred(traindf, testdf, f_lst, my_params, my_seed):
    X_train, y_train, X_test = traindf[f_lst], traindf.target_y, testdf[f_lst]
    my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=False, verbosity=-1, seed=my_seed)
    my_lgb.set_params(**my_params)
    calibrate = CalibratedClassifierCV(my_lgb, method='isotonic', cv=5)
    calibrate.fit(X_train, y_train)
    y_pred = my_lgb.predict_proba(X_test)[:, 1].tolist()
    return y_pred

def get_iter_predictions(mydf, nmr_f_lst, cov_f_lst, fold_id, inner_cv_fold_id_lst, candidate_params_lst, my_seed):
    traindf, testdf = mydf.loc[mydf.Region_code != fold_id], mydf.loc[mydf.Region_code == fold_id]
    traindf.reset_index(inplace = True, drop = True)
    testdf.reset_index(inplace = True, drop = True)
    top_nmr_f_lst, lgb_nmr = get_top_nmr_lst(traindf, nmr_f_lst, my_seed)
    params_nmr = get_best_params(traindf, top_nmr_f_lst, inner_cv_fold_id_lst, candidate_params_lst, my_seed)
    params_cov = get_best_params(traindf, cov_f_lst, inner_cv_fold_id_lst, candidate_params_lst, my_seed)
    params_nmr_cov = get_best_params(traindf, top_nmr_f_lst+cov_f_lst, inner_cv_fold_id_lst, candidate_params_lst, my_seed)
    y_pred_nmr = model_train_pred(traindf, testdf, top_nmr_f_lst, params_nmr, my_seed)
    y_pred_cov = model_train_pred(traindf, testdf, cov_f_lst, params_cov, my_seed)
    y_pred_nmr_cov = model_train_pred(traindf, testdf, top_nmr_f_lst+cov_f_lst, params_nmr_cov, my_seed)
    totalgain_imp = lgb_nmr.booster_.feature_importance(importance_type='gain')
    totalgain_imp = dict(zip(lgb_nmr.booster_.feature_name(), totalgain_imp.tolist()))
    tg_imp_cv = Counter(normal_imp(totalgain_imp))
    return (testdf.eid.tolist(), tg_imp_cv, testdf.target_y.tolist(), y_pred_nmr, y_pred_cov, y_pred_nmr_cov)

params_dict = {'n_estimators': [100, 200, 300, 400, 500],
               'max_depth': np.linspace(5, 30, 6).astype('int32').tolist(),
               'num_leaves': np.linspace(5, 30, 6).astype('int32').tolist(),
               'subsample': np.linspace(0.6, 1, 9).tolist(),
               'learning_rate': [0.1, 0.05, 0.01, 0.001],
               'colsample_bytree': np.linspace(0.6, 1, 9).tolist()}

candidate_params_lst = select_params_combo(params_dict, nb_params, my_seed)
candidate_params_lst = candidate_params_lst

dpath = '/home1/jiayou/Documents/Projects/MetaAtlas/'
#dpath = '/Volumes/JasonWork/Projects/MetaAtlas/'

tgt2pred_df = pd.read_csv(dpath + 'Data/TargetData/PrevalentDiseaseTable.csv', encoding='latin-1')
tgt2pred_lst = tgt2pred_df.NAME.tolist()
nmr_df = pd.read_csv(dpath + 'Data/MetaData/NMR_Preprocessed.csv')
nmr_f_lst = nmr_df.columns.tolist()[1:]
cov_df = pd.read_csv(dpath + 'Data/Covariates/Covariates.csv')
nmr_cov_df = pd.merge(nmr_df, cov_df, how = 'inner', on = ['eid'])
region_fold_id_lst = list(set(nmr_cov_df.Region_code))
inner_cv_fold_id_lst = list(set(nmr_cov_df.in_cv_code))

finished_tgt_lst = glob.glob(dpath + 'Revision/Results/Prediction/Prevalent/NMR_Importance/*.csv')
finished_tgt_lst = [os.path.basename(tgt)[:-4] for tgt in finished_tgt_lst]
tgt2pred_lst = [tgt for tgt in tgt2pred_lst if tgt not in finished_tgt_lst]

for tmp_tgt in tqdm(tgt2pred_lst):
    cov_f_lst = get_cov_f_lst(tgt2pred_df, tmp_tgt)
    tmp_tgt_df = read_target(dpath, tmp_tgt)
    tmp_df = pd.merge(tmp_tgt_df, nmr_cov_df, how='left', on=['eid'])
    eid_lst, y_test_lst = [], []
    y_pred_nmr_lst, y_pred_cov_lst, y_pred_nmr_cov_lst = [], [], []
    imp_df = pd.DataFrame({'NMR_code': nmr_f_lst})
    for fold_id in region_fold_id_lst:
        traindf, testdf = tmp_df.loc[tmp_df.Region_code != fold_id], tmp_df.loc[tmp_df.Region_code == fold_id]
        traindf.reset_index(inplace=True, drop=True)
        testdf.reset_index(inplace=True, drop=True)
        top_nmr_f_lst, lgb_nmr = get_top_nmr_lst(traindf, nmr_f_lst, my_seed)
        params_nmr = get_best_params(traindf, top_nmr_f_lst, inner_cv_fold_id_lst, candidate_params_lst, my_seed)
        params_cov = get_best_params(traindf, cov_f_lst, inner_cv_fold_id_lst, candidate_params_lst, my_seed)
        params_nmr_cov = get_best_params(traindf, top_nmr_f_lst+cov_f_lst, inner_cv_fold_id_lst, candidate_params_lst, my_seed)
        y_pred_nmr_lst += model_train_pred(traindf, testdf, top_nmr_f_lst, params_nmr, my_seed)
        y_pred_cov_lst += model_train_pred(traindf, testdf, cov_f_lst, params_cov, my_seed)
        y_pred_nmr_cov_lst += model_train_pred(traindf, testdf, top_nmr_f_lst + cov_f_lst, params_nmr_cov, my_seed)
        eid_lst += testdf.eid.tolist()
        y_test_lst += testdf.target_y.tolist()
        tg_imp = lgb_nmr.booster_.feature_importance(importance_type='gain')
        tg_imp_df = pd.DataFrame({'NMR_code': lgb_nmr.booster_.feature_name(),
                                  'Imp_iter' + str(fold_id): lgb_nmr.booster_.feature_importance(
                                      importance_type='gain')})
        tg_imp_df['Imp_iter' + str(fold_id)] = tg_imp_df['Imp_iter' + str(fold_id)] / tg_imp_df[
            'Imp_iter' + str(fold_id)].sum()
        imp_df = pd.merge(imp_df, tg_imp_df, how='left', on='NMR_code')
    imp_df['Importance'] = imp_df[['Imp_iter' + str(fold_id) for fold_id in region_fold_id_lst]].mean(axis=1)
    imp_df.sort_values(by='Importance', ascending=False, inplace=True)
    pred_df = pd.DataFrame({'eid': eid_lst, 'target_y': y_test_lst, 'y_pred_cov': y_pred_cov_lst,
                            'y_pred_nmr': y_pred_nmr_lst, 'y_pred_nmr_cov': y_pred_nmr_cov_lst})
    pred_df = pd.merge(pred_df, cov_df[['eid', 'Region_code', 'in_cv_code']], how='left', on='eid')
    imp_df.to_csv(dpath + 'Revision/Results/Prediction/Prevalent/NMR_Importance/' + tmp_tgt + '.csv', index=False)
    pred_df.to_csv(dpath + 'Revision/Results/Prediction/Prevalent/Predictions/' + tmp_tgt + '.csv', index=False)


'''
#!/bin/bash
#SBATCH -p DCU         # Queue
#SBATCH -N 1          # Node count required for the job
#SBATCH -n 12           # Number of tasks to be launched
#SBATCH --mem=128G
#SBATCH -J prev_s0          # Job name
#SBATCH -o prev_s0.out       # Standard output
#SBATCH -w node70

cd /home1/jiayou/Documents/Projects/MetaAtlas/Revision/Code/Prediction/Prevalent/
python s0_PrevalentPredict.py
'''


