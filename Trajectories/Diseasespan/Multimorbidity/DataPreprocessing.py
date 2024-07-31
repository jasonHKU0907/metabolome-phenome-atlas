
import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

dpath = '/Volumes/JasonWork/Projects/MetaAtlas/'
tgt_info_df = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv', usecols = ['NAME', 'LONGNAME', 'Root', 'ICD_10', 'NB_all', 'NB_case', 'Incident_analysis'])
tgt_info_df = tgt_info_df.loc[tgt_info_df.Incident_analysis == 1]
tgt_name_lst = tgt_info_df.NAME.tolist()
tgt_name_lst.sort()

mydf = pd.read_csv(dpath + 'Results/Trajectory/DiseaseSpan/HiearichicalClustering.csv', usecols = ['NAME', 'cls_44'])

for i in range(1, 44):
    cls_name_df = mydf.copy()
    cls_name_df = cls_name_df.loc[cls_name_df.cls_44 == i]
    cls_name_lst = cls_name_df.NAME.tolist()
    cls_df = pd.read_csv(dpath + 'Data/MetaData/NMR_Preprocessed.csv', usecols = ['eid'])
    for cls_name in cls_name_lst:
        tmpdf = pd.read_csv(dpath + 'Data/TargetData/TargetData/' + cls_name + '.csv', usecols=['eid', 'target_y', 'BL2Target_yrs'])
        rm_bl_idx = tmpdf.index[tmpdf.BL2Target_yrs <= 0]
        tmpdf.drop(rm_bl_idx, axis=0, inplace=True)
        #print(tmpdf.target_y.sum())
        tmpdf.reset_index(inplace=True, drop=True)
        tmpdf.rename(columns={'target_y': 'target_y_' + cls_name, 'BL2Target_yrs': 'BL2Target_yrs_' + cls_name}, inplace=True)
        cls_df = pd.merge(cls_df, tmpdf, how='inner', on=['eid'])
    y_lst = ['target_y_' + cls_name for cls_name in cls_name_lst]
    cls_df['target_y_continuous'] = cls_df[y_lst].sum(axis=1)
    nb_y_cls = len(cls_df['target_y_continuous'].value_counts())
    cls_df['target_y_ordinal'] = cls_df['target_y_continuous'].copy()
    if nb_y_cls == 2:
        print('cluster ' + str(i) + ' have ' + str(len(cls_name_lst)) + ' disease: 2 category')
    elif nb_y_cls == 3:
        cls_df.to_csv(dpath + 'Results/Trajectory/DiseaseSpan/Multimorbidity/ClusterData/Cluster_'+ str(i) +'.csv', index = False)
        print('cluster ' + str(i) + ' have ' + str(len(cls_name_lst)) + ' disease: 3 category')
    elif nb_y_cls > 3:
        cls_df['target_y_ordinal'].loc[cls_df['target_y_continuous']>=3] = 3
        cls_df.to_csv(dpath + 'Results/Trajectory/DiseaseSpan/Multimorbidity/ClusterData/Cluster_' + str(i) + '.csv', index=False)
        print('cluster ' + str(i) + ' have ' + str(len(cls_name_lst)) + ' disease: 4 category')
    else:
        pass



