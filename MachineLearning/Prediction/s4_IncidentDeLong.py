
import glob
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from Utility.Training_Utilities import *
from Utility.DelongTest import delong_roc_test
pd.options.mode.chained_assignment = None  # default='warn'

def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c.replace("_","")) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l

dpath = '/Volumes/JasonWork/Projects/MetaAtlas/'
tgt_dir_lst = sort_nicely(glob.glob(dpath + 'Revision/Results/Prediction/Incident/Predictions/*.csv'))
tgt_dict = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv', encoding='latin-1')

nb_preds = 3
cols_lst = ['y_pred_cov', 'y_pred_nmr', 'y_pred_nmr_cov']

for tgt_dir in tqdm(tgt_dir_lst):
    tgt = os.path.basename(tgt_dir)[:-4]
    outfile = dpath + 'Revision/Results/Prediction/Incident/DeLongStat/'+tgt+'.csv'
    pred_df = pd.read_csv(tgt_dir)
    delong_df = pd.DataFrame(np.zeros((nb_preds, nb_preds)))
    delong_df.columns = cols_lst
    delong_df.index = cols_lst
    for i in range(nb_preds):
        for j in range(nb_preds):
            try:
                tmpdf = pred_df[['target_y', cols_lst[i], cols_lst[j]]]
                tmpdf.dropna(how='any', inplace=True)
                tmpdf.reset_index(inplace=True, drop=True)
                log10_p = delong_roc_test(tmpdf.target_y, tmpdf.iloc[:, 1], tmpdf.iloc[:, 2])
                delong_df.iloc[i, j] = 10 ** log10_p[0][0]
                print(str(i) + ' ' + str(j))
            except:
                delong_df.iloc[i, j] = np.nan
    delong_df.to_csv(outfile, index=True)


