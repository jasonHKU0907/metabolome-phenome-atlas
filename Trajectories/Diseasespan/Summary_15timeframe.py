
import glob
import numpy as np
import pandas as pd
import os
import re
import warnings
warnings.filterwarnings('error')


dpath = '/Volumes/JasonWork/Projects/MetaAtlas/'

mydf = pd.read_csv(dpath + 'Results/Trajectory/DiseaseSpan/temporal_pval_15frames.csv', low_memory=False)
nmr_df = pd.read_csv(dpath + 'Data/MetaData/NMR_Preprocessed.csv')
nmr_dict_df = pd.read_csv(dpath + 'Data/MetaData/NMR_Dict_final.csv', encoding='latin-1', usecols = ['Metabolite_code', 'plot_code', 'Metabolite', 'Group1'])

tmpdf = pd.DataFrame({'Metabolite_code': nmr_df.columns.tolist()[1:]})

mydf_over10yrs, mydf_5_10yrs, mydf_within5yrs = mydf.copy(), mydf.copy(), mydf.copy()
mydf_over10yrs = mydf_over10yrs.loc[mydf_over10yrs.sig_yrs2>10]
mydf_5_10yrs = mydf_5_10yrs.loc[(mydf_5_10yrs.sig_yrs2<=10)&(mydf_5_10yrs.sig_yrs2>=5)]
mydf_within5yrs = mydf_within5yrs.loc[mydf_within5yrs.sig_yrs2<5]

tmp_over10yrs = mydf_over10yrs.NMR.value_counts()
tmp_over10yrs = pd.DataFrame({'Metabolite_code':tmp_over10yrs.index, 'over10yrs':tmp_over10yrs.tolist()})

tmp_5_10yrs = mydf_5_10yrs.NMR.value_counts()
tmp_5_10yrs = pd.DataFrame({'Metabolite_code':tmp_5_10yrs.index, '5_10yrs':tmp_5_10yrs.tolist()})

tmp_within5yrs = mydf_within5yrs.NMR.value_counts()
tmp_within5yrs = pd.DataFrame({'Metabolite_code':tmp_within5yrs.index, 'within5yrs':tmp_within5yrs.tolist()})


tmpdf = pd.merge(tmpdf, tmp_over10yrs, how = 'left', on = ['Metabolite_code'])
tmpdf = pd.merge(tmpdf, tmp_5_10yrs, how = 'left', on = ['Metabolite_code'])
tmpdf = pd.merge(tmpdf, tmp_within5yrs, how = 'left', on = ['Metabolite_code'])

tmpdf = pd.merge(tmpdf, nmr_dict_df, how = 'left', on = ['Metabolite_code'])

tmpdf.to_csv(dpath + 'Results/Trajectory/DiseaseSpan/NMR_sig_yrs_15frames.csv', index = False)

