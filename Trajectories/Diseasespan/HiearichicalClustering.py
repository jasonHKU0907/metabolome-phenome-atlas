
import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c.replace("_","")) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l


dpath = '/Volumes/JasonWork/Projects/MetaAtlas/'
tgt_info_df = pd.read_csv(dpath + 'Data/TargetData/DiseaseTable.csv', usecols = ['NAME', 'LONGNAME', 'Root', 'ICD_10', 'NB_all', 'NB_case', 'Incident_analysis'])
tgt_info_df = tgt_info_df.loc[tgt_info_df.Incident_analysis == 1]
tgt_name_lst = tgt_info_df.NAME.tolist()
tgt_name_lst.sort()

my_array = np.load(dpath + 'Results/Trajectory/DiseaseSpan/TrajectoryZscore.npy')
my_array = np.reshape(my_array, (my_array.shape[0], my_array.shape[1]*my_array.shape[2]))
scaler = StandardScaler()
my_X = scaler.fit_transform(my_array)
my_X = pd.DataFrame(my_X)
dist_linkage = hierarchy.ward(my_X)

outdf = pd.DataFrame({'NAME':tgt_name_lst})

ct = 500
nb_cls = 1

while nb_cls<101:
    cluster_ids = hierarchy.fcluster(dist_linkage, ct, criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    nb_cls = len(set(cluster_ids))
    outdf['cls_'+ str(nb_cls)] = cluster_ids
    ct = ct - 0.01

cls_df = pd.merge(tgt_info_df, outdf, how = 'right', on = ['NAME'])
cls_df.to_csv(dpath + 'Results/Trajectory/DiseaseSpan/HiearichicalClustering.csv')


from sklearn.metrics import silhouette_score

out_lst = []
for i in tqdm(range(3, 101)):
    try:
        out_lst.append([i, silhouette_score(my_array, cls_df['cls_' + str(i)], metric="euclidean")])
    except:
        pass


out_df = pd.DataFrame(out_lst)
out_df.to_csv(dpath + 'Results/Trajectory/DiseaseSpan/Silhouette_skl.csv', index = False)





import matplotlib.pyplot as plt

sh_df = pd.read_csv(dpath + 'Results/Trajectory/DiseaseSpan/Silhouette_skl.csv')

x = sh_df.nb_clusters
y = sh_df.silhouette
plt.plot(x, y)

