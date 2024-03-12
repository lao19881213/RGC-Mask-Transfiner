import os,sys
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd


ht_csv = pd.read_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_ht_flux_fixed_vlass.csv')
source_names = ht_csv['source_name'].values

ht_check_csv = pd.read_csv('/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/HT/first_final.csv')

source_names_check = ht_check_csv['source_name'].values

lower_id = []
for n in range(len(source_names)):
    if source_names[n] in source_names_check:
       print('%s is a good candidate!' % source_names[n])
    else:
       lower_id.append(n)

lower_id_final = list(set(lower_id))
ht_csv.drop(lower_id_final, inplace=True)

ht_csv.to_csv("/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_ht_flux_fixed_vlass_rm_lower.csv",index=False,sep=',')
