import pandas as pd
import numpy as np
import os,sys,shutil,re

FIRST_result = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FRIIRGcat_final.csv'

hetu_csv = pd.read_csv(FIRST_result)

RA = hetu_csv['R.A.'].values
Dec = hetu_csv['Decl.'].values
Ref = hetu_csv['Ref.'].values

ned_result = '/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/NED/fr2/info_fr2_flag.csv'
ned_csv = pd.read_csv(ned_result)
ned_names = ned_csv['source_name'].values
z_fs = ned_csv['Redshift_flag'].values

mn = 0
for mm in range(len(RA)):
    RA_s = RA[mm].replace(' ','')
    Dec_s = Dec[mm].replace(' ','')
    Ref_s = Ref[mm]
    source_name = Ref_s + ' J' + RA_s + Dec_s
    #print(source_name)
    for nn in range(len(ned_names)):
        if ned_names[nn] == source_name:
           mn = mn + 1
           print(source_name, mm, mn)
           if z_fs[nn] == 'SUN' or z_fs[nn] == 'SST' or z_fs[nn] == 'SMU' or z_fs[nn] == 'SLS' or z_fs[nn] == 'S1L': 
              hetu_csv.loc[mm,'f_z'] = 's'
           else:
              hetu_csv.loc[mm,'f_z'] = 'p'

hetu_csv.to_csv('FRIIRGcat_final_fixed.csv', index = False)
