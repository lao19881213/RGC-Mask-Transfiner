
import pandas as pd
import numpy as np

vlass_rms = 120E-6 #120 μJy
first_rms = 0.15E-3 #0.15 mJy
nvss_rms =  0.45E-3 #0.45 mJy
hetu_csv = pd.read_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_ht_flux_fixed_vlass.csv')

FIRST_flux = hetu_csv['int_flux'].values
NVSS_flux = hetu_csv['nvss_flux'].values
VLASS_flux =hetu_csv['vlass_flux'].values
S_1_4GHz = []
S_3GHz = []
print(len(FIRST_flux))
for mm in range(len(FIRST_flux)):
    print(mm)
    if FIRST_flux[mm] <= 0.0:
       FIRST_flux_s  = first_rms
    else:
       FIRST_flux_s  = FIRST_flux[mm]

    if NVSS_flux[mm] <= 0.0:
       NVSS_flux_s  = nvss_rms     
    else:
       NVSS_flux_s  = NVSS_flux[mm]   

    if VLASS_flux[mm] <= vlass_rms:
       VLASS_flux_s  = np.nan #'--'#vlass_rms     
    else:
       VLASS_flux_s  = VLASS_flux[mm]

    if FIRST_flux_s > NVSS_flux_s:
       S_1_4GHz.append(FIRST_flux_s)
    else:
       S_1_4GHz.append(NVSS_flux_s)

    S_3GHz.append(VLASS_flux_s)



hetu_csv['S_1.4GHz'] = S_1_4GHz
hetu_csv['S_3GHz'] = S_3GHz 
hetu_csv.to_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_ht_flux_fixed_vlass.csv', index = False)


final_csv = pd.read_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_ht_flux_fixed_vlass.csv')

FIRST_flux = final_csv['int_flux'].values
NVSS_flux = final_csv['nvss_flux'].values
VLASS_flux = final_csv['vlass_flux'].values

bad_id = []
for mm in range(len(FIRST_flux)):
    if FIRST_flux[mm] <= 3*first_rms or NVSS_flux[mm] <= 3*nvss_rms:
       bad_id.append(mm) 

bad_id_final = list(set(bad_id))
final_csv.drop(bad_id_final, inplace=True)

final_csv.to_csv("/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_ht_flux_fixed_vlass.csv",index=False,sep=',')

final1_csv = pd.read_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_ht_flux_fixed_vlass.csv')
objns = final1_csv['source_name'].values
duplicate_id = []

for mm in range(len(objns)):
    name = objns[mm]
    print(mm)
    for nn in range(mm+1,len(objns)):
        if objns[nn] == name :
           duplicate_id.append(nn)


duplicate_id_final = list(set(duplicate_id))
final1_csv.drop(duplicate_id_final, inplace=True)

final1_csv.to_csv("/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_ht_flux_fixed_vlass.csv",index=False,sep=',')
