#!/usr/bin/env python

# This is a program to calculate two-point spectral indexes

import math
import pandas as pd

'''
Program to calculate spectral indices. 
input parameters: S_3GHz S_1.4GHz v_3GHz v_1.4GHz 
'''
hetu_csv = pd.read_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_fr2_flux_fixed_vlass.csv')
S_1_4GHz = hetu_csv['S_1.4GHz'].values
S_3GHz = hetu_csv['S_3GHz'].values
v1 = 3.0 # 3 GHz
v2 = 1.4 # 1.4 GHz

si_all = []
for mm in range(len(S_1_4GHz)):
    if S_3GHz[mm] == '--':
       si_all.append('--')
       #print('...')
    else:
       #print(S_3GHz[mm], S_1_4GHz[mm])
       si=math.log10(float(S_3GHz[mm])/float(S_1_4GHz[mm]))/math.log10(v1/v2)
       
       print("v1 = %4.3f" % v1)
       print("v2 = %4.3f" % v2)
       print("S1  = %4.3f" % float(S_3GHz[mm]))
       print("S2  = %4.3f" % S_1_4GHz[mm])
       print("Spectral index is %4.3f" % si)
       if -si >= 3.6:
          si_all.append('--')
          hetu_csv.loc[mm,'S_3GHz'] = '--'
       else:
          si_all.append(-si)
       
 
hetu_csv['alpha_3v1.4'] = si_all
hetu_csv.to_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_fr2_flux_fixed_vlass.csv', index = False)
