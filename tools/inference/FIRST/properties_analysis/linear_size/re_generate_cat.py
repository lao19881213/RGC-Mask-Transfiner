import pandas as pd
from astropy.coordinates import SkyCoord
import numpy as np

cat_csv = pd.read_csv('J_A+A_642_A153.csv')

RAh = cat_csv['RAh'].values
RAm = cat_csv['RAm'].values
RAs = cat_csv['RAs'].values
DE_symbol = cat_csv['DE-'].values
DEd = cat_csv['DEd'].values
DEm = cat_csv['DEm'].values
DEs = cat_csv['DEs'].values

ra_final = []
dec_final = []
#RAh,RAm,RAs,DE-,DEd,DEm,DEs
for mm in range(len(RAh)):
    #print(DE_symbol[mm])
    if DE_symbol[mm] != '-':
    #if np.isnan(DE_symbol[mm]):
       DE_symbol[mm] = '+'
    c_name = "%sh%sm%ss %s%sd%sm%ss" % (RAh[mm],RAm[mm],RAs[mm],DE_symbol[mm],DEd[mm],DEm[mm],DEs[mm])
    print(c_name)
    c = SkyCoord(c_name, frame='fk5')
    ra_final.append(float(c.ra.value))
    dec_final.append(float(c.dec.value)) 

cat_csv['RA_deg'] = ra_final
cat_csv['DEC_deg'] = dec_final

cat_csv.to_csv("J_A+A_642_A153.csv",index=False,sep=',') 
