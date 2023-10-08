import pandas as pd
from astropy.coordinates import SkyCoord
import numpy as np

cat_csv = pd.read_csv('GRGs.csv')

RAs = cat_csv['R.A.'].values
DEs = cat_csv['Decl.'].values

ra_final = []
dec_final = []
#RAh,RAm,RAs,DE-,DEd,DEm,DEs
for mm in range(len(RAs)):
    #print(DE_symbol[mm]
    c_name = "%sh%sm%ss %sd%sm%ss" % (RAs[mm].split(' ')[0], RAs[mm].split(' ')[1], RAs[mm].split(' ')[2], DEs[mm].split(' ')[0], DEs[mm].split(' ')[1], DEs[mm].split(' ')[2])
    print(c_name)
    c = SkyCoord(c_name, frame='fk5')
    ra_final.append(float(c.ra.value))
    dec_final.append(float(c.dec.value)) 

cat_csv['RA_deg'] = ra_final
cat_csv['DEC_deg'] = dec_final

cat_csv.to_csv("GRGs.csv",index=False,sep=',') 
