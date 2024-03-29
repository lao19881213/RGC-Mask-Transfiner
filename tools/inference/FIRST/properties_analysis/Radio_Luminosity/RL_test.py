import math
import numpy as np
import numpy.ma as ma  # for masking, gets rid of invalid/null redshifts
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
from astropy import units as u
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.cosmology import LambdaCDM  #for luminosity distance
# Ref. Aghanim et al. 2020: H0=67.4 Om0=0.315, Odec0=0.685
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7) #creates an instance of the LambdaCDM model
#cosmo = LambdaCDM(H0=67.4, Om0=0.315, Ode0=0.685)

parser = argparse.ArgumentParser()
parser.add_argument('--nedresult', help='ned results file')
args = parser.parse_args()

ned_table = pd.read_csv(args.nedresult)
hetu_name = ned_table['source_name'].values 
source_name = ned_table['NED_name'].values
redshift = ned_table['Redshift'].values
total_flux = ned_table['int_flux'].values
    

z_arr = redshift
masked_z_arr = ma.masked_array(z_arr, mask=[~(z > 0.0) for z in z_arr])
print(masked_z_arr[1])
lum_dist_list = []  # will make a list of each D_L(z)
for zj in masked_z_arr:  # for each redshift, find their luminosity distance
    lum_dist_list.append(cosmo.luminosity_distance(zj))  # each value of D_L for each corresponding z


D_L = lum_dist_list
#DL = D_L*u.pc.to(u.m)
DL = []
print(D_L[0])
for n in range(len(D_L)):
    try:
       DL.append(D_L[n].to(u.m).value)
    except:
       DL.append(D_L[n])

print(DL[0])
alpha = 0.7 #slope of the radio spectrum in Lum-freq log-log space

total_flux_masked_arr = ma.masked_array(total_flux, mask=[~(z > 0.0) for z in z_arr])

#Lum_1400MHz(W/Hz), total_flux_masked_arr(Jy, 1Jy=10^(-26) W*m^-2*Hz^-1), DL(m), The spectral index (alpha) is typically 0.7.
Lum_1400MHz = (10**(-26))*np.array(total_flux_masked_arr)* 4*math.pi * np.array(DL)**2 *(1+np.array(masked_z_arr))**(alpha-1)

print(np.log10(Lum_1400MHz[0]), np.log10(Lum_1400MHz[1]))
z = masked_z_arr
plt.plot(z, Lum_1400MHz, 'o', markersize=2.0, markerfacecolor='none', markeredgecolor='r', alpha=0.3)
#plt.title('Luminosity of Radio Sources increasing with redshift')
plt.ylabel(r'$L$ (${\rm M} \cdot {\rm Hz}^{-1}$)')
plt.xlabel('Redshift z')
plt.semilogy()
plt.savefig('L_z_test.png')
#plt.savefig('L_z.pdf')
