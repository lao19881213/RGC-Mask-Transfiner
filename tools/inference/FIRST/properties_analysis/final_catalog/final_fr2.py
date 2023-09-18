import pandas as pd
import numpy as np
import os,sys,shutil,re
from astropy import coordinates
import astropy.units as u
import math
from astropy.cosmology import FlatLambdaCDM  #for luminosity distance

#Planck 2018 results. VI. Cosmological parameters
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)
#cosmo = LambdaCDM(H0=67.4, Om0=0.315) #, Ode0=0.7)

def linear_size(z, alpha_arcsec): 
#Inputs:
#z->redshift
#alpha_arcsec -> angular size of source in arcsec
#Function to estimate cosmological scale in kpc/arcsec or pc/mas unit for the given redshift of source:
# The output of this function can be verified using the Cosmology calculator webpage:
#    http://www.astro.ucla.edu/~wright/CosmoCalc
# The cosmological parameters taken here are generally used ones. If you know the cosmological parameters for your case, use them in the following function for better accuracy.
    # Cosmological parameters:
    H0 = 67.4         # Hubble constant in units of km/s/Mpc
    WM = 0.315        # Omega(matter)
    universe='flat' # Assuming flat Universe cosmology
    # Omega(vacuum) or lambda is specified according to the Universe type:
    if universe=='open':
        WV = 0.0
    elif universe=='flat':
        WV = 1.0 - WM
    else:
        WV = 1.0 - WM - 0.4165/(H0*H0)   # Universe in general or user specified value

    # Initialize constants:
    WR = 0.         # Omega(radiation)
    WK = 0.         # Omega curvature = 1-Omega(total)
    c = 299792.458  # velocity of light in km/sec
    DCMR = 0.0      # comoving radial distance in units of c/H0
    DA = 0.0        # angular size distance
    DA_Mpc = 0.0
    kpc_DA = 0.0
    a = 1.0         # 1/(1+z), the scale factor of the Universe
    az = 0.5        # 1/(1+z(source))
    h = H0/100.
    WR = 4.165E-5/(h*h)   # includes 3 massless neutrino species, T0 = 2.72528
    WK = 1-WM-WR-WV
    az = 1.0/(1+1.0*z)
    n=1000          # number of points in integrals
    DCMR = 0.0

    # do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
    for i in range(n):
        a = az+(1-az)*(i+0.5)/n
        adot = np.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
        DCMR = DCMR + 1./(a*adot)
    DCMR = (1.-az)*DCMR/n

    # tangential comoving distance
    ratio = 1.00
    x = np.sqrt(abs(WK))*DCMR
    if x > 0.1:
        if WK > 0:
            ratio =  0.5*(exp(x)-exp(-x))/x
        else:
            ratio = sin(x)/x
    else:
        y = x*x
    if WK < 0: y = -y
    ratio = 1. + y/6. + y*y/120.
    DCMT = ratio*DCMR
    DA = az*DCMT
    DA_Mpc = (c/H0)*DA
    kpc_DA = DA_Mpc/206.264806    #scale in kpc/arcsec or pc/mas
    #This is cosmological scale conversion factor which can be used to convert angular distance into linear distance.
    d_pc = kpc_DA * alpha_arcsec
    return d_pc


FIRST_result = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_fr2_sdss_ned_flux_fixed_vlass_optical_spectra.csv'

hetu_csv = pd.read_csv(FIRST_result)

objns = hetu_csv['source_name'].values
ras = hetu_csv['centre_ra'].values
decs = hetu_csv['centre_dec'].values
image_filename = hetu_csv['image_filename'].values
sdss_names = hetu_csv['SDSS16'].values
NED_Redshifts = hetu_csv['NED_Redshift']
spec_z = hetu_csv['z']
photo_z = hetu_csv['photo_z']
LASs = hetu_csv['LAS'].values
S_1_4 = hetu_csv['S_1.4GHz'].values
S_3 = hetu_csv['S_3GHz'].values
alpha_3v1_4 = hetu_csv['alpha_3v1.4'].values
NED_name = hetu_csv['NED_name'].values
NED_RA = hetu_csv['NED_RA'].values
NED_DEC = hetu_csv['NED_DEC'].values

FIRST_NED = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_fr2_ned_optical_IR_final.csv'
ned_csv = pd.read_csv(FIRST_NED)
NED_name_IR = ned_csv['NED_name'].values
 
name_final = []
RA_final = []
DEC_final = []
Ref_final = []
z_final = []
f_z_final = []
LAS_final = []
LLS_final = []
S_1_4_final = []
S_3_final = []
alpha_final = []
for mm in range(len(objns)):
    print(objns[mm])
    if not pd.isnull(sdss_names[mm]): 
       short_name = sdss_names[mm].split(' ')[1][0:5] + sdss_names[mm].split(' ')[1][10:15]
       name_final.append(short_name)
       ra_h = sdss_names[mm].split(' ')[1][1:3]
       ra_m = sdss_names[mm].split(' ')[1][3:5]
       ra_s = sdss_names[mm].split(' ')[1][5:10]
       RA_final.append('%s %s %s' % (ra_h, ra_m, ra_s))
       dec_d = sdss_names[mm].split(' ')[1][10:13]
       dec_m = sdss_names[mm].split(' ')[1][13:15]
       dec_s = sdss_names[mm].split(' ')[1][15:19]
       DEC_final.append('%s %s %s' % (dec_d, dec_m, dec_s))
       Ref_final.append('SDSS')
    elif NED_name[mm] in NED_name_IR:
       c_ned = coordinates.SkyCoord(ra=NED_RA[mm], dec=NED_DEC[mm], unit=(u.deg, u.deg), frame='fk5')
       c_1 = c_ned.to_string('hmsdms', decimal=False, precision=2, sep=' ')
       name_ra = '%s %s %s' % (c_1.split(' ')[0], c_1.split(' ')[1], c_1.split(' ')[2])
       c_2 = c_ned.to_string('hmsdms', decimal=False, precision=1, sep=' ') 
       name_dec = '%s %s %s' % (c_2.split(' ')[3], c_2.split(' ')[4], c_2.split(' ')[5])
       short_name = 'J%s%s' % (name_ra.replace(' ', '')[0:4], name_dec.replace(' ', '')[0:5])
       name_final.append(short_name)
       RA_final.append(name_ra)
       DEC_final.append(name_dec)
       Ref_final.append('NED') 
    else:
       c_hetu = coordinates.SkyCoord(ra=ras[mm], dec=decs[mm], unit=(u.deg, u.deg), frame='fk5')
       c_hetu_1 = c_hetu.to_string('hmsdms', decimal=False, precision=2, sep=' ')
       hetu_ra = '%s %s %s' % (c_hetu_1.split(' ')[0], c_hetu_1.split(' ')[1], c_hetu_1.split(' ')[2])
       c_hetu_2 = c_hetu.to_string('hmsdms', decimal=False, precision=1, sep=' ')
       hetu_dec = '%s %s %s' % (c_hetu_2.split(' ')[3], c_hetu_2.split(' ')[4], c_hetu_2.split(' ')[5])
       hetu_name = 'J%s%s' % (hetu_ra.replace(' ', '')[0:4], hetu_dec.replace(' ', '')[0:5])
       name_final.append(hetu_name)
       RA_final.append(hetu_ra)
       DEC_final.append(hetu_dec)
       Ref_final.append('HeTu')
     
    if spec_z[mm] != '--' and round(float(spec_z[mm]), 2) > 0.0:
       z_final.append(round(float(spec_z[mm]), 2))
       f_z_final.append('s')
    elif photo_z[mm] != '--' and round(float(photo_z[mm]), 2) > 0.0:
       z_final.append(round(float(photo_z[mm]), 2))
       f_z_final.append('p')
    elif NED_Redshifts[mm] != '--' and (NED_name[mm] in NED_name_IR) and round(float(NED_Redshifts[mm]), 2) > 0.0:
       z_final.append(round(float(NED_Redshifts[mm]), 2))
       f_z_final.append('p')
    else:
       z_final.append(np.nan)
       f_z_final.append(' ')   
    LAS_final.append(round(float(LASs[mm])*60.0))
    if str(S_1_4[mm]) != '--':
       S_1_4_final.append(math.ceil(float(S_1_4[mm])*1000.))
    else:
       S_1_4_final.append(np.nan)
    if str(S_3[mm]) != '--':
       S_3_final.append(math.ceil(float(S_3[mm])*1000.))
    else:
       S_3_final.append(np.nan)
    if str(alpha_3v1_4[mm]) != '--':
       alpha_final.append(round(float(alpha_3v1_4[mm]), 2))
    else:
       alpha_final.append(np.nan)
for nn in range(len(LAS_final)):
    if np.isnan(z_final[nn]):
       LLS_final.append(np.nan)
    else:
       LLS_final.append(round(linear_size(z_final[nn], LAS_final[nn])))      

lum_final = []
#Lrad at 1.4 GHz
for n in range(len(z_final)):
    if z_final[n] > 0.0 and (not np.isnan(S_1_4_final[n])) and (not np.isnan(alpha_final[n])):
       D_L_Mpc = cosmo.luminosity_distance(z_final[n])
       D_L_m = D_L_Mpc.to(u.m).value
       #Lum_1400MHz(W/Hz), total_flux_masked_arr(Jy, 1Jy=10^(-26) W*m^-2*Hz^-1), DL(m), The spectral index (alpha) is typically 0.7.
       Lum_1400MHz = (10**(-26))*(S_1_4_final[n]*10E-3)* 4*math.pi * (D_L_m**2) *(1+z_final[n])**(alpha_final[n]-1)
       lum_final.append(round(np.log10(Lum_1400MHz), 2))
    else:
       lum_final.append(np.nan)

FRIIcat = pd.DataFrame({'Name':name_final,'R.A.':RA_final,'Decl.':DEC_final,'Ref.':Ref_final,\
          'z':z_final,'f_z':f_z_final, 'LAS':LAS_final, 'LLS':LLS_final,\
          'S1.4':S_1_4_final, 'S3':S_3_final,'alpha':alpha_final, 'log(Lrad)':lum_final})

FRIIcat.to_csv("/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FRIIRGcat.csv",index=False,sep=',')
