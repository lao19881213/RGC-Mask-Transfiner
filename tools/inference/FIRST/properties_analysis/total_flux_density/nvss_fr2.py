import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.wcs as wcs
from astropy import coordinates
import astropy.units as u
import pandas as pd
import numpy as np
import matplotlib as mpl
import os
import json
import matplotlib.ticker as mticker

def find_bbox_flux(bbox, fitsfile):
    hdu = fits.open(fitsfile)[0]

    # Set any NaN areas to zero or the interpolation will fail
    hdu.data[np.isnan(hdu.data)] = 0.0

    # Get vital stats of the fits file
    bmaj = hdu.header["BMAJ"]
    bmin = hdu.header["BMIN"]
    bpa = hdu.header["BPA"]
    xmax = hdu.header["NAXIS1"]
    ymax = hdu.header["NAXIS2"]
    try:
        pix2deg = hdu.header["CD2_2"]
    except KeyError:
        pix2deg = hdu.header["CDELT2"]
    # Montaged images use PC instead of CD
    if pix2deg == 1.0:
        pix2deg = hdu.header["PC2_2"]
    beamvolume = (1.1331 * bmaj * bmin)
    x1 = float(bbox.split('-')[0])
    y1 = float(bbox.split('-')[1])
    x2 = float(bbox.split('-')[2])
    y2 = float(bbox.split('-')[3])

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    #print(hdu.data.shape)
    box_data = hdu.data[hdu.data.shape[0]-y2:hdu.data.shape[0]-y1,x1:x2]
    int_flux = np.sum(box_data) #Jy/pix
    int_flux = int_flux * (pix2deg**2) / beamvolume #Jy
    return int_flux


hetu_csv = pd.read_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_fr2_flux_fixed.csv')

ras = hetu_csv['centre_ra'].values
#decs = hetu_csv['centre_dec'].values
#boxs = hetu_csv['box'].values
#labels = hetu_csv['label'].values
source_names = hetu_csv['source_name'].values
FIRST_int_flux = hetu_csv['int_flux'].values

#FIRST_fits = '/home/data0/lbq/inference_data/FIRST_fits'
#FIRST_bkg = '/home/data0/lbq/inference_data/FIRST_rms'

NVSS_bkg_dir = '/home/data0/lbq/inference_data/NVSS_bkg'


NVSS_fits_dir = '/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/FRII/NVSS4flux'
NVSS_json_dir = '/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/FRII/NVSS4flux_final'

fitsfs = os.listdir(NVSS_fits_dir)

FIRST_int_flux_s = []
NVSS_int_flux_s = []
for m in range(len(fitsfs)):
      source_name = os.path.splitext(fitsfs[m])[0].split('_')[1] 
      
      NVSS_fits = '%s/NVSS_%s.fits' % (NVSS_fits_dir, source_name)

      for mm in range(len(ras)):
          if source_name == source_names[mm]:
             FIRST_int_flux_s.append(FIRST_int_flux[mm])
             json_file = '%s/NVSS_%s.json' % (NVSS_json_dir, source_name)
             with open(json_file,'r', encoding='utf-8')as f:
                  json_data = json.load(f)  
                  obj = json_data['shapes'][0] 
                  x1 = obj['points'][0][0]
                  y1 = obj['points'][0][1]
                  x2 = obj['points'][1][0]
                  y2 = obj['points'][1][0]
                  box = '{:.5f}-{:.5f}-{:.5f}-{:.5f}'.format(float(x1), float(y1), float(x2), float(y2))

             total_flux = find_bbox_flux(box, NVSS_fits)
             bkg_fits = os.path.join(NVSS_bkg_dir, 'NVSS_%s_bkg.fits' % source_name)
             bkg_flux = find_bbox_flux(box, bkg_fits)
             final_flux = total_flux - bkg_flux
             NVSS_int_flux_s.append(final_flux)
             if final_flux/FIRST_int_flux[mm] <= 1.0:
                print(source_name, final_flux/FIRST_int_flux[mm])
             break 

FIRST_int_flux_s = np.array(FIRST_int_flux_s)
NVSS_int_flux_s = np.array(NVSS_int_flux_s)

plt.figure(1)
ax1 = plt.gca()
x=np.linspace(0,np.max(NVSS_int_flux_s*1000.0),100)
for mm in range(len(FIRST_int_flux_s)): 
    plt.plot(FIRST_int_flux_s[mm]*1000.0, NVSS_int_flux_s[mm]*1000.0, '+', c='b', markersize=5, alpha=0.5, linewidth=0.5) #'tab:blue')

plt.plot(x,x,'r--')

plt.tick_params(labelsize=20)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.tick_params(which='major', length=8,direction='in')
plt.tick_params(which='minor', length=4,direction='in')
ax1.tick_params(axis='both', which='both', width=1,direction='in')
ax1.tick_params(axis = 'x', which = 'major', labelsize = 14, pad=4,direction='in')
ax1.tick_params(axis = 'y', which = 'major', labelsize = 14, pad=4,direction='in')

plt.xlabel(r'FIRST flux density (mJy)',fontsize=16,labelpad=0)
plt.ylabel(r'NVSS flux density (mJy)',fontsize=16,labelpad=0)

ax1.set_xscale('log')
ax1.set_yscale('log')

#plt.xlim([3,np.max(NVSS_int_flux_s)*1000.+10])
#plt.ylim([3,np.max(NVSS_int_flux_s)*1000.+10])

#ax1.set_aspect('equal')

ticks_loc = ax1.get_yticks().tolist()
#print(ticks_loc, ax1.get_xticks().tolist())
ticks_loc = [0.1, 1, 10, 100, 1000, 10000]
ax1.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
ax1.set_yticklabels(ticks_loc)
ax1.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
ax1.set_xticklabels(ticks_loc)
ax1.set_xlim([4,10000])
ax1.set_ylim([4,10000])

ax1.set_aspect('equal')
plt.tight_layout()
plt.savefig('FIRSTvsNVSS_flux.png')
plt.savefig('FIRSTvsNVSS_flux.pdf')

FIRST_results = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_fr2_sdss_ned_flux_fixed_vlass.csv'

hetu_csv = pd.read_csv(FIRST_results)

LAS = hetu_csv['LAS'].values
source_names = hetu_csv['source_name'].values
LAS_NVSS = []
for m in range(len(fitsfs)):
      source_name = os.path.splitext(fitsfs[m])[0].split('_')[1]
      for mm in range(len(LAS)):
          if source_name == source_names[mm]:
             LAS_NVSS.append(LAS[mm])
             break


LAS_NVSS = np.array(LAS_NVSS)
ratio_flux = NVSS_int_flux_s/FIRST_int_flux_s
#print(ratio_flux.shape, LAS_NVSS.shape)

plt.figure(2)
ax2 = plt.gca()

plt.plot(LAS_NVSS, ratio_flux, '+', c='b', markersize=5, alpha=0.5, linewidth=0.5) #'tab:blue')

plt.tick_params(labelsize=20)
labels = ax2.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.tick_params(which='major', length=8,direction='in')
plt.tick_params(which='minor', length=4,direction='in')
ax2.tick_params(axis='both', which='both', width=1,direction='in')
ax2.tick_params(axis = 'x', which = 'major', labelsize = 14, pad=4,direction='in')
ax2.tick_params(axis = 'y', which = 'major', labelsize = 14, pad=4,direction='in')

plt.xlabel(r'LAS (arcminute)',fontsize=16,labelpad=0)
plt.ylabel(r'Flux ratio',fontsize=16,labelpad=0)


plt.xlim([0,5])
#plt.ylim([3,np.max(NVSS_int_flux_s)*1000.+10])

#ax1.set_aspect('equal')

plt.savefig('LAS_flux_ratio.png')
