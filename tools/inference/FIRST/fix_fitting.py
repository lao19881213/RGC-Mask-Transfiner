import numpy as np
import os
import pandas as pd
import subprocess
import csv
from astropy.io import fits
from scipy.special import erfinv
from astropy import units as u
from astropy.coordinates import SkyCoord

datadir = "/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results"

imgdir = "/home/data0/lbq/inference_data/FIRST_fits"
nonmatch_csv = "FIRST_infer_part0-4_th0.1_first_14dec17_matched_cs_diff_new.csv"
hetu_csv = "FIRST_infer_part0-4_th0.1_cs_final.csv"
noncsv = pd.read_csv(os.path.join(datadir, nonmatch_csv))

objname = noncsv['objectname'] 
box = noncsv['box']
imgf = noncsv['imagefilename']
label = noncsv['label']
score = noncsv['score']
masks = noncsv['mask']
local_rms = noncsv['local_rms']
centre_ra = noncsv['centre_ra'] 
centre_dec = noncsv['centre_dec']
deconv_majors = noncsv['deconv_major']
deconv_minors = noncsv['deconv_minor']
deconv_pas = noncsv['deconv_pa']

mirdir = "./FIRST_mir"

def derive_miriad_from_msg(msg):
    peak_flux=[]
    err_peak_flux=[]
    int_flux=[]
    err_int_flux=[]
    ra =  []
    dec=[]
    major = []
    err_major = []
    minor = []
    err_minor =[]
    pa=[]
    err_pa=[]
    deconv_major = np.nan
    deconv_minor = np.nan
    deconv_pa = np.nan
    for line in msg.split(os.linesep):
        if (line.find('Peak value') > -1):
            fds = line.split()
            if '+/-' in line:
                for idx, fd in enumerate(fds):
                    if (fd == '+/-'):
                        #print(idx - 1)
                        peak_flux = float(fds[idx - 1])
                        err_peak_flux = float(fds[idx + 1])
            else:
                peak_flux = float(fds[-1])
                err_peak_flux = 0.0
        elif (line.find('Total integrated flux')> -1):
            fds = line.split()
            for idx, fd in enumerate(fds):
                if (fd == '+/-'):
                    int_flux = float(fds[idx - 1])
                    err_int_flux = float(fds[idx + 1])
        elif (line.find('Right Ascension')> -1):
            fds = line.split()
            ra = fds[2]
        elif (line.find('Declination')> -1):
            fds = line.split()
            dec = fds[1]
        elif (line.find('Major axis (arcsec)')> -1):
            fds = line.split()
            if '+/-' in line:
               if(len(fds)==6):
                  for idx, fd in enumerate(fds):
                      if (fd == '+/-'):
                          major = float(fds[idx - 1])
                          err_major = float(fds[idx + 1])
               else:
                  major = float(fds[-2])
                  if(fds[-1].split('+/-')[-1]=='*******'):
                     err_major = 0.0
                  else:
                     err_major = float(fds[-1].split('+/-')[-1])
            else:
               major = float(fds[-1])
               err_major = 0.0
        elif (line.find('Minor axis (arcsec)')> -1):
            fds = line.split()
            if '+/-' in line:
               if(len(fds)==6):
                  for idx, fd in enumerate(fds):
                      if (fd == '+/-'):
                         minor = float(fds[idx - 1])
                         err_minor = float(fds[idx + 1])
               else:
                  minor = float(fds[-2])
                  if(fds[-1].split('+/-')[-1]=='*******'):
                     err_minor = 0
                  else:
                     err_minor = float(fds[-1].split('+/-')[-1])
            else:
               minor = float(fds[-1])
               err_minor = 0.0
        elif (line.find('Position angle (degrees)')> -1):
            fds = line.split()
            if '+/-' in line:
                for idx, fd in enumerate(fds):
                    if (fd == '+/-'):
                        pa = float(fds[idx - 1])
                        err_pa = float(fds[idx + 1])
            else:
                pa = float(fds[-1])
                err_pa = 0.0
        elif (line.find('Deconvolved Major') > -1):
            fds = line.split()
            deconv_major = float(fds[5]) # for gaussian sources, this overwrites the "Major/minor axis" above
            deconv_minor = float(fds[6])
        if (line.find('Deconvolved Position angle') > -1):
            fds = line.split()
            #logger.info(fds)
            deconv_pa = float(fds[4]) # for gaussian sources, this overwrites the "pa axis" above
            #err_pa = 0.0

    return peak_flux, err_peak_flux, int_flux, err_int_flux, ra, dec, major, err_major, minor, err_minor,pa, err_pa, deconv_major, deconv_minor, deconv_pa

with open(os.path.join(datadir, os.path.splitext(nonmatch_csv)[0]+"_fixed.csv"), 'w') as f:
     csv_w = csv.writer(f)
     csv_w.writerow(["objectname", "imagefilename", "label", "score", "box", "mask", "local_rms", "peak_flux", \
                     "err_peak_flux", "int_flux", "err_int_flux", "ra", "dec", "centre_ra", "centre_dec", \
                     "major", "err_major", "minor", "err_minor", "pa", "err_pa", "deconv_major", "deconv_minor", "deconv_pa"])

     for m in range(len(box)):
         x1 = float(box[m].split('-')[0])
         y1 = float(box[m].split('-')[1])
         x2 = float(box[m].split('-')[2])
         y2 = float(box[m].split('-')[3])

         x1 = int(x1)
         y1 = int(y1)
         x2 = int(x2)
         y2 = int(y2)

         fits_file = os.path.splitext(imgf[m])[0] + ".fits" 
         hdu = fits.open(os.path.join(imgdir, fits_file))[0]
         med = np.nanmedian(hdu.data)
         mad = np.nanmedian(np.abs(hdu.data - med))
         local_sigma = mad / np.sqrt(2) * erfinv(2 * 0.75 - 1)
         clip_level = med + 3 * local_sigma
         mir_file = os.path.splitext(imgf[m])[0] + ".mir"

         y1_new = hdu.data.shape[0]-y2
         y2_new = hdu.data.shape[0]-y1
         centre_y = int(y1_new + (y2_new - y1_new)/2)
         centre_x = int(x1 + (x2 - x1)/2)
        
         xmin = centre_x - 3 
         xmax = centre_x + 3
         ymin = centre_y - 3
         ymax = centre_y + 3  
         if centre_x-3 <0 :
            xmin = 0
         if centre_x + 3 >132:
            xmax = 132
         if centre_y-3 <0 :
            ymin = 0
         if centre_y + 3 >132:
            ymax = 132

         miriad_cmd = "imfit in=%s/%s 'region=boxes(%d,%d,%d,%d)' object=point clip=%f" \
             % (mirdir, mir_file, xmin, ymin, \
             xmax, ymax, clip_level)
         print(miriad_cmd)
         status, msg = subprocess.getstatusoutput(miriad_cmd)
         peak_flux, err_peak_flux, int_flux, err_int_flux, \
         ra, dec, major, err_major, minor, err_minor,pa, \
         err_pa, deconv_major, deconv_minor, deconv_pa = derive_miriad_from_msg(msg)
         major = hdu.header['BMAJ'] * 3600
         minor = hdu.header['BMIN'] * 3600
         pa = hdu.header['BPA'] 
         err_major = 0.0
         err_minor = 0.0
         err_pa = 0.0
         err_peak_flux = 0.0
         err_int_flux = 0.0
         int_flux = peak_flux
         print(ra, dec)
         c = SkyCoord('%s %s' % (ra, dec), unit=(u.hourangle, u.deg))
         ra_d = c.to_string('decimal', precision=5).split()[0]
         dec_d = c.to_string('decimal', precision=5).split()[1]
         csv_w.writerow([objname[m], imgf[m], label[m], score[m], box[m], masks[m], local_rms[m], \
                        peak_flux, err_peak_flux, int_flux, err_int_flux, \
                        ra_d, dec_d, centre_ra[m], centre_dec[m], major, err_major, minor, err_minor, \
                        pa, err_pa, deconv_majors[m], deconv_minors[m], deconv_pas[m]])
         print(major, minor)


fixed_nmatch = pd.read_csv(os.path.join(datadir, os.path.splitext(nonmatch_csv)[0]+"_fixed.csv"))

hetucsv = pd.read_csv(os.path.join(datadir, hetu_csv))

diff_hetu = pd.concat([hetucsv, fixed_nmatch]).drop_duplicates(['imagefilename', 'label', 'score', 'box'], keep=False)

diff_hetu.to_csv(os.path.join(datadir, os.path.splitext(hetu_csv)[0]+"_diff.csv"), index=False)


diff = pd.read_csv(os.path.join(datadir, os.path.splitext(hetu_csv)[0]+"_diff.csv"))

concat_hetu = pd.concat([diff, fixed_nmatch])

concat_hetu.to_csv(os.path.join(datadir, "FIRST_infer_part0-4_th0.1_cs_final_fixed.csv"), index=False)



