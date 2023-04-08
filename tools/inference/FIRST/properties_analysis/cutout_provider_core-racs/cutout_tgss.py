import os
import numpy as np
from astropy.io import fits
import astropy.wcs as wcs
import pandas
import subprocess

tgss_csv = pandas.read_csv("/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper.csv")

RA = tgss_csv['centre_ra'].values
DEC = tgss_csv['centre_dec'].values 
source_name = tgss_csv['source_name'].values
labels = tgss_csv['label'].values

input_dir = "/home/data0/lbq/inference_data/TGSS/img_mosaics"

#clns = {'fr1': 'FRI', 'fr2': 'FRII', 'ht': 'HT', 'cj': 'CJ'}
clns = {'cj': 'CJ'}
for i in range(len(RA)):
    for cln in clns.keys():
            #print(cls)
            if(labels[i]==cln):
                with open('tgss_image_range.txt') as f:
                     for line in f:
                         lines = line.split('\n')[0]
                         fn = lines.split(',')[0]
                         RA_min = float(lines.split(',')[1]) 
                         DEC_min = float(lines.split(',')[2]) 
                         RA_max = float(lines.split(',')[3])
                         DEC_max = float(lines.split(',')[4])
                         if RA[i] > RA_min and RA[i] < RA_max and DEC[i] > DEC_min and DEC[i] < DEC_max:
                            hdu = fits.open(os.path.join(input_dir, fn))
                            w = wcs.WCS(hdu[0].header, naxis=2)
                            source_x, source_y = w.wcs_world2pix([[RA[i],DEC[i]]],0).transpose()
                            filename_fits = "%s.FITS" % source_name[i]
                            dst_path = "/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/%s/TGSS" % clns[cln]
                            if np.isnan(source_x) or np.isnan(source_y):
                               continue
                            print('/home/data0/lbq/software/wcstools/bin/getfits -o %s -d %s %s %d %d %d %d' % (filename_fits, dst_path, os.path.join(input_dir, fn), source_x, source_y, 132,132))
                            cutout_cmd = '/home/data0/lbq/software/wcstools/bin/getfits -o %s -d %s %s %d %d %d %d' % (filename_fits, dst_path, os.path.join(input_dir, fn), source_x, source_y, 132,132)
                            status, msg = subprocess.getstatusoutput(cutout_cmd)

                            #print(status)
                            if status == 0:
                               try:
                                  hdu_o = fits.open(os.path.join(dst_path, filename_fits))
                                  centre_data = hdu_o[0].data[66, 66]
                                  if np.isnan(centre_data):
                                     continue
                                  else:
                                     break
                               except:
                                  continue
                   #print(source_name[i],RA[i],DEC[i]) 
                #try:
                #   os.system('/home/software/wcstools-3.9.6/bin/getfits -o %s -d %s %s %d %d %d %d' % (filename_fits, dst_path, os.path.join(input_dir, fn), source_x, source_y, 160,160))
                #except:
                #   continue
