import os
import numpy as np
from astropy.io import fits
import astropy.wcs as wcs
import pandas
import subprocess

Mingo_csv = pandas.read_csv("/p9550/LOFAR/LoTSS-DR1/Mingo19_LoMorph_Cat_v2.csv")

RA = Mingo_csv['RA'].values
DEC = Mingo_csv['DEC'].values 
source_name = Mingo_csv['Source_Name'].values

input_dir = "/p9550/LOFAR/LoTSS-DR1/fits"

for i in range(len(RA)):
    with open('lofar_image_range.txt') as f:
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
                filename_fits = "%s.fits" % source_name[i]
                dst_path = "/p9550/LOFAR/LoTSS-DR1/Mingo_fits"
                cutout_cmd = '/home/software/wcstools-3.9.6/bin/getfits -o %s -d %s %s %d %d %d %d' % (filename_fits, dst_path, os.path.join(input_dir, fn), source_x, source_y, 160,160)
                status, msg = subprocess.getstatusoutput(cutout_cmd)

                #print(status)
                if status == 1:
                   print(source_name[i],RA[i],DEC[i]) 
                #try:
                #   os.system('/home/software/wcstools-3.9.6/bin/getfits -o %s -d %s %s %d %d %d %d' % (filename_fits, dst_path, os.path.join(input_dir, fn), source_x, source_y, 160,160))
                #except:
                #   continue
