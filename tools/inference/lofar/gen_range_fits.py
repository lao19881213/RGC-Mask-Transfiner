import os 
import numpy as np 
from astropy.io import fits 
import astropy.wcs as wcs

input_dir = "/p9550/LOFAR/LoTSS-DR1/fits"

file_nms = os.listdir(input_dir)
tags = [] 
for fn in file_nms:
    if not fn.endswith(".fits"): 
       continue
    fits_file = fn#os.path.splitext(image_file[n])[0] + ".fits" 
    hdu = fits.open(os.path.join(input_dir, fits_file)) 
    image = hdu[0].data 
    w = wcs.WCS(hdu[0].header, naxis=2) 
    width = hdu[0].data.shape[1] 
    height = hdu[0].data.shape[0] 
    
    bottom_left = [0, 0] 
    top_left = [0, height - 1] 
    top_right = [width - 1, height - 1] 
    bottom_right = [width - 1, 0] 
    
    ret = np.zeros([4, 2]) 
    ret[0, :] = w.wcs_pix2world([bottom_left], 0)[0][0:2]  
    ret[1, :] = w.wcs_pix2world([top_left], 0)[0][0:2]  
    ret[2, :] = w.wcs_pix2world([top_right], 0)[0][0:2]  
    ret[3, :] = w.wcs_pix2world([bottom_right], 0)[0][0:2]  
    RA_min, DEC_min, RA_max, DEC_max = np.min(ret[:, 0]),   np.min(ret[:, 1]),  np.max(ret[:, 0]),  np.max(ret[:, 1])
    tags.append('{},{:.5f},{:.5f},{:.5f},{:.5f}'.format(fn, RA_min, DEC_min, RA_max, DEC_max))

with open('lofar_image_range.txt', 'w') as f: 
     f.write(os.linesep.join(tags)) 
