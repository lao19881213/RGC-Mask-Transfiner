import os 
import numpy as np 
from astropy.io import fits 
import astropy.wcs as wcs

input_dir = "/home/data0/lbq/inference_data/TGSS/img_mosaics"

file_nms = os.listdir(input_dir)
tags = [] 
for fn in file_nms:
    if not fn.endswith(".FITS"): 
       continue
    fits_file = fn#os.path.splitext(image_file[n])[0] + ".fits" 
    hdu = fits.open(os.path.join(input_dir, fits_file))
    print(fn)
    if len(hdu[0].data.shape)==4:
       img_data = hdu[0].data[0,0,:,:]
    else :
       img_data = hdu[0].data 
    image = hdu[0].data 
    w = wcs.WCS(hdu[0].header, naxis=2) 
    width = img_data.shape[1] 
    height = img_data.shape[0] 
    
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
    print('{},{:.5f},{:.5f},{:.5f},{:.5f}'.format(fn, RA_min, DEC_min, RA_max, DEC_max))
    tags.append('{},{:.5f},{:.5f},{:.5f},{:.5f}'.format(fn, RA_min, DEC_min, RA_max, DEC_max))

with open('tgss_image_range.txt', 'w') as f: 
     f.write(os.linesep.join(tags)) 
