import math
import os
import numpy as np
from astropy.io import fits
import astropy.wcs as wcs
import pandas as pd
from scipy import spatial

def PositionAngle(ra1,dec1,ra2,dec2):
	"""
	Given the positions (ra,dec) in degrees,
	calculates the position angle of the 2nd source wrt to the first source
	in degrees. The position angle is measured North through East

	Arguments:
	(ra,dec) -- Coordinates (in degrees) of the first and second source

	Returns: 
	  -- The position angle measured in degrees,

	"""

	#convert degrees to radians
	ra1,dec1,ra2,dec2 = ra1 * math.pi / 180. , dec1 * math.pi / 180. , ra2 * math.pi / 180. , dec2 * math.pi / 180. 
	return (math.atan( (math.sin(ra2-ra1))/(
			math.cos(dec1)*math.tan(dec2)-math.sin(dec1)*math.cos(ra2-ra1))
				)* 180. / math.pi )# convert radians to degrees


FIRST_csv = '/home/data0/lbq/inference_data/first_14dec17.csv'

csv_FIRST = pd.read_csv(FIRST_csv)

ra_FIRST = csv_FIRST['RA'].values

dec_FIRST = csv_FIRST['DEC'].values


result_file = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_fr2_flux_fixed_RPA_comp.csv'
hetu_csv = pd.read_csv(result_file)
image_file = hetu_csv['image_filename'].values
box = hetu_csv['box'].values
ra = hetu_csv['centre_ra'].values
dec = hetu_csv['centre_dec'].values
ra_peak = hetu_csv['ra'].values
dec_peak = hetu_csv['dec'].values

FIRST_fits_dir = '/home/data0/lbq/inference_data/FIRST_fits'
RPAs = hetu_csv['RPA_comp'].values
#RPA_all = []
#for n in range(len(ra)):
#    RPA_all.append('--') 
#
#hetu_csv['RPA_comp'] = RPA_all
#hetu_csv.to_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_fr2_flux_fixed_RPA_comp.csv', index = False)

for n in range(len(ra)):
   if RPAs[n] == '--': 
      print('processing row ', n)
      fits_file = os.path.splitext(image_file[n])[0] + ".fits"
      hdu = fits.open(os.path.join(FIRST_fits_dir, fits_file))
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

      x1 = float(box[n].split('-')[0])
      y1 = float(box[n].split('-')[1])
      x2 = float(box[n].split('-')[2])
      y2 = float(box[n].split('-')[3])

      x1 = int(x1)
      y1 = int(y1)
      x2 = int(x2)
      y2 = int(y2)
      #print(x1,x2,y1,y2)
      y1_new = hdu[0].data.shape[0]-y2
      y2_new = hdu[0].data.shape[0]-y1

      #print(ra_re)
      
      ra_comp = []
      dec_comp = []
      xy_comp = []
      for m in range(len(ra_FIRST)):
          ra_first = float(ra_FIRST[m])
          dec_first = float(dec_FIRST[m])
          img_x, img_y = w.wcs_world2pix([[ra_first,dec_first]],0).transpose()
          if ra_first <= RA_max and ra_first >= RA_min and dec_first <= DEC_max and dec_first >= DEC_min:
             if img_x <= x2 and img_x >= x1 and img_y <= y2_new and img_y >= y1_new: 
                ra_comp.append(ra_first)
                dec_comp.append(dec_first)
                xy_comp.append([img_x[0], img_y[0]])
      print(len(ra_comp))
      if(len(ra_comp)<=1):
         print("number of comps is 1")
         ra1 = ra[n]
         ra2 = ra_peak[n]
         dec1 = dec[n]
         dec2 = dec_peak[n]
      elif(len(ra_comp)==2):
         ra1 = ra_comp[0]
         ra2 = ra_comp[1]
         dec1 = dec_comp[0]
         dec2 = dec_comp[1]
      else:
         pts = np.array(xy_comp)
         #print(pts.shape)
         # two points which are fruthest apart will occur as vertices of the convex hull
         candidates = pts[spatial.ConvexHull(pts).vertices]
         # get distances between each pair of candidate points
         dist_mat = spatial.distance_matrix(candidates, candidates)
         # get indices of candidates that are furthest apart
         i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
         #print(candidates[i], candidates[j])
         for nn in range(len(xy_comp)):
             if pts[nn][0] == candidates[i][0] and pts[nn][1] == candidates[i][1]:
                ra1 = ra_comp[nn]
                dec1 = dec_comp[nn]
             if pts[nn][0] == candidates[j][0] and pts[nn][1] == candidates[j][1]:
                ra2 = ra_comp[nn]
                dec2 = dec_comp[nn]
      RPA = PositionAngle(ra1,dec1,ra2,dec2)
      print(RPA)
      if RPA < 0:
         RPA += 180.
      
      hetu_csv.loc[n,'RPA_comp'] = RPA
      hetu_csv.to_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_fr2_flux_fixed_RPA_comp.csv', index = False) 

##hetu_csv['RPA_comp'] = RPA_all
##hetu_csv.to_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_fr2_flux_fixed_vlass.csv', index = False)      
#      
