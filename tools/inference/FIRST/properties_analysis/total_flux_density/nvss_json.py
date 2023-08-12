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
import cv2
import json
import base64

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

NVSS_dir = '/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/FRII/NVSS4flux_png'

NVSS_fits_dir = '/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/FRII/NVSS4flux'

fitsfs = os.listdir(NVSS_fits_dir)

hetu_csv = pd.read_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_fr2.csv')

ras = hetu_csv['centre_ra'].values
decs = hetu_csv['centre_dec'].values
boxs = hetu_csv['box'].values
labels = hetu_csv['label'].values
source_names = hetu_csv['source_name'].values

root_dir = '/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis'

boxs_new = []
for m in range(len(fitsfs)):
      source_name = os.path.splitext(fitsfs[m])[0].split('_')[1]
      FIRST_fits = '%s/FRII/FIRST/%s.fits' % (root_dir, source_name)
      hdu_FIRST = fits.open(FIRST_fits)[0] 
      try:
          pix_scale_hetu = hdu_FIRST.header["CD2_2"]
      except KeyError:
          pix_scale_hetu = hdu_FIRST.header["CDELT2"]
      
      NVSS_fits = '%s/FRII/NVSS4flux/NVSS_%s.fits' % (root_dir, source_name)
      try:
          hdu_NVSS = fits.open(NVSS_fits)[0]
      except:
          continue
      try:
          pix_scale_nvss = hdu_NVSS.header["CD2_2"]
      except KeyError:
          pix_scale_nvss = hdu_NVSS.header["CDELT2"]

      for mm in range(len(ras)):
          if source_name == source_names[mm]:
             box = boxs[mm] 
             ra = ras[mm]
             dec = decs[mm]
             x1 = float(box.split('-')[0])
             y1 = float(box.split('-')[1])
             x2 = float(box.split('-')[2])
             y2 = float(box.split('-')[3])
             width = x2 - x1
             height = y2 - y1
             factor = pix_scale_nvss / pix_scale_hetu
             w = wcs.WCS(hdu_NVSS.header, naxis=2)
             centre_x, centre_y = w.wcs_world2pix([[ra,dec]],0).transpose() 
             width_new = width / factor * 4.0
             height_new = height / factor * 4.0
             x1_new = centre_x - width/2.0
             x2_new = centre_x + width/2.0
             y1_new = centre_y - height/2.0
             y2_new = centre_y + height/2.0 

             print(fitsfs[m])
             pngf = os.path.splitext(fitsfs[m])[0] + '.png'
             json_file = 'example.json'
             image = cv2.imread(os.path.join(NVSS_dir, pngf))

             (height, width) = image.shape[:2]
             
             with open(json_file,'r', encoding='utf-8')as f:
                  json_data = json.load(f)
                  #shapes = json_data['shapes']
             with open(os.path.join(NVSS_dir, pngf), 'rb') as img_f:
                  image_data = img_f.read()
                  image_bytes = base64.b64encode(image_data)
                  image_tring = image_bytes.decode('utf-8')
                  json_data['imageData'] = image_tring
                  json_data['imageHeight'] = height
                  json_data['imageWidth'] = width
                  json_data['imagePath'] = '%s' % pngf
                  json_data['shapes'] = []
                  label = "%s" % 'fr2'
                  json_data['shapes'].append({
                       "label": label,
                       "points": [[x1_new[0], y1_new[0]],[x2_new[0], y2_new[0]]],
                       "group_id": None,
                       "shape_type": "rectangle",
                       "flags": {}
                       })
             fn_json = os.path.splitext(pngf)[0] + '.json'
             print(x1_new[0])
             #json_data = json.dumps(json_data, cls=NumpyEncoder)
             with open(os.path.join(NVSS_dir, fn_json),'w')as dump_f:
                  json.dump(json_data, dump_f)
             print('Successful generated  ')
             

   
