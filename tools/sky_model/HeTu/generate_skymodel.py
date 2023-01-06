import json

import argparse

import os
import sys

from astropy.io import fits

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import patches,  lines
#from matplotlib.patches import Polygon
import pycocotools.mask as cocomask

import pandas

import cv2

import matplotlib as mpl

hdu = fits.open('J085556.090+491113.15.fits')
img_data = hdu[0].data



csv_hetu = pandas.read_csv('sky_model_paper.csv')
imagefiles = csv_hetu['imagefilename'].values
labels = csv_hetu['label'].values
scores = csv_hetu['score'].values
boxs = csv_hetu['box'].values
masks = csv_hetu['mask'].values

print(imagefiles)
image = cv2.imread(imagefiles[0])
(height, width) = image.shape[:2]

segm = {
        "size": [width, height],
        "counts": masks[0]}

mask = cocomask.decode(segm)


masks_re = np.array(mask)
print(masks_re.shape)
#masks_plt = np.zeros([masks_re.shape[0], masks_re.shape[1]])
#masks_plt[:,:] = masks_re[:,:]

data_index = np.where(masks_re>=1)

#hdu1 = fits.open('J085556.090+491113.15_bkg.fits')
#bkg_data = hdu1[0].data

#model_data = bkg_data #np.ones(masks_re.shape)*(0.000002)

model_data = np.ones(masks_re.shape)*(0.00)

img_new = np.flipud(img_data)

for m in range(len(data_index[0])):
      x = data_index[0][m]
      y = data_index[1][m]
      model_data[x,y] = img_new[x,y]

model_data[model_data<0] =0

datamax = np.nanmax(model_data)
datamin = np.nanmin(model_data)
datawide = datamax - datamin
model_data_new = np.log10((model_data - datamin) / datawide * 1000 + 1) / 3

hdu[0].data = np.flipud(model_data)

hdu.writeto('model.fits', overwrite=True)

def CoolColormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#000000', '#000069', '#00188a', '#0d6bff', '#1abaff',
                                                                 '#d9ffff', '#ffffff'], 256)


plt.figure(1)
plt.axis('off')
plt.imshow(np.flipud(model_data_new), origin='lower', cmap=CoolColormap(), vmin=np.nanmin(model_data_new), vmax=np.nanmax(model_data_new)) 

plt.savefig('model_hetu.pdf')



residual_data = img_new - model_data

datamax = np.nanmax(residual_data)
datamin = np.nanmin(residual_data)
datawide = datamax - datamin
residual_data_new = np.log10((residual_data - datamin) / datawide * 1000 + 1) / 3

hdu[0].data = np.flipud(residual_data)

hdu.writeto('residual.fits', overwrite=True)


plt.figure(2)
plt.axis('off')
plt.imshow(np.flipud(residual_data_new), origin='lower', cmap=CoolColormap())

plt.savefig('residual_hetu.pdf')


hdu2 = fits.open('model_smooth.fits')
model_data_s = hdu2[0].data

residual_data_s = img_new - model_data_s

hdu[0].data = residual_data_s

hdu.writeto('residual_smooth.fits', overwrite=True)


datamax = np.nanmax(img_data)
datamin = np.nanmin(img_data)
datawide = datamax - datamin
img_data_new = np.log10((img_data - datamin) / datawide * 1000 + 1) / 3

plt.figure(3)
plt.axis('off') 
plt.imshow(img_data_new, origin='lower', cmap=CoolColormap())
plt.savefig('original.pdf')
#plot color scale -0.001 - 0.001
