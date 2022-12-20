# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import matplotlib.patches as patches
import json

def CoolColormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#000000', '#000069', '#00188a', '#0d6bff', '#1abaff',
                                                                 '#d9ffff', '#ffffff'], 256)

def deflate_hull(points):
    hull = convex_hull(points)

    for p in hull:
        points.remove(p)

    while points:
        l = len(hull)
        _, p, i = min((distance(hull[i-1], p) + distance(p, hull[i]) - distance(hull[i-1], hull[i]), p, i) 
                      for p in points 
                      for i in range(l))
        points.remove(p)
        hull = hull[:i] + [p] + hull[i:]

    return hull

def convex_hull(points):
    if len(points) <= 3:
        return points
    upper = half_hull(sorted(points))
    lower = half_hull(reversed(sorted(points)))
    return upper + lower[1:-1]

def half_hull(sorted_points):
    hull = []
    for C in sorted_points:
        while len(hull) >= 2 and turn(hull[-2], hull[-1], C) <= -1e-6: 
            hull.pop()
        hull.append(C)
    return hull

def turn(A, B, C):
    return (B[0]-A[0]) * (C[1]-B[1]) - (B[1]-A[1]) * (C[0]-B[0]) 

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

hdu = fits.open('J135414+491315.fits')
img_data = hdu[0].data

model_data = []
with open('model.csv') as f:
     i = 0
     for line in f:
         data = line.split('\n')[0]
         if i > 0:
            model_data.extend(data)
         i = i + 1



plt.figure(1)
plt.imshow(np.flipud(img_data), origin='lower', vmin=-60*0.00002, vmax=60*0.00002, interpolation='gaussian', cmap=CoolColormap())
#plt.title('Data')
plt.axis('off')

plt.tight_layout()


img_new = np.flipud(img_data)
plt.figure(2)
plt.imshow(np.flipud(segm.data), origin='lower', cmap=cmap1,
           interpolation='nearest')
# plt.title('Original Segment')
cmap2 = segm_deblend.make_cmap(seed=123)
plt.axis('off')

new_data=np.flipud(segm.data)

# print(np.min(new_data))

new_data_index = np.where(new_data>=1)

model_data = np.ones(new_data.shape)*(-5*0.00002)

img_new = np.flipud(img_data)

for m in range(len(new_data_index[0])):
      x = new_data_index[0][m]
      y = new_data_index[1][m]
      model_data[x,y] = img_new[x,y]

plt.figure(3)
plt.imshow(model_data, origin='lower', vmin=-60*0.00002, vmax=60*0.00002, cmap=CoolColormap(),
            interpolation='gaussian')
# plt.title('Original Segment')
# cmap2 = segm_deblend.make_cmap(seed=123)
plt.axis('off')


plt.figure(4)
plt.imshow(model_data-img_new, origin='lower', vmin=-60*0.00002, vmax=60*0.00002, cmap=CoolColormap(),
            interpolation='gaussian')
# plt.title('Original Segment')
# cmap2 = segm_deblend.make_cmap(seed=123)
plt.axis('off')



