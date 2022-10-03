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
import math
#import matplotlib.patches as patches
#from photutils.datasets import make_100gaussians_image
from photutils.segmentation import (detect_threshold, detect_sources,
                                    deblend_sources)
from photutils.segmentation import SegmentationImage
import json
import pandas as pd
from shapely import geometry
import cv2
import base64
import argparse
import os

def if_inPoly(polygon, Points):
    line = geometry.LineString(polygon)
    point = geometry.Point(Points)
    polygon = geometry.Polygon(line)
    return polygon.contains(point)

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

parser = argparse.ArgumentParser()
parser.add_argument('--inpdir', help='input img file directory')
parser.add_argument('--outdir', help='output anno directory')
parser.add_argument('--csvfn', help='output csv directory')

args = parser.parse_args()

annos = pd.read_csv(os.path.join(args.inpdir, args.csvfn))

img_fn = annos['imagefilename'].values
labels = annos['label'].values
boxs = annos['box'].values

file_nms = os.listdir(args.inpdir)

for fits_file in file_nms:
    if not fits_file.endswith('.fits'):
       continue
    fn = fits_file
    png_fn = os.path.splitext(fn)[0] + ".png"
    box_re = []
    label_re = []
    for m in range(len(img_fn)):
        if png_fn == img_fn[m]:
           box_re.append(boxs[m])
           label_re.append(labels[m])

    print(png_fn)
    hdu = fits.open(os.path.join(args.inpdir,fn))
    img_data = hdu[0].data
    #print(img_data.shape)
    data = img_data #make_100gaussians_image()
    threshold = detect_threshold(data, nsigma=4)
    #print(threshold)
    sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()
    segm = detect_sources(data, threshold, npixels=5, kernel=kernel)
    segm_deblend = deblend_sources(data, segm, npixels=5, kernel=kernel)
    
    #slc = (slice(273, 297), slice(425, 444))
    datamax = np.nanmax(img_data)
    datamin = np.nanmin(img_data)
    datawide = datamax - datamin
    image_data = np.log10((img_data - datamin) / datawide * 1000 + 1) / 3
    
    print(label_re, box_re) 
    for m in range(len(label_re)):
        x1 = float(box_re[m].split('-')[0])
        y1 = float(box_re[m].split('-')[1])
        x2 = float(box_re[m].split('-')[2])
        y2 = float(box_re[m].split('-')[3])
        top, left, bottom, right = data.shape[0]-y2, x1, data.shape[0]-y1, x2
        #ax1.add_patch(
        #plt.Rectangle((left, top),
        #                   abs(left - right),
        #                   abs(top - bottom), fill=False,
        #                   edgecolor=colors[label_re[m]], linewidth=2)
        #              )
    
    segm_re = dict()
    #rename label 
    label_re_rn = [v + str(label_re[:i].count(v) + 1) if label_re.count(v) > 1 else v for i, v in enumerate(label_re)] 
    #collect segm id for object
    for m in range(len(label_re_rn)):
        x1 = float(box_re[m].split('-')[0]) - 1
        y2 = data.shape[0] - float(box_re[m].split('-')[1]) + 1
        x2 = float(box_re[m].split('-')[2]) + 1
        y1 = data.shape[0] - float(box_re[m].split('-')[3]) - 1
        
        square = [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
        print(square, segm.labels)
        #square = [(0,0), (122,0), (122,122), (0,122)] #
        for n in range(len(segm.labels)):
            p0 = (segm.bbox[n].ixmin, segm.bbox[n].iymin)
            p1 = (segm.bbox[n].ixmin, segm.bbox[n].iymax)
            p2 = (segm.bbox[n].ixmax, segm.bbox[n].iymin)
            p3 = (segm.bbox[n].ixmax, segm.bbox[n].iymax)
            print(p0,p1,p2,p3)
            if(if_inPoly(square, p0)) and (if_inPoly(square, p1)) and \
            (if_inPoly(square, p2)) and (if_inPoly(square, p3)):
               if label_re_rn[m] in segm_re.keys():
                  segm_re[label_re_rn[m]].append(n+1)
               else:
                  segm_re[label_re_rn[m]] = [n+1]
    
    
    
    # reassign labels
    print(segm_re)
    for m in range(len(label_re)):  
        segm.reassign_labels(labels=segm_re[label_re_rn[m]], new_label=m+10)
    
    rm_label = len(segm.labels) - len(label_re)
    #print(rm_label)
    rm_labels = []
    if rm_label > 0:
       for m in range(9):
           if (m+1) in segm.labels:
              rm_labels.append(m+1)
    
       segm.remove_labels(labels=rm_labels)
    
    boundaries = []
    pp = []
    
    
    
    for m in range(len(segm.labels)):
        segm_I = SegmentationImage(segm.data)
        #segm_I.reassign_labels(labels=[2,3], new_label=1)
        segm_I.keep_label(label=segm.labels[m])
        boundaries.append(segm_I.outline_segments())
        boundaries_index = np.where(boundaries[m]>0)
    
        boundaries_index_array = np.vstack((boundaries_index[1], img_data.shape[0]-boundaries_index[0])).T.tolist()
    
    # boundaries_index_array = np.array(boundaries_index).reshape(len(boundaries_index[0]), 2)
    
        pp.append(boundaries_index_array) 
    
    # from matplotlib.path import Path
    # aa= Path(pp[0]).vertices
    # aa = aa.tolist()
    for m in range(len(pp)):  
        # pp[m] = (Path(pp[m]).vertices).tolist()
        
        pp[m] = deflate_hull(pp[m])
    
    
    image = cv2.imread(os.path.join(args.inpdir, png_fn))
    
    (height, width) = image.shape[:2]
    
    with open('example.json','r')as f:
         json_data = json.load(f)
         #shapes = json_data['shapes']
         with open(os.path.join(args.inpdir, png_fn), 'rb') as img_f:
              image_data = img_f.read()
              image_bytes = base64.b64encode(image_data)
              image_tring = image_bytes.decode('utf-8')
              json_data['imageData'] = image_tring
              json_data['imageHeight'] = height
              json_data['imageWidth'] = width
              json_data['imagePath'] = '%s' % png_fn
              json_data['shapes'] = []
              for m in range(len(label_re)):#(shapes)):
                  json_data['shapes'].append({
                  "label": "%s" % label_re[m],
                  "points": pp[m],
                  "group_id": None,
                  "shape_type": "polygon",
                  "flags": {}
                 })
    
    json_fn = fn.replace('fits', 'json')
    with open(os.path.join(args.outdir, json_fn),'w')as dump_f:
         json.dump(json_data, dump_f)
          
     

