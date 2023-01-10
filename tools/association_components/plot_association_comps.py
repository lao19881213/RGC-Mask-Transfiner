import json

import argparse

import os
import sys
import random
import itertools
import colorsys

import skimage.io
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
#from matplotlib import patches,  lines
#from matplotlib.patches import Polygon
import pycocotools.mask as cocomask

import visualize

from astropy.io import fits
import astropy.wcs as wcs


cls = ['cs','fr1','fr2','ht','cj']

img_fns = ['J141653.051+104828.53.png', 'J140813.859+101013.77.png', 'J131616.979+070246.57.png']

for img_fn in img_fns:
    boxes = []
    masks = []
    class_ids = []
    class_names = []
    scores = []
    cls_id = 0
    image_name = None
    ras_re = []
    decs_re = []
    xcs = []
    ycs = []
    with open('instances_results.json', 'r') as fre:
         data_re = json.load(fre)
         for i in range(len(data_re)):
             cl_id = int(data_re[i]['category_id'])
             img_id = data_re[i]['image_name']
             #print(img_id)
             score = float(data_re[i]['score'])
             box = data_re[i]['bbox']
             segm = data_re[i]['segmentation']
             mask = cocomask.decode(segm)
             x1 = float(box[0])
             y1 = float(box[1])
             x2 = x1 + float(box[2])
             y2 = y1 + float(box[3])
             #print(mask)
             #print(img_fn)
             if img_fn==img_id:
                image_name = img_id
                boxes.extend([(int(y1), int(x1), int(y2), int(x2))])
                masks.append(mask)
                class_ids.append(cls_id)
                cls_id = cls_id + 1
                class_names.append(cls[cl_id-1])
                scores.append(score)
                print(box)
         with open('FIRST_extended_components.txt') as f:
              for line in f:
                  imgfn = line.split(',')[0]
                  ras = float(line.split(',')[1])
                  decs = float(line.split(',')[2].split('\n')[0])
                  if img_fn == imgfn:
                     ras_re.append(ras)
                     decs_re.append(decs)
                     fits_file = os.path.splitext(img_fn)[0] + ".fits"
                     hdu = fits.open(fits_file)
                     w = wcs.WCS(hdu[0].header, naxis=2)
                     img_x, img_y = w.wcs_world2pix([[ras,decs]],0).transpose()
                     xcs.append(img_x)
                     ycs.append(hdu[0].data.shape[0]-img_y)
                     print('%s, %f, %f, %f, %f' % (img_fn, ras, decs, img_x, hdu[0].data.shape[0]-img_y)) 
       
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    print(np.array(masks).shape)
    image_dir = "./"
    image_data = skimage.io.imread(os.path.join(image_dir, image_name))
    boxes = np.array(boxes)
    masks = np.array(masks)
    masks_re = np.zeros([masks.shape[1], masks.shape[2], masks.shape[0]])
    for mm in range(masks.shape[0]):
        masks_re[:,:,mm] = masks[mm,:,:]
    #masks = masks.reshape(masks.shape[1], masks.shape[2], masks.shape[0])
    class_ids = np.array(class_ids)
    class_names = np.array(class_names)
    scores = np.array(scores)
    #print(boxes.shape[0]) 
    #print(masks.shape[-1])
    #print(class_ids.shape[0])
    #print(class_names)
    #print(image_name) 
    visualize.display_instances(
       image_data, boxes, masks_re, class_ids,
       class_names, xcs, ycs, scores,
       show_bbox=True, show_mask=True,
       title="Predictions")
    out_dir = './'
    png_fn = image_name.replace('.png', '_comps.pdf')
    plt.savefig(os.path.join(out_dir,png_fn), dpi=600)
      #plt.

