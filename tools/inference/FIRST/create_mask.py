import argparse
import json
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from labelme import utils
import numpy as np
import glob
import PIL.Image
import pycocotools.mask as mask_util

def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def getRLEmask(points, heitht, width):
    # img = np.zeros([self.height,self.width],np.uint8)
    # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # plot boundaries
    # cv2.fillPoly(img, [np.asarray(points)], 1)  # plot polylines, pixel values are 1
    polygons = points
    #print(polygons)

    mask = polygons_to_mask([height, width], polygons)
    return mask 

with open('J144830.946+435216.27.json', 'r') as fp:
    data = json.load(fp)  # load json file
    #print(json_file)
    height = data['imageHeight']
    width = data['imageWidth']
    for shapes in data['shapes']:
        points = shapes['points']#here point is using rectangle labeled has two points, need to be change to four points
        print(points)
        mask = getRLEmask(points,height, width)
        print(mask)
        rle = mask_util.encode(np.array(mask[:,:,None], order="F", dtype="uint8"))[0] 
        rle["counts"] = rle["counts"].decode("utf-8")
        print(rle["counts"])
        
