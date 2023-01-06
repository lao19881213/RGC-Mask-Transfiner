
import pycocotools.mask as mask_util
import json

import PIL.Image
import PIL.ImageDraw
import numpy as np

def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

with open('J085556.090+491113.15.json', 'r') as fp:
     data = json.load(fp)  # load json file
     for shapes in data['shapes']:
         points = shapes['points']



mask  = polygons_to_mask([132, 132], points)


rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]

rle["counts"] = rle["counts"].decode("utf-8")

print(rle["counts"])
