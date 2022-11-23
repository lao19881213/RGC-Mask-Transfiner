# -*- coding: utf-8 -*-
import sys
import os
import glob
import cv2
import numpy as np
import json
#---below---imgaug module
import imgaug as ia
import imgaug.augmenters as iaa
#from imgaug.augmentables import Keypoint, KeypointsOnImage
from PIL import Image 
from labelme import utils
import base64
from shapely import geometry
 
'''
ticks:
1) picture type : png;
2) while augumenting, mask not to go out image shape;
3) maybe some error because data type not correct.
'''
def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.mkdir(path)
        print('====================')
        print('creat path : ', path)
        print('====================')
    return 0
 
 
def check_json_file(path):
    for i in path:
        json_path = i[:-3] + 'json'
        if not os.path.exists(json_path):
            print('error')
            print(json_path, ' not exist !!!')
            sys.exit(1)
 
 
def read_jsonfile(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
 
 
def save_jsonfile(object, save_path):
    json.dump(object, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
 
 
def get_points_from_json(json_file):
    polygons = []
    shapes = json_file['shapes']
    for i in range(len(shapes)):
         polygons.append(ia.Polygon(shapes[i]["points"]))
    #    for j in range(len(shapes[i]["points"])):
    #        point_list.append(shapes[i]["points"][j])
    return polygons
 
 
def write_points_to_json(json_file, polygons_aug):
    #k = 0
    #print(polygons_aug[0].exterior[0][1])
    new_json = json_file
    shapes = new_json['shapes']
    for i in range(len(shapes)):
        new_polygons = polygons_aug[i].exterior.tolist()
        #for j range(len())
        new_json['shapes'][i]["points"] = new_polygons
        #for j in range(len(shapes[i]["points"])):
        #    new_point = [polygons_aug.x, polygons_aug.y]
        #    new_json['shapes'][i]["points"][j] = new_point
            #k = k + 1
    return new_json
 

def if_inPoly(polygon, Points):
    line = geometry.LineString(polygon)
    point = geometry.Point(Points)
    polygon = geometry.Polygon(line)
    return polygon.contains(point)

 
#-----------------------------Sequential-augument choose here-----
ia.seed(1)
 
# Define our augmentation pipeline.
#sometimes = lambda aug : iaa.Sometimes(0.3, aug)
seq = iaa.Sequential([
        iaa.Affine(
            rotate=(-180, 180)
        )  
])
 
 
if __name__ == '__main__':
    # TO-DO-BELOW
    aug_times = 50
    in_dir = "/home/data0/lbq/training_sets/train_final"
    out_dir = "/home/data0/lbq/training_sets/train_final_aug"
    #---check-------------
    mkdir(out_dir)
    imgs_dir_list = glob.glob(os.path.join(in_dir, '*.png'))
    check_json_file(imgs_dir_list)
 
    # for : image
    for idx_jpg_path in imgs_dir_list:
        idx_json_path = idx_jpg_path[:-3] + 'json'
        # get image file
        #idx_img = cv2.imdecode(np.fromfile(idx_jpg_path, dtype=np.uint8), 1)
        img = Image.open(idx_jpg_path)
        # sp = img.size
        idx_img = np.asarray(img)
        # get json file
        idx_json = read_jsonfile(idx_json_path)
        # get point_list from json file
        polygons = get_points_from_json(idx_json)
        # convert to Keypoint(imgaug mode)
        #kps = KeypointsOnImage([Keypoint(x=p[0], y=p[1]) for p in  points_list], shape=idx_img.shape)
        # Augument Keypoints and images
        for idx_aug in range(aug_times):
            do_aug = 1
            image_aug, polygons_aug = seq(image=idx_img, polygons=polygons)
            sps = idx_json['shapes']
            for i in range(len(sps)):
                for p in polygons_aug[i].exterior.tolist():
                    square = [(0,0), (132,0), (132,132), (0,132)] #
                    pt = (p[0], p[1])
                    if(if_inPoly(square, pt)):
                       print('good point')
                    else:
                       print('bad point')
                       do_aug = 0
            #image_aug.astype(np.uint8)
            #image_data = image_aug.read()
            #image_bytes = base64.b64encode(image_aug.astype(np.uint8))
            #image_tring = image_bytes.decode('utf-8')
            # write aug_points in json file
            #idx_new_json = write_points_to_json(idx_json, polygons_aug)
            #idx_new_json["imagePath"] = idx_jpg_path.split(os.sep)[-1][:-4] + str(idx_aug) + '.png'
            #idx_new_json["imageData"] = image_tring #str(utils.img_arr_to_b64(image_aug), encoding='utf-8')
            # save
            if do_aug == 1:
               new_img_path = os.path.join(out_dir, idx_jpg_path.split(os.sep)[-1][:-4] + '_' + str(idx_aug) + '.png')
               Image.fromarray(image_aug).save(new_img_path)
               #image_data = image_aug.read()
               with open(new_img_path, 'rb') as img_f:
                    image_data = img_f.read()
                    image_bytes = base64.b64encode(image_data)
                    image_tring = image_bytes.decode('utf-8')
                    idx_new_json = write_points_to_json(idx_json, polygons_aug)
                    idx_new_json["imagePath"] = idx_jpg_path.split(os.sep)[-1][:-4] + '_' + str(idx_aug) + '.png'
                    idx_new_json["imageData"] = image_tring #str(utils.img_arr_to_b64(image_aug), encoding='utf-8')
                    #cv2.imwrite(new_img_path, image_aug)
                    new_json_path = new_img_path[:-3] + 'json'
                    save_jsonfile(idx_new_json, new_json_path)



#import numpy as np
#import imgaug as ia
#import imgaug.augmenters as iaa
#
#images = np.zeros((2, 128, 128, 3), dtype=np.uint8)  # two example images
#images[:, 64, 64, :] = 255
#polygons = [
#    [ia.Polygon([(10.5, 10.5), (50.5, 10.5), (50.5, 50.5)])],
#    [ia.Polygon([(0.0, 64.5), (64.5, 0.0), (128.0, 128.0), (64.5, 128.0)])]
#]
#
#seq = iaa.Sequential([
#    iaa.AdditiveGaussianNoise(scale=0.05*255),
#    iaa.Affine(translate_px={"x": (1, 5)})
#])
#
#images_aug, polygons_aug = seq(images=images, polygons=polygons)
