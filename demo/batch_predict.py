# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import torch

import pycocotools.mask as mask_util

from detectron2.structures import Boxes, BoxMode
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    #cfg.VIS_PERIOD = 100
    #cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 batch predict for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/transfiner/mask_rcnn_R_101_FPN_3x_deform.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.png'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--nosourcefinding", 
        action='store_true', 
        help='Do not run souce finding. Defalult = False', 
        default=False)
    parser.add_argument(
        "--catalogfn", 
        help="output catalog file name", 
        default='catalog')
    return parser



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    # results
    tags = []

    labels = ['cs', 'fr1', 'fr2', 'ht', 'cj']
    no_source_finding = args.nosourcefinding
    catalog_fn = args.catalogfn

    cpu_device = torch.device("cpu")
    
    if not no_source_finding:
       tags.append("objectname,imagefilename,label,score,box,mask,local_rms,peak_flux,err_peak_flux,int_flux,err_int_flux,ra,dec,centre_ra,centre_dec,major,err_major,minor,err_minor,pa,err_pa,deconv_major,deconv_minor,deconv_pa")
    else:
       tags.append("imagefilename,label,score,box,mask")
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            #logger.info(path)
            imagefilename = os.path.basename(path)
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            #logger.info(predictions["instances"])
            if "instances" in predictions:
               logger.info(
                   "{}: {} in {:.2f}s".format(
                       path,
                       "detected {} instances".format(len(predictions["instances"]))
                       if "instances" in predictions
                       else "finished",
                       time.time() - start_time,
                   )
               )

               instances = predictions["instances"].to(cpu_device)
               num_instance = len(instances)
               if num_instance == 0:
                   results = []

               boxes = instances.pred_boxes.tensor.numpy()
               boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
               boxes = boxes.tolist()
               scores = instances.scores.tolist()
               classes = instances.pred_classes.tolist()

               has_mask = instances.has("pred_masks")
               if has_mask:
                   # use RLE to encode the masks, because they are too large and takes memory
                   # since this evaluator stores outputs of the entire dataset
                   rles = [
                       mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                       for mask in instances.pred_masks
                   ]
                   for rle in rles:
                       # "counts" is an array encoded by mask_util as a byte-stream. Python3's
                       # json writer which always produces strings cannot serialize a bytestream
                       # unless you decode it. Thankfully, utf-8 works out (which is also what
                       # the pycocotools/_mask.pyx does).
                       rle["counts"] = rle["counts"].decode("utf-8")
               if not no_source_finding:
                  logger.info("TO DO source finding.")
               else :
                  for k in range(num_instance):
                      #format: xmin-ymin-xmax-ymax
                      box_str = '{}-{}-{}-{}'.format(*boxes[k])
                      #image file name,label,score,box 
                      tags.append(
                       "{},{},{:.2f},{},{}".format(imagefilename, labels[classes[k]], scores[k], box_str, rles[k]['counts']))

    with open('%s.csv' % (catalog_fn), 'w') as fou:
         fc = os.linesep.join(tags)
         fou.write(fc)
            #results = []
            #for k in range(num_instance):
            #    result = {
            #        "image_id": img_id,
            #        "category_id": classes[k],
            #        "bbox": boxes[k],
            #        "score": scores[k],
            #    }
            #    if has_mask:
            #        result["segmentation"] = rles[k]
            #    results.append(result)
            #return results

            #logger.info(boxes, scores, classes)

