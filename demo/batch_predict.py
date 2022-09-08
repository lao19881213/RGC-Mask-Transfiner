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

import linecache
import subprocess

from scipy.special import erfinv
from astropy.io import fits
import astropy.wcs as wcs
from astropy import coordinates
import astropy.units as u

# constants
WINDOW_NAME = "COCO detections"

#Reference code in https://github.com/nhurleywalker/polygon-flux/blob/master/defs.py
def find_bbox_flux(bbox, fitsfile):
    hdu = fits.open(fitsfile)[0]

    # Set any NaN areas to zero or the interpolation will fail
    hdu.data[np.isnan(hdu.data)] = 0.0

    # Get vital stats of the fits file
    bmaj = hdu.header["BMAJ"]
    bmin = hdu.header["BMIN"]
    bpa = hdu.header["BPA"]
    xmax = hdu.header["NAXIS1"]
    ymax = hdu.header["NAXIS2"]
    try:
        pix2deg = hdu.header["CD2_2"]
    except KeyError:
        pix2deg = hdu.header["CDELT2"]
    # Montaged images use PC instead of CD
    if pix2deg == 1.0:
        pix2deg = hdu.header["PC2_2"]
    beamvolume = (1.1331 * bmaj * bmin)
    x1 = float(bbox.split('-')[0])
    y1 = float(bbox.split('-')[1])
    x2 = float(bbox.split('-')[2])
    y2 = float(bbox.split('-')[3])

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    box_data = hdu.data[hdu.data.shape[0]-y2:hdu.data.shape[0]-y1,x1:x2]
    int_flux = np.sum(box_data) #Jy/pix
    int_flux = int_flux * (pix2deg**2) / beamvolume #Jy
    return int_flux

def find_segm_flux(mask, fitsfile):
    hdu = fits.open(fitsfile)[0]

    # Set any NaN areas to zero or the interpolation will fail
    hdu.data[np.isnan(hdu.data)] = 0.0

    # Get vital stats of the fits file
    bmaj = hdu.header["BMAJ"]
    bmin = hdu.header["BMIN"]
    bpa = hdu.header["BPA"]
    xmax = hdu.header["NAXIS1"]
    ymax = hdu.header["NAXIS2"]
    try:
        pix2deg = hdu.header["CD2_2"]
    except KeyError:
        pix2deg = hdu.header["CDELT2"]
    # Montaged images use PC instead of CD
    if pix2deg == 1.0:
        pix2deg = hdu.header["PC2_2"]
    beamvolume = (1.1331 * bmaj * bmin)
    
    int_flux = 0.0
    ix, iy = np.where(np.flipud(mask)) 
    for m in range(len(ix)):
       x = ix[m]
       y = iy[m]
       if not np.isnan(hdu.data[x,y]):
           int_flux += hdu.data[x,y] # In Jy/pix
    #print(int_flux)
    int_flux = int_flux * (pix2deg**2) / beamvolume #Jy
    return int_flux


def derive_miriad_from_msg(msg):
    peak_flux=[]
    err_peak_flux=[]
    int_flux=[]
    err_int_flux=[]
    ra =  []
    dec=[]
    major = []
    err_major = []
    minor = []
    err_minor =[]
    pa=[]
    err_pa=[]
    deconv_major = np.nan
    deconv_minor = np.nan
    deconv_pa = np.nan
    for line in msg.split(os.linesep):
        if (line.find('Peak value') > -1):
            fds = line.split()
            if '+/-' in line:
                for idx, fd in enumerate(fds):
                    if (fd == '+/-'):
                        #print(idx - 1)
                        peak_flux = float(fds[idx - 1])
                        err_peak_flux = float(fds[idx + 1])
            else:
                peak_flux = float(fds[-1])
                err_peak_flux = 0.0
        elif (line.find('Total integrated flux')> -1):
            fds = line.split()
            for idx, fd in enumerate(fds):
                if (fd == '+/-'):
                    int_flux = float(fds[idx - 1])
                    err_int_flux = float(fds[idx + 1])
        elif (line.find('Right Ascension')> -1):
            fds = line.split()
            ra = fds[2]
        elif (line.find('Declination')> -1):
            fds = line.split()
            dec = fds[1]
        elif (line.find('Major axis (arcsec)')> -1):
            fds = line.split()
            if '+/-' in line:
               if(len(fds)==6):
                  for idx, fd in enumerate(fds):
                      if (fd == '+/-'):
                          major = float(fds[idx - 1])
                          err_major = float(fds[idx + 1])
               else:
                  major = float(fds[-2])
                  if(fds[-1].split('+/-')[-1]=='*******'):
                     err_major = 0.0
                  else:
                     err_major = float(fds[-1].split('+/-')[-1])
            else:
               major = float(fds[-1])
               err_major = 0.0
        elif (line.find('Minor axis (arcsec)')> -1):
            fds = line.split()
            if '+/-' in line:
               if(len(fds)==6):
                  for idx, fd in enumerate(fds):
                      if (fd == '+/-'):
                         minor = float(fds[idx - 1])
                         err_minor = float(fds[idx + 1])
               else:
                  minor = float(fds[-2])
                  if(fds[-1].split('+/-')[-1]=='*******'):
                     err_minor = 0
                  else:
                     err_minor = float(fds[-1].split('+/-')[-1])
            else:
               minor = float(fds[-1])
               err_minor = 0.0
        elif (line.find('Position angle (degrees)')> -1):
            fds = line.split()
            if '+/-' in line:
                for idx, fd in enumerate(fds):
                    if (fd == '+/-'):
                        pa = float(fds[idx - 1])
                        err_pa = float(fds[idx + 1])
            else:
                pa = float(fds[-1])
                err_pa = 0.0
        elif (line.find('Deconvolved Major') > -1):
            fds = line.split()
            deconv_major = float(fds[5]) # for gaussian sources, this overwrites the "Major/minor axis" above
            deconv_minor = float(fds[6])
        if (line.find('Deconvolved Position angle') > -1):
            fds = line.split()
            #logger.info(fds)
            deconv_pa = float(fds[4]) # for gaussian sources, this overwrites the "pa axis" above
            #err_pa = 0.0

    return peak_flux, err_peak_flux, int_flux, err_int_flux, ra, dec, major, err_major, minor, err_minor,pa, err_pa, deconv_major, deconv_minor, deconv_pa


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
    parser.add_argument("--input", 
       help="Run prediction for all images in a given path "
            "This argument is the path to the input image directory")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--pnglists", 
        help="input png image file name lists.")
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
    parser.add_argument(
        "--mirdir", 
        help="miriad images dir corresponding for pred images")
    parser.add_argument(
        "--rmsdir", 
        help="rms file dir", default=None)

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
    pnglists = args.pnglists
    rms_dir = args.rmsdir
    mir_dir = args.mirdir

    cpu_device = torch.device("cpu")

    with open(pnglists, 'r') as file:
         linecount=len(file.readlines())
    
    if not no_source_finding:
       tags.append("objectname,imagefilename,label,score,box,mask,local_rms,peak_flux,err_peak_flux,int_flux,err_int_flux,ra,dec,centre_ra,centre_dec,major,err_major,minor,err_minor,pa,err_pa,deconv_major,deconv_minor,deconv_pa")
    else:
       tags.append("imagefilename,label,score,box,mask")
    if args.input:
        input_dir = args.input
        #if len(args.input) == 1:
        #    args.input = glob.glob(os.path.expanduser(args.input[0]))
        #    assert args.input, "The input path(s) was not found"
        for nm in range(linecount):
            fn = os.path.join(args.input,linecache.getline(pnglists, nm+1).split('\n')[0])
            # use PIL, to be consistent with evaluation
            #logger.info(path)
            imagefilename = os.path.basename(fn)
            img = read_image(fn, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            #logger.info(predictions["instances"])
            if "instances" in predictions:
               logger.info(
                   "{}: {} in {:.2f}s".format(
                       fn,
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
               masks = np.array(instances.pred_masks)
               masks_re = np.zeros([masks.shape[1], masks.shape[2], masks.shape[0]])
               for mm in range(masks.shape[0]):
                   masks_re[:,:,mm] = masks[mm,:,:]
               if not no_source_finding:
                  #logger.info("source finding... ...")
                  fits_file = os.path.splitext(imagefilename)[0] + ".fits" #fn.split('.')[0] + ".fits"
                  hdu = fits.open(os.path.join(input_dir, fits_file))[0]
                  #logger.info(os.path.join(input_dir, fits_file))
                  w = wcs.WCS(hdu.header, naxis=2)
                  if rms_dir == None:
                     rms_file = None
                  else:
                     if os.path.exists(os.path.join(rms_dir, os.path.splitext(imagefilename)[0] + "_rms.fits")):
                        rms_file = os.path.splitext(imagefilename)[0] + "_rms.fits"
                     else:
                        rms_file = None
                  if rms_file == None:
                     med = np.nanmedian(hdu.data)
                     mad = np.nanmedian(np.abs(hdu.data - med))
                     local_rms = mad / np.sqrt(2) * erfinv(2 * 0.75 - 1)
                  else:
                     hdu_rms = fits.open(os.path.join(rms_dir, rms_file))[0]
                     local_rms = np.nanmin(hdu_rms.data) #calculate_rms_from_fits(middle=True, mean=True, \
                                 #         boxsize=18, filename=os.path.join(rms_dir,rms_file)) 
                             
                  med = np.nanmedian(hdu.data)
                  mad = np.nanmedian(np.abs(hdu.data - med))
                  local_sigma = mad / np.sqrt(2) * erfinv(2 * 0.75 - 1)

                  for k in range(num_instance):
                      #print(instances.pred_masks.shape)
                      box_str = '{:.5f}-{:.5f}-{:.5f}-{:.5f}'.format(*boxes[k])
                      x1 = float(box_str.split('-')[0])
                      y1 = float(box_str.split('-')[1])
                      x2 = x1 + float(box_str.split('-')[2])
                      y2 = y1 + float(box_str.split('-')[3])
                      
                      box_str = '{:.5f}-{:.5f}-{:.5f}-{:.5f}'.format(x1,y1,x2,y2)
                      
                      x1 = int(x1)
                      y1 = int(y1)
                      x2 = int(x2)
                      y2 = int(y2)


                      if labels[classes[k]] == 'cs':
                         med = np.nanmedian(hdu.data)
                         clip_level = med + 3 * local_sigma
                         mir_file = mir_dir +"/" + os.path.splitext(imagefilename)[0] + ".mir"
                         logger.info("imfit in=%s 'region=boxes(%d,%d,%d,%d)' object=gaussian clip=%f" \
                             % (mir_file, x1, int(hdu.data.shape[0]-y2), \
                                x2, int(hdu.data.shape[0]-y1), clip_level))
                         miriad_cmd = "imfit in=%s 'region=boxes(%d,%d,%d,%d)' object=gaussian clip=%f" \
                             % (mir_file, x1, int(hdu.data.shape[0]-y2), \
                                x2, int(hdu.data.shape[0]-y1), clip_level)

                         status, msg = subprocess.getstatusoutput(miriad_cmd)
                         peak_flux, err_peak_flux, int_flux, err_int_flux, \
                         ra, dec, major, err_major, minor, err_minor,pa, \
                         err_pa, deconv_major, deconv_minor, deconv_pa = derive_miriad_from_msg(msg)
                         
                         c_deg = coordinates.SkyCoord('%s %s' % (ra, dec), unit=(u.hourangle, u.deg))
                         ra_deg = c_deg.ra.degree
                         dec_deg = c_deg.dec.degree

                         y1_new = hdu.data.shape[0]-y2
                         y2_new = hdu.data.shape[0]-y1
                         centre_y = int(y1_new + (y2_new - y1_new)/2)
                         centre_x = int(x1 + (x2 - x1)/2)
                         centre_ra, centre_dec = w.wcs_pix2world([[centre_x, centre_y]], 0)[0][0:2]

                         if centre_ra < 0 :
                            centre_ra = centre_ra + 360

                         c1 = coordinates.SkyCoord(ra=centre_ra, dec=centre_dec, unit=(u.deg, u.deg), frame='fk5')
                         c_new = c1.to_string('hmsdms', decimal=False, precision=1)
                         c_new = c_new.replace('h','').replace('d','').replace('m','').replace('s','').replace(' ','')
                         objname = "J%s" % c_new
                         if peak_flux == []:
                            logger.info("Too few pixels to fit to or fitting failed on %s" % fn)
                         else:
                            tags.append(
                               "{},{},{},{:.2f},{},{},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}".format(\
                                objname, imagefilename, labels[classes[k]], \
                                scores[k], box_str, rles[k]['counts'], local_rms, peak_flux, err_peak_flux, int_flux, err_int_flux, ra_deg, \
                                dec_deg, centre_ra, centre_dec, major, err_major, minor, err_minor, pa, err_pa, deconv_major, deconv_minor, deconv_pa))
                      else:
                         #logger.info('%d %d %d %d' % (x1,y1,x2,y2))
                         box_data = hdu.data[hdu.data.shape[0]-y2:hdu.data.shape[0]-y1,x1:x2]
                         peak_flux = np.nanmax(box_data)
                         peak_xy_offset = np.where(box_data==np.nanmax(box_data))
                         peak_x = x1 + peak_xy_offset[0][0]
                         peak_y = hdu.data.shape[0]-y2 + peak_xy_offset[1][0]
                         peak_ra, peak_dec = w.wcs_pix2world([[peak_x, peak_y]], 0)[0][0:2]
                         if peak_ra < 0 :  #
                            peak_ra = peak_ra + 360
                         y1_new = hdu.data.shape[0]-y2
                         y2_new = hdu.data.shape[0]-y1
                         centre_y = int(y1_new + (y2_new - y1_new)/2)
                         centre_x = int(x1 + (x2 - x1)/2)
                         centre_ra, centre_dec = w.wcs_pix2world([[centre_x, centre_y]], 0)[0][0:2]

                         if centre_ra < 0 :
                            centre_ra = centre_ra + 360

                         c1 = coordinates.SkyCoord(ra=centre_ra, dec=centre_dec, unit=(u.deg, u.deg), frame='fk5')
                         c_new = c1.to_string('hmsdms', decimal=False, precision=1)
                         c_new = c_new.replace('h','').replace('d','').replace('m','').replace('s','').replace(' ','')
                         objname = "J%s" % c_new
                         ra = peak_ra#np.nan #'None'
                         dec = peak_dec#np.nan #'None' 
                         err_peak_flux = np.nan #'None'
                         int_flux = find_segm_flux(masks_re[:,:,k], os.path.join(input_dir, fits_file))
                         #find_segm_flux(instances.pred_masks[k], os.path.join(input_dir, fits_file)) #find_bbox_flux(box_str,os.path.join(input_dir, fits_file))#np.nan #'None' 
                         err_int_flux = np.nan #'None'  
                         major = np.nan #'None' 
                         err_major = np.nan #'None' 
                         minor = np.nan #'None' 
                         err_minor = np.nan #'None' 
                         pa = np.nan #'None' 
                         err_pa = np.nan #'None'
                         deconv_major = np.nan
                         deconv_minor = np.nan
                         deconv_pa = np.nan
                         tags.append(
                         "{},{},{},{:.2f},{},{},{:.5f},{},{},{},{},{},{:.5f},{:.5f},{},{},{},{},{},{},{},{},{},{}".format(\
                         objname, imagefilename, labels[classes[k]], scores[k], \
                         box_str, rles[k]['counts'], local_rms, peak_flux, err_peak_flux, int_flux, err_int_flux, ra, dec, centre_ra, \
                         centre_dec, major, err_major, minor, err_minor, pa, err_pa, deconv_major, deconv_minor, deconv_pa))

               else :
                  for k in range(num_instance):
                      #format: xmin-ymin-xmax-ymax
                      box_str = '{:.5f}-{:.5f}-{:.5f}-{:.5f}'.format(*boxes[k])
                      x1 = float(box_str.split('-')[0])
                      y1 = float(box_str.split('-')[1])
                      x2 = x1 + float(box_str.split('-')[2])
                      y2 = y1 + float(box_str.split('-')[3])
                      
                      box_str = '{:.5f}-{:.5f}-{:.5f}-{:.5f}'.format(x1,y1,x2,y2)
                      #image file name,label,score,box 
                      tags.append(
                       "{},{},{:.2f},{},{}".format(imagefilename, labels[classes[k]], scores[k], box_str, rles[k]['counts']))

    with open('%s.csv' % (catalog_fn), 'w') as fou:
         fc = os.linesep.join(tags)
         fou.write(fc)

