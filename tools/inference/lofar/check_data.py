# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import numpy as np
import os
import linecache

from astropy.io import fits

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 batch predict for builtin configs")
    parser.add_argument("--input", 
       help="Run prediction for all images in a given path "
            "This argument is the path to the input image directory")
    parser.add_argument(
        "--pnglists", 
        help="input png image file name lists.")

    return parser



if __name__ == "__main__":
    args = get_parser().parse_args()
    pnglists = args.pnglists
    with open(pnglists, 'r') as file:
         linecount=len(file.readlines())
    pnglists = args.pnglists 
    if args.input:
        input_dir = args.input
        for nm in range(linecount):
            fn = os.path.join(args.input,linecache.getline(pnglists, nm+1).split('\n')[0])
            imagefilename = os.path.basename(fn)
            fits_file = os.path.splitext(imagefilename)[0] + ".fits" #fn.split('.')[0] + ".fits"
            hdu = fits.open(os.path.join(input_dir, fits_file))[0]
            if hdu.data.shape[0] != hdu.data.shape[1]:
               print(imagefilename)
