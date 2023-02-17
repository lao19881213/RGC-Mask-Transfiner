import os
import os.path as osp

from astropy.io import fits
import numpy as np
import argparse

#mSubimage [-d] [-a] [-h hdu] [-s statusfile] in.fits out.fits ra dec xsize [ysize]
#mSubimage -p [-d] [-h hdu] [-s statusfile] in.fits out.fits xstartpix ystartpix xpixsize [ypixsize]

montage_path = '/home/data0/lbq/software/Montage-6.0/bin'
subimg_exec = '%s/mSubimage' % montage_path
regrid_exec = '%s/mProject' % montage_path
imgtbl_exec = '%s/mImgtbl' % montage_path
coadd_exec = '%s/mAdd' % montage_path
subimg_cmd = '{0} %s %s %.4f %.4f %.4f %.4f'.format(subimg_exec)
splitimg_cmd = '{0} -p %s %s %d %d %d %d'.format(subimg_exec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split_fits')
    parser.add_argument('--fname', dest='fname', type=str, default='/o9000/MWA/GLEAM/data_challenge1/SKAMid_B1_1000h_v3.fits', help='fits file name')
    parser.add_argument('--workdir', dest='workdir', type=str, default='/o9000/MWA/GLEAM/data_challenge1/split_B1_1000h', help='output png dir')
    parser.add_argument('--ImageSize', dest='ImageSize', type=int, default=132, help='size of split image')
    parser.add_argument('--OverlappingSize', dest='OverlappingSize', type=int, default=20, help='overlapping size of split image')
    args = parser.parse_args() 
    fname = args.fname
    work_dir = args.workdir
    ImageSize = args.ImageSize
    OverlappingSize = args.OverlappingSize
    hdu_list = fits.open(fname)
    h1,image_data = hdu_list[0].header, hdu_list[0].data
    print(image_data.shape)
    for i, x1 in enumerate(range(0, image_data.shape[3], ImageSize - OverlappingSize)):
        for j, y1 in enumerate(range(0, image_data.shape[2], ImageSize - OverlappingSize)):
            fid = osp.basename(fname).replace('.fits', '_%d-%d.fits' % (i, j))
            out_fname = osp.join(work_dir, fid)
            print(splitimg_cmd % (fname, out_fname, x1, y1, ImageSize, ImageSize))
            os.system(splitimg_cmd % (fname, out_fname, x1, y1, ImageSize, ImageSize))
                    

 
