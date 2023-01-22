import os,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import astropy.wcs as wcs
import matplotlib as mpl
import numpy as np
import argparse
from mpi4py import MPI

def CoolColormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#000000', '#000069', '#00188a', '#0d6bff', '#1abaff',
                                                                 '#d9ffff', '#ffffff'], 256)

parser = argparse.ArgumentParser()
parser.add_argument('--inpdir', dest='inpdir', type=str, default='/o9000/MWA/GLEAM/hetu_images/deep_learn/inference_sets/FIRST_fits', help='pred input png file directory')
parser.add_argument('--outdir', dest='outdir', type=str, default='/o9000/MWA/GLEAM/hetu_images/deep_learn/inference_sets/FIRST_fits', help='output png file directory')
args = parser.parse_args()


def CoolColormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#000000', '#000069', '#00188a', '#0d6bff', '#1abaff',
                                                                 '#d9ffff', '#ffffff'], 256)

comm=MPI.COMM_WORLD
num_process=comm.Get_size()
rank=comm.Get_rank()


input_dir = args.inpdir
file_nms = os.listdir(input_dir)

pro_arr = np.array_split(np.arange(len(file_nms)),num_process)
for n in pro_arr[rank]:
#for fn in file_nms:
    fn = file_nms[n]
    if not fn.endswith('.fits'):
       continue
   
    fits_file = fn
    try:
       hdu = fits.open(os.path.join(input_dir, fits_file))[0]
    except:
       print("fits file %s is broken!" % fits_file)
       continue
       #sys.exit()

    if len(hdu.data.shape)==4:
       img_data = hdu.data[0,0,:,:]
    else :
       img_data = hdu.data
    w = wcs.WCS(hdu.header).celestial
    ax=plt.subplot(projection=w)
    #fig = plt.figure()
    outdir = args.outdir
    pngfile = os.path.splitext(fn)[0] + ".png"
    datamax = np.nanmax(img_data)
    datamin = np.nanmin(img_data)
    if np.isnan(datamax) and np.isnan(datamin):
       print(fits_file + " is all nan, skip it")
       continue
    datawide = datamax - datamin
    image_data = np.log10((hdu.data - datamin) / datawide * 1000 + 1) / 3 
    plt.imsave(os.path.join(outdir, pngfile), image_data, cmap=CoolColormap(),origin='lower')

    print("Successful generate %s" % fn)
    plt.clf()
