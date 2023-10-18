import os,sys
import argparse
#from mpi4py import MPI
import linecache
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--inpdir', dest='inpdir', type=str, default='/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid/SB21834_split_fits_png', help='pred input png file directory')
parser.add_argument('--outdir', dest='outdir', type=str, default='/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid/SB21834_split_rms', help='output rms file directory')
#parser.add_argument('--fitslists', dest='fitslists', type=str, help='input fits lists')

args = parser.parse_args()


#comm=MPI.COMM_WORLD
#num_process=comm.Get_size()
#rank=comm.Get_rank()


input_dir = args.inpdir
file_nms = os.listdir(input_dir)

#with open(args.fitslists, 'r') as file:
#     linecount=len(file.readlines())

#pro_arr = np.array_split(np.arange(linecount),num_process)
#for n in pro_arr[rank]:
for fn in file_nms:
    if not fn.endswith('.fits'):
       continue
    #line = linecache.getline(args.fitslists, n+1)
    fits_file = fn #line.split('\n')[0]
    if os.path.exists('%s/%s_rms.fits' % (args.outdir, os.path.splitext(fits_file)[0])) and os.path.exists('%s/%s_bkg.fits' % (args.outdir, os.path.splitext(fits_file)[0])):
       print('%s/%s_rms.fits already exists' % (args.outdir, os.path.splitext(fits_file)[0]))
    else:
       os.system('BANE --cores 1 %s/%s --out %s/%s' % (input_dir, fits_file, args.outdir, os.path.splitext(fits_file)[0]))
       print("Successful generate %s" % fits_file)
