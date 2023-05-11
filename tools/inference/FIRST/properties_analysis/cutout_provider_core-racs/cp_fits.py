import os

fits_dir = '/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/FRI/VLASS'
out_dir = '/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/FRI/VLASS_final'



with open('../FRI/vlass_fr1.txt', 'r') as f:
     for line in f:
         line = line.strip()
         #pngn = line[0:19]
         fitsn = line.replace('.png', '.fits')
         os.system('cp %s/*%s %s' % (fits_dir, fitsn, out_dir))

         
