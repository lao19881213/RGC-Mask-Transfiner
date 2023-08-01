import os

fits_dir = '/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/HT/VLASS'
out_dir = '/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/HT/VLASS_final'



with open('../HT/vlass_ht.txt', 'r') as f:
     for line in f:
         line = line.strip()
         #pngn = line[0:19]
         fitsn = line.replace('.png', '.fits')
         print('cp %s/*%s %s' % (fits_dir, fitsn, out_dir))
         os.system('cp %s/*%s %s' % (fits_dir, fitsn, out_dir))

         
