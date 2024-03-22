import bdsf
import sys
import os

for m in range(2,5):
    img_list = '/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid/part%d_fits_fixed.txt' % m
    
    data_dir = '/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid/part%d_conv' % m
    
    cat_dir = '/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid/part%d_pybdsf' % m
    
    with open(img_list) as f:
         for line in f:
             fits_n = line.split('\n')[0]
             cat_n = fits_n.split('.')[3]
             if os.path.isfile('%s/%s_gaussian_list.csv' % (cat_dir, cat_n)) and os.path.isfile('%s/%s_source_list.csv' % (cat_dir, cat_n)):
                print('catalogs already exists for %s' % cat_n)
             else: 
                img = bdsf.process_image('%s/%s' % (data_dir, fits_n), 
                             thresh = 'hard', 
                             advanced_opts=True, 
                             rms_box = (150, 30), 
                             atrous_do = True,
                             atrous_jmax = 3,
                             mean_map = 'zero',
                             group_by_isl = True)
                
                #cat_n = fits_n.split('.')[3]
                img.write_catalog(outfile='%s/%s_gaussian_list.csv' % (cat_dir, cat_n), 
                        catalog_type='gaul', clobber=True, format='csv')
                img.write_catalog(outfile='%s/%s_source_list.csv' % (cat_dir, cat_n), 
                        catalog_type='srl', clobber=True, format='csv')


#img.show_fit()
