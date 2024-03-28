
import pandas as pd
import os

FR_csv_f = '/media/hero/Intel6/RACS_mid/rgz_resnet_fpn/FIRST_HT_FR_1.csv'

hetu_csv_f = '/media/hero/Intel6/RACS_mid/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_ht_flux_fixed_vlass_rm_lower_peak_centroid_fixed.csv'

HT_png_dir = '/media/hero/Intel5/FIRST_HT_png'

hetu_csv = pd.read_csv(hetu_csv_f)
source_names = hetu_csv['source_name'].values

FR_csv = pd.read_csv(FR_csv_f)
imagefilenames = FR_csv['imagefilename'].values
labels = FR_csv['label'].values

for m in range(len(source_names)):
    for n in range(len(imagefilenames)):
        sn = imagefilenames[n].replace('.png', '')
        if source_names[m] == sn:
           if labels[n] == 'fr1' or labels[n] == 'cj':
              os.system('mv /media/hero/Intel5/FIRST_HT_png/%s.png /media/hero/Intel5/fr1' % source_names[m])
           elif labels[n] == 'fr2':
              os.system('mv /media/hero/Intel5/FIRST_HT_png/%s.png /media/hero/Intel5/fr2' % source_names[m])       
           else:
              os.system('mv /media/hero/Intel5/FIRST_HT_png/%s.png /media/hero/Intel5/fr1' % source_names[m])       
           break
