import os

clsn = 'FRII'
FIRST_dir = '/home/data0/lbq/inference_data/FIRST_HeTu_png/%s_selected' % clsn
pngfs = os.listdir(FIRST_dir)
NVSS_dir = '/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/%s/NVSS' % clsn
out_dir = '/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/%s/NVSS4flux' % clsn

for m in range(len(pngfs)):
      fitsf = 'NVSS_%s.fits' % (os.path.splitext(pngfs[m])[0])
      print('cp %s/%s %s' % (NVSS_dir, fitsf, out_dir))
      os.system('cp %s/%s %s' % (NVSS_dir, fitsf, out_dir))


