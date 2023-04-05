import bdsf
import sys

img = bdsf.process_image('J000410+082157.fits')

img.show_fit()

#img.export_image(clobber=True)

#img.export_image(img_type='gaus_model', clobber=True)

