import numpy
from astropy.table import Table
import requests
from PIL import Image
from io import BytesIO
from astropy.io import fits
import argparse
import pandas as pd
import os

def getimages(ra,dec,filters="grizy"):
    
    """Query ps1filenames.py service to get a list of images
    
    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """
    
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
    table = Table.read(url, format='ascii')
    return table


def geturl(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False):
    
    """Get URL for images in the table
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """
    
    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg","png","fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra,dec,filters=filters)
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           f"ra={ra}&dec={dec}&size={size}&format={format}")
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[numpy.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0,len(table)//2,len(table)-1]]
        for i, param in enumerate(["red","green","blue"]):
            url = url + "&{}={}".format(param,table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase+filename)
    return url


def getcolorim(ra, dec, size=240, output_size=None, filters="grizy", format="jpg"):
    
    """Get color image at a sky position
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png")
    Returns the image
    """
    
    if format not in ("jpg","png"):
        raise ValueError("format must be jpg or png")
    url = geturl(ra,dec,size=size,filters=filters,output_size=output_size,format=format,color=True)
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))
    return im


def getgrayim(ra, dec, size=240, output_size=None, filter="g", format="jpg"):
    
    """Get grayscale image at a sky position
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filter = string with filter to extract (one of grizy)
    format = data format (options are "jpg", "png")
    Returns the image
    """
    
    if format not in ("jpg","png"):
        raise ValueError("format must be jpg or png")
    if filter not in list("grizy"):
        raise ValueError("filter must be one of grizy")
    url = geturl(ra,dec,size=size,filters=filter,output_size=output_size,format=format)
    r = requests.get(url[0])
    im = Image.open(BytesIO(r.content))
    return im
 
parser = argparse.ArgumentParser()
parser.add_argument('--resultcsv', help='hetu results file')
parser.add_argument('--outdir', help='output directory')
args = parser.parse_args()

hetu_data = pd.read_csv(args.resultcsv)

labels = hetu_data['label'].values
ras_PanSTARRS = hetu_data['RAJ2000'].values
decs_PanSTARRS = hetu_data['DEJ2000'].values
ras = hetu_data['centre_ra'].values
decs = hetu_data['centre_dec'].values
source_names = hetu_data['source_name'].values
boxs = hetu_data['box'].values
objIDs = hetu_data['objID'].values


#ra = 256.073204
#dec = 60.654467 

clns = {'fr1': 'FRI', 'fr2': 'FRII', 'ht': 'HT', 'cj': 'CJ'}

tags_non = []

for m in range(len(labels)):
        for cln in clns.keys():
            #print(cln)
            if(labels[m]==cln):
               #print(cln)
               fits_fn = '%s_%s' % ('PanSTARRS', source_names[m])
               if os.path.isfile(f'%s/%s/%s/%s_%s.fits' % (args.outdir, clns[cln], 'PanSTARRS', fits_fn, 'r')):
                  print('%s_%s.fits already exists!' % ('PanSTARRS', fits_fn))
               else:
                  try:
                     fitsurl = geturl(ras_PanSTARRS[m], decs_PanSTARRS[m], size=350, filters="r", format="fits")
                     fh = fits.open(fitsurl[0])
                     print('fetching images....\n')          
                     #band='ugriz' 
                     #print(len(hdul_lists))
                     print(fh)
                     print('writing %s' % source_names[m], end='')
                     #counter = 0
                     fits_fn = '%s_%s' % ('PanSTARRS', source_names[m])
                     fh.writeto(f'%s/%s/%s/%s_%s.fits' % (args.outdir, clns[cln], 'PanSTARRS', fits_fn, 'r'),
                                         overwrite=True)
                     print('\n')
                  except:
                     try:
                        fitsurl = geturl(ras[m], decs[m], size=350, filters="r", format="fits")
                        fh = fits.open(fitsurl[0])
                        print('fetching images....\n')
                        #band='ugriz' 
                        print('writing %s' % source_names[m], end='')
                        #counter = 0
                        fits_fn = '%s_%s' % ('PanSTARRS', source_names[m])
                        fh.writeto(f'%s/%s/%s/%s_%s.fits' % (args.outdir, clns[cln], 'PanSTARRS', fits_fn, 'r'),
                                            overwrite=True)
                        print('\n')
                     except:
                        print('Not found ', source_names[m])#name)
                        tags_non.append("{},{},{},{},{}".format(m,labels[m],source_names[m],ras[m],decs[m]))


resultsData_non = tags_non
with open(os.path.join('./', 'NOT_FOUND_PanSTARRS.txt'), 'w') as fn:
     fn.write(os.linesep.join(resultsData_non))
