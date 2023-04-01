import urllib
from astropy import units as u
from astroquery.sdss import SDSS as astroSDSS
from astropy import coordinates

parser = argparse.ArgumentParser()
parser.add_argument('--resultcsv', help='hetu results file')
parser.add_argument('--outdir', help='output directory')
parser.add_argument('--surveys', default='0', type=str, help='surveys id, formats: 0,1,...')
args = parser.parse_args()

hetu_data = pd.read_csv(args.resultcsv)

labels = hetu_data['label'].values
ras = hetu_data['centre_ra'].values
decs = hetu_data['centre_dec'].values
source_names = hetu_data['source_name'].values
boxs = hetu_data['box'].values

#ra = 256.073204
#dec = 60.654467 

clns = {'fr1': 'FRI', 'fr2': 'FRII', 'ht': 'HT', 'cj': 'CJ'}

tags_non = []

for m in range(len(labels)):
        for cln in clns.keys():
            #print(cln)
            if(labels[m]==cln):
               position = coordinates.SkyCoord(ra=ras[m], dec=decs[m], unit=(u.deg, u.deg))
               try:               
                  result = astroSDSS.query_region(position)
                  
                  hdul_lists = astroSDSS.get_images(matches=result, band='r',  data_release=16)[0]
                  fits_fn = '%s_%s.fits' % ('SDSS', source_names[m]) 
                  hdul_lists.writeto("%s/%s/%s/%s" % (args.outdir, clns[cln], 'SDSS', fits_fn), overwrite=True)
                  #hdul_lists = astroSDSS.get_images(coordinates=position, radius = 5.0*u.arcmin, band='r',  data_release=16)
               except:
                  print('Not found ', source_names[m])#name)
                  tags_non.append("{},{},{},{},{}".format(m,labels[m],source_names[m],ras[m],decs[m]))


resultsData_non = tags_non
with open(os.path.join('./', 'NOT_FOUND_NED_%s.txt' % cln), 'w') as fn:
     fn.write(os.linesep.join(resultsData_non))

