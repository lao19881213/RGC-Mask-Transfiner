import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii


data = ascii.read("extended_sources.csv", encoding='utf-8')
ra = data['RA']
dec = data['DEC']
with open ('sources_name.csv','a') as fn:
    fn.write('RA, DEC, Source\n')
for i in range(len(ra)):
    target = SkyCoord(ra[i], dec[i], unit=(u.deg, u.deg))
    with open ('sources_name.csv','a') as fn:
        fn.write('%f, %f, %s\n'%(ra[i],dec[i],target))