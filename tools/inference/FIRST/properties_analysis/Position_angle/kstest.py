from scipy import stats
import pandas as pd
import numpy as np

def KSTEST(tdata,redshift):
	if redshift:
		z_available = np.invert(np.isnan(tdata['z_best']))
		z_zero = tdata['z_best'] == 0#
		# also remove sources with redshift 0, these dont have 3D positions
		z_available = np.logical_xor(z_available,z_zero)
		print ('Number of sources with available redshift:', np.sum(z_available))
		tdata = tdata[z_available] # use redshift data for power bins and size bins !!!

	position_angles = np.asarray(tdata['RPA'])

	statistic, p_value = stats.kstest(position_angles,stats.uniform(loc=0.0, scale=180.0).cdf)

	#print ('KS test: ', filename)
	print ('Statistic: %f | p value (per cent): %f'%(statistic,p_value*100))

	return p_value*100 # in percent


results_dir = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results'
filename = 'FRIIRGcat_final'#sys.argv[1]
print(filename)
tdata1 = pd.read_csv('%s/%s.csv' % (results_dir, filename)) #Table(fits.open('../%s.fits'%filename)[1].data)
tdata2 = tdata1[tdata1['RA']>=90]
tdata = tdata2[tdata2['RA']<=270]
print('data length -->', len(tdata))
p_value = KSTEST(tdata=tdata, redshift=False)
print(p_value)

tdata3 = tdata[tdata['z'].notnull()]
print('data length have redshift -->', len(tdata3))
p_value = KSTEST(tdata=tdata3, redshift=False)
print(p_value)
