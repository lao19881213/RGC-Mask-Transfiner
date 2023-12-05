'''
The original code from https://github.com/ErikOsinga/MRP1,
developed by Erik Osinga.
'''
import sys

import numpy as np 
import numexpr as ne

from astropy.io import fits
from astropy.table import Table, join, vstack

import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process

from matplotlib import pyplot as plt

import math

from scipy.spatial import cKDTree

from utils import (select_on_size, load_in, distanceOnSphere
	, angular_dispersion_vectorized_n, angular_dispersion_vectorized_n_parallel_transport)

from general_statistics import (select_flux_bins1, select_size_bins1
		, select_flux_bins11, select_power_bins, select_physical_size_bins,
		select_size_bins_cuts_biggest_selection, select_flux_bins_cuts_biggest_selection)

import pandas as pd

######### SETUP ################ 
# THE FUNCTION MONTE_CARLO IS EXECUTED FOR TABLE tdata

# Parameters
n_sim = 1000
n = 600 #1600 #800 #1600
n_cores = 4 #multiprocessing.cpu_count()

parallel_transport = True
print ('Using parallel_transport = %s' %parallel_transport)

print ('Using RPA')

################################
def histedges_equalN(x, nbin):
	"""
 	Make nbin equal height bins
 	Call plt.hist(x, histedges_equalN(x,nbin))
	"""
	npt = len(x)
	return np.interp(np.linspace(0,npt,nbin+1),
					np.arange(npt),
					np.sort(x))

def random_data(tdata):
	'''
	Makes random data with the same no. of sources in the same area
	Better: just shuffle the array of position angles
	'''
	maxra = np.max(tdata['RA'])
	minra = np.min(tdata['RA'])
	maxdec = np.max(tdata['DEC'])
	mindec = np.min(tdata['DEC'])
	minpa = np.min(tdata['RPA'])
	maxpa = np.max(tdata['RPA'])
	length = len(tdata)

	rdata=Table()
	rdata['RA'] = np.random.randint(minra,maxra,length)
	rdata['DEC'] = np.random.randint(mindec,maxdec,length)
	rdata['RPA'] = np.random.randint(minpa,maxpa,length)
	return rdata

def parallel_datasets(number_of_simulations=n_sim/n_cores):
	print ('Number of simulations per core:' + str(number_of_simulations))
	np.random.seed() # changes the seed
	Sn_datasets = []
	
	ra = np.asarray(tdata['RA'])
	dec = np.asarray(tdata['DEC'])
	pa = np.asarray(tdata['RPA'])
	length = len(tdata)
	if redshift: z_best = np.asarray(tdata['z_best'])

	max_ra = np.max(ra)
	min_ra = np.min(ra)
	max_dec = np.max(dec)
	min_dec = np.min(dec)
       	
	rdata = Table()
	for i in range(int(number_of_simulations)):
		# print ('Simulation number: %i/%i'%(i,number_of_simulations))
		np.random.shuffle(pa)
		rdata['RA'] = ra
		rdata['DEC'] = dec
		if redshift: rdata['z_best'] = z_best
		rdata['RPA'] = pa
		if parallel_transport:
			Sn = angular_dispersion_vectorized_n_parallel_transport(rdata,n=n,redshift=redshift) # up to n nearest neighbours
		else:
			Sn = angular_dispersion_vectorized_n(rdata,n=n,redshift=redshift) # up to n nearest neighbours
		Sn_datasets.append(Sn)


	return Sn_datasets

def monte_carlo(totally_random=False,filename=''):
	'''
	Make (default) n_sim random data sets and calculate the Sn statistic

	If totally_random, then generate new positions and position angles instead
	of shuffeling the position angles among the sources
	'''
					# a 4 x 250 x n array for 1000 simulations on a 4 core system.
	Sn_datasets = [] # a n_core x n_sim/n_core x n array containing n_sim simulations of n different S_n
	print ('Starting '+ str(n_sim) + ' Monte Carlo simulations for n = 0 to n = ' + str(n))
	
	p = Pool(n_cores)
	# set up 4 processes that each do 1/num_cores simulations on a num_cores system
	print ('Using ' +str(n_cores) + ' cores')
	processes = [p.apply_async(parallel_datasets) for _ in range(n_cores)]
	# get the results into a list
	[Sn_datasets.append(process.get()) for process in processes]
	p.close()
		
	Sn_datasets = np.asarray(Sn_datasets)
	print ('Shape of Sn_datasets: ', Sn_datasets.shape)
	if Sn_datasets.shape[2] < n:
		# hard fix when n > len(tdata)
		Sn_datasets = Sn_datasets.reshape((n_sim,len(tdata)-1)) 
	else:
		Sn_datasets = Sn_datasets.reshape((n_sim,n))

	Result = Table()

	if n > len(tdata):
		# hard fix when n > len(tdata)
		for i in range(1,len(tdata)):
			Result['S_'+str(i)] = Sn_datasets[:,i-1] # the 0th element is S_1 and so on..
	else:
		for i in range(1,n+1):
			Result['S_'+str(i)] = Sn_datasets[:,i-1] # the 0th element is S_1 and so on..

	if parallel_transport:
		# np.save('./data/Sn_monte_carlo_PT'+filename,Sn_datasets)
		if n < 998:
			Result.write('./data/2D/Sn_monte_carlo_PT'+filename+'_%d.fits' % n,overwrite=True)
		else:
			print ('Creating csv file')
			Result.write('./data/2D/Sn_monte_carlo_PT'+filename+'_%d.csv' % n, overwrite=True)
	else:
		if n < 998:
			# np.save('./data/Sn_monte_carlo_'+filename,Sn_datasets)
			Result.write('./data/2D/Sn_monte_carlo_'+filename+'_%d.fits' % n,overwrite=True)
		else:
			print ('Creating csv file')
			Result.write('./data/2D/Sn_monte_carlo_'+filename+'_%d.csv' % n, overwrite=True)


'''
#Running just one file
redshift = True

filename = sys.argv[1]
print (filename)
# tdata = Table(fits.open('../redshift_bins/%s.fits'%filename)[1].data)
tdata = Table(fits.open('../%s.fits'%filename)[1].data)

tdata['RA'] = tdata['RA_2']
tdata['DEC'] = tdata['DEC_2']

z_available = np.invert(np.isnan(tdata['z_best']))  
z_zero = tdata['z_best'] == 0asdasdaw#
# also remove sources with redshift 0, these dont have 3D positions
z_available = np.logical_xor(z_available,z_zero)
print ('Number of sources with available redshift:', np.sum(z_available))
filename += '_redshift_' 

tdata_original = tdata
filename_original = filename

filename += '_astropy'

# fluxbins11 = select_flux_bins11(tdata_original)
# tdata = fluxbins11
# filename = filename_original + 'flux11_360'
# THE FUNCTION MONTE_CARLO IS EXECUTED FOR TABLE tdata WITH RA DEC and final_PA
tdata = tdata[z_available]
monte_carlo(totally_random=False,filename=filename)
'''

equal_width = False

#Running all the statistics without redshift
redshift = False

results_dir = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results'
filename = 'FRIIRGcat_final'#sys.argv[1]
print(filename)
tdata1 = pd.read_csv('%s/%s.csv' % (results_dir, filename)) #Table(fits.open('../%s.fits'%filename)[1].data)
#if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
#
#if position_angle: filename += '_PA' # only when using 'position_angle'
#
#try:
#	tdata['RA'] = tdata['RA_2']
#	tdata['DEC'] = tdata['DEC_2']
#except KeyError:
#	pass # then we're doing biggest_selection
tdata1 = tdata1[tdata1['LAS']>=80]
tdata2 = tdata1[tdata1['RA']>=90]
tdata = tdata2[tdata2['RA']<=270]
print('data length -->', len(tdata))
tdata_original = tdata
filename_original = filename

# # THE FUNCTION MONTE_CARLO IS EXECUTED FOR TABLE tdata WITH RA DEC and final_PA
monte_carlo(totally_random=False,filename=filename)


#if equal_width: 
#	fluxbins = select_flux_bins1(tdata_original)
#else:
#	fluxbins = select_flux_bins_cuts_biggest_selection(tdata_original)
#
#for key in fluxbins:
#	tdata = fluxbins[key]
#	filename = filename_original + 'flux%s'%key
#	
#	monte_carlo(totally_random=False,filename=filename)
#
#if equal_width:
#	sizebins = select_size_bins1(tdata_original)
#else:
#	sizebins = select_size_bins_cuts_biggest_selection(tdata_original)
#
#for key in sizebins:
#	tdata = sizebins[key]
#	filename = filename_original + 'size%s'%key
#	if len(tdata) != 0:
#		monte_carlo(totally_random=False,filename=filename)
#	else:
#		print ('This bin contains 0 sources:')
#		print (filename)
#
#
# fluxbins11 = select_flux_bins11(tdata_original)
# tdata = fluxbins11
# filename = filename_original + 'flux11'

# monte_carlo(totally_random=False,filename=filename)







