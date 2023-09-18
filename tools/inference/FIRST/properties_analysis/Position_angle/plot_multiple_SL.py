import sys
sys.path.insert(0,'/net/reusel/data1/osinga/anaconda2')

import numpy as np 
import numexpr as ne
import math

from astropy.io import fits
from astropy.table import Table, join, vstack

from scipy.spatial import cKDTree
from scipy.stats import norm

import matplotlib
from matplotlib import pyplot as plt

from utils import angular_dispersion_vectorized_n, distanceOnSphere, angular_dispersion_vectorized_n_parallel_transport

from general_statistics import (select_flux_bins1, select_size_bins1
		, select_flux_bins11, select_power_bins, select_physical_size_bins
		, select_flux_bins_cuts_biggest_selection, select_size_bins_cuts_biggest_selection)


"""
FOR ANY PLOTTING CONSIDERING THE SL-vs-n 2D 
"""


parallel_transport = True
print ('Using parallel_transport = %s' %parallel_transport)

position_angle = sys.argv[2]
if position_angle == 'True':
	position_angle = True
else:
	position_angle = False
if position_angle:
	print ('Using position_angle')
else: 
	print ('Using final_PA')

n = 800

def histedges_equalN(x, nbin):
	"""
 	Make nbin equal height bins
 	Call plt.hist(x, histedges_equalN(x,nbin))
	"""
	npt = len(x)
	return np.interp(np.linspace(0,npt,nbin+1),
					np.arange(npt),
					np.sort(x))

def tick_function(X):
	return ["%.2f" % z for z in X]

def SL_vs_n_plot(filename,all_sl,angular_radius,ending_n=180,ax=None,label=None):
	"""
	Make a plot of the SL (Significance level) statistic vs n.
	"""

	if ending_n > len(all_sl):
		print ('ending_n = %i, but this tdata only contains N=%i sources'%(ending_n,len(all_sl)+1))
		ending_n = len(all_sl)
		print ('Setting ending_n=%i'%ending_n)

	starting_n = 1 # In the data file the S_1 is the first column

	# fig, axarr = plt.subplots(2, sharex=True, gridspec_kw= {'height_ratios':[3, 1]})

	ax.plot(range(starting_n,ending_n+1),np.log10(all_sl),color='k',ls='solid',label=label)
	# print np.argmin(all_sl)
	# axarr[0].set_title('SL vs n for '+filename+'\n\n')
	print ('SL vs n for '+filename+'\n\n')
	# fig.suptitle(title,fontsize=14)

	ax.set_xlim(2,ending_n)
	ax.set_ylim(ylim_lower,0)
	
	if angular_radius is not None:
		ax2 = ax.twiny()
		ax2.set_xlabel('angular radius (degrees)',fontsize=14)
		
		xticks = ax.get_xticks()
		print (np.asarray(xticks,dtype='int'))
		if np.asarray(xticks)[-1] > ending_n: # then we need to set the ax.set_xlim 1 tick less
			ax.set_xlim(2,np.asarray(xticks)[-2])
			xticks = ax.get_xticks()

		ax2.set_xticks(xticks)
		xticks2 = np.append(0,angular_radius)[np.asarray(xticks,dtype='int')]
		ax2.set_xticklabels(tick_function(xticks2))
		ax2.tick_params(labelsize=12)

def Sn_vs_n(tdata,Sn_mc,Sn_data,filename,angular_radius,ending_n=180):
	"""
	Calculate SL (Significance level) statistic vs n.
	"""

	# hard fix for when n > amount of sources in a bin 
	if ending_n > len(tdata):
		print ('ending_n = %i, but this tdata only contains N=%i sources'%(ending_n,len(tdata)))
		ending_n = len(tdata)-1
		print ('Setting ending_n=%i'%ending_n)

	starting_n = 1 # In the data file the S_1 is the first column
	n_range = range(0,ending_n) # index 0 corresponds to Sn_1 

	all_sn = []
	all_sn_mc = []
	all_sl = [] # (length n list containing SL_1 to SL_80)
	all_std = [] # contains the standard deviations of the MC simulations
	N = len(tdata)
	jain_sigma = (0.33/N)**0.5
	for n in n_range:
		# print 'Now doing n = ', n+starting_n , '...'  
		Sn_mc_n = np.asarray(Sn_mc['S_'+str(n+starting_n)])
		av_Sn_mc_n = np.mean(Sn_mc_n)
		sigma = np.std(Sn_mc_n)
		if sigma == 0:
			print ('Using Jain sigma for S_%i'%(n+starting_n))
			sigma = jain_sigma

		Sn_data_n = Sn_data[n]
		# print Sn_data_n, av_Sn_mc_n
		SL = 1 - norm.cdf(   (Sn_data_n - av_Sn_mc_n) / (sigma)   )
		all_sn.append(Sn_data_n)
		all_sl.append(SL)
		all_sn_mc.append(av_Sn_mc_n)
		all_std.append(sigma)
	
	Results = Table()
	Results['n'] = range(starting_n,ending_n+1)
	Results['Sn_data'] = all_sn
	Results['SL'] = all_sl
	Results['Sn_mc'] = all_sn_mc
	Results['angular_radius'] = angular_radius
	Results['sigma_S_n'] = all_std
	Results['jain_sigma'] = [jain_sigma]
	if parallel_transport:
		Results.write('./data/2D/Statistics_PT'+filename+'_results.fits',overwrite=True)
	else:
		Results.write('./data/2D/Statistics_'+filename+'_results.fits',overwrite=True)
		
	return all_sl, angular_radius

def angular_radius_vs_n(tdata,filename,n=180,starting_n=2):
	"""
	Make a plot of the angular separation vs the amount of neighbours n
	"""

	# hard fix for when n > amount of sources in a bin 
	if n > len(tdata):
		print ('Announcement about %s'%filename)
		print ('n = %i, but this bin only contains N=%i sources'%(n,len(tdata)))
		n = len(tdata)-1
		print ('Setting n=%i'%n)

	n_range = range(starting_n,n+1) 

	N = len(tdata)
	RAs = np.asarray(tdata['RA'])
	DECs = np.asarray(tdata['DEC'])
	position_angles = np.asarray(tdata['final_PA'])

	#convert RAs and DECs to an array that has following layout: [[x1,y1,z1],[x2,y2,z2],etc]
	x = np.cos(np.radians(RAs)) * np.cos(np.radians(DECs))
	y = np.sin(np.radians(RAs)) * np.cos(np.radians(DECs))
	z = np.sin(np.radians(DECs))
	coordinates = np.vstack((x,y,z)).T
	
	#make a KDTree for quick NN searching	
	coordinates_tree = cKDTree(coordinates,leafsize=16)

	# for every source: find n closest neighbours, calculate median dist to furthest
	furthest_nearestN = np.zeros((N,n),dtype='int') # array of shape (N,n) that contains index of furthest_nearestN
	for i in range(N):
		indices = coordinates_tree.query(coordinates[i],k=n,p=2)[1] # include source itself
		furthest_nearestN[i] = indices
		# source = [RAs[i],DECs[i]]

	temp = np.arange(0,N).reshape(N,1) - np.zeros(n,dtype='int') # Array e.g.
		# to make sure we calc the distance between				#[[0,0,0,0],
		# the current source and the neighbours					#[[1,1,1,1]]
	distance = distanceOnSphere(RAs[temp],DECs[temp]
						,RAs[furthest_nearestN],DECs[furthest_nearestN])
	
	median = np.median(distance,axis=0)
	# std = np.std(distance,axis=0)

	assert median.shape == (n,) # array of the median of (the distance to the furthest 
								# neighbour n for all sources), for every n. 

	plt.plot(n_range,median[starting_n-1:],color='k') # index 1 corresponds to n=2
	# plt.title('Angular radius vs n for ' + filename)
	plt.ylabel('Median angular radius (degrees)',fontsize=14)
	plt.xlabel(r'$n_n$',fontsize=14)
	plt.tick_params(labelsize=12)
	title = 'Median angular radius vs n'
	plt.title(title,fontsize=16)
	# plt.savefig('/data1/osinga/figures/statistics/show/angular_radius_vs_n_'+filename+'.png',overwrite=True)
	# plt.savefig('./figures/angular_radius_vs_n_'+filename+'.png',overwrite=True)
	# plt.show()
	plt.close()
		
	return median

def statistics(filename,tdata,redshift=False,recalculate=False):
	if parallel_transport:
		Sn_mc = Table(fits.open('./data/2D/Sn_monte_carlo_PT'+filename+'.fits')[1].data)
	else:
		Sn_mc = Table(fits.open('./data/2D/Sn_monte_carlo_'+filename+'.fits')[1].data)
	
	if not recalculate: # Not forcing a recalculation
		try:  # if saved, no need to calculate again.
			if parallel_transport:
				Results = Table(fits.open('./data/2D/Statistics_PT'+filename+'_results.fits')[1].data)
			else:
				Results= Table(fits.open('./data/2D/Statistics_'+filename+'_results.fits')[1].data)

			all_sl = np.asarray(Results['SL'])
			angular_radius = np.asarray(Results['angular_radius'])

		except (IOError, KeyError) as e: # calculating anywayss
			print ('Calculating Stuff')
			# calculate angular radius for number of nn, 
			angular_radius = angular_radius_vs_n(tdata,filename,n)
			# calculate SN data
			if parallel_transport:
				Sn_data = angular_dispersion_vectorized_n_parallel_transport(tdata,n,redshift)
			else:
				Sn_data = angular_dispersion_vectorized_n(tdata,n,redshift)
			# calculate SL vs n
			all_sl, angular_radius = Sn_vs_n(tdata,Sn_mc,Sn_data,filename,angular_radius,n)

	else: # Forcing a recalculation, whether the file exists or not
		print ('Calculating Stuff')
		# calculate angular radius for number of nn, 
		angular_radius = angular_radius_vs_n(tdata,filename,n)
		# calculate SN data
		if parallel_transport:
			Sn_data = angular_dispersion_vectorized_n_parallel_transport(tdata,n,redshift)
		else:
			Sn_data = angular_dispersion_vectorized_n(tdata,n,redshift)
		# calculate SL vs n
		all_sl, angular_radius = Sn_vs_n(tdata,Sn_mc,Sn_data,filename,angular_radius,n)

	return all_sl, angular_radius

def biggest_selection_2D():
	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	#Running all the statistics without redshift
	redshift = False

	filename = 'biggest_selection'# EDIT THIS
	print ('\n!! Filename:', filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	if position_angle: filename += '_PA' # only when using 'position_angle'
	try:
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
	except KeyError:
		pass # then we're doing biggest_selection
	tdata_original = tdata
	filename_original = filename

	title = 'Initial subsample: 2D analysis\n\n'# EDIT THIS
	all_sl, angular_radius = statistics(filename,tdata,redshift)
	
	fig, ax = plt.subplots()
	# have to give angular_radius = None for multiple plots in same figure
	SL_vs_n_plot(filename,all_sl,angular_radius,n,ax)

	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)

	ax.tick_params(labelsize=12)
	
	# plt.show()
	plt.title(title,fontsize=16)
	plt.tight_layout()
	plt.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_biggest_selection.pdf')
	plt.close()

def biggest_selection_2D_flux():

	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	redshift = False

	filename = 'biggest_selection'# EDIT THIS
	print ('\n!! Filename:', filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	if position_angle: filename += '_PA' # only when using 'position_angle'
	try:
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
	except KeyError:
		pass # then we're doing biggest_selection
	tdata_original = tdata
	filename_original = filename

	title = 'Initial subsample: Flux bins\n\n'# EDIT THIS
	fluxbins = select_flux_bins1(tdata_original)

	fig, ax = plt.subplots()
	
	for key in sorted(fluxbins):
		tdata = fluxbins[key]
		filename = filename_original + 'flux%s'%key
		all_sl, angular_radius = statistics(filename,tdata,redshift)
		# have to give angular_radius = None for multiple plots in same figure
		# since different subsets have different angular radii per number of nn
		if key == '3':
			SL_vs_n_plot(filename,all_sl,angular_radius,n,ax,label='Bin %s'%key)
			
			min_sl = np.min(all_sl)
			position = angular_radius[np.argmin(all_sl)]
			print ('Minimum SL: ', min_sl)
			print ('At angular radius: %f degrees'%position)
			
			print '\n'

		else:
			ax.plot(range(1,n+1),np.log10(all_sl),ls='solid',label='Bin %s'%key)
		

	ax.set_xlim(2,n)
	ax.set_ylim(ylim_lower,0)

	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	title = ''
	plt.title(title,fontsize=16)
	plt.tight_layout()
	# plt.show()
	plt.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_biggest_selection_fluxbins.pdf')

	return min_sl,position
	# wait until later

	# fluxbins11 = select_flux_bins11(tdata_original)
	# tdata = fluxbins11
	# filename = filename_original + 'flux11'

def biggest_selection_2D_size():	

	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	redshift = False

	filename = 'biggest_selection'# EDIT THIS
	print ('\n!! Filename:', filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	if position_angle: filename += '_PA' # only when using 'position_angle'
	try:
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
	except KeyError:
		pass # then we're doing biggest_selection
	tdata_original = tdata
	filename_original = filename

	title = 'Initial subsample: Size bins\n\n'# EDIT THIS
	sizebins = select_size_bins1(tdata_original,True)

	fig, ax = plt.subplots()
	
	for key in sorted(sizebins):
		tdata = sizebins[key]
		filename = filename_original + 'size%s'%key # important !! (for loading data)

		all_sl, angular_radius = statistics(filename,tdata,redshift)
		
		min_sl = np.min(all_sl)
		position = angular_radius[np.argmin(all_sl)]
		print ('For Bin %s'%key)
		print ('Minimum SL: ', min_sl)
		print ('At angular radius: %f degrees'%position)
		# have to give angular_radius = None for multiple plots in same figure
		# since different subsets have different angular radii per number of nn
		if key == '3':
			SL_vs_n_plot(filename,all_sl,angular_radius,n,ax,label='Bin %s'%key)
			min_sl3 = np.min(all_sl)
			position3 = angular_radius[np.argmin(all_sl)]
			print ('Minimum SL: ', min_sl)
			print ('At angular radius: %f degrees'%position)
			print '\n'
		else:
			ax.plot(range(1,n+1),np.log10(all_sl),ls='solid',label='Bin %s'%key)

	ax.set_xlim(2,n)
	ax.set_ylim(ylim_lower,0)
	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	title = ''
	plt.title(title,fontsize=16)

	plt.tight_layout()
	# plt.show()
	plt.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_biggest_selection_sizebins.pdf')
	return min_sl3,position3

def value_added_selection_2D():
	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	#Running all the statistics without redshift
	redshift = False

	filename = 'value_added_selection' # EDIT THIS
	print ('\n!! Filename:', filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	if position_angle: filename += '_PA' # only when using 'position_angle'
	try:
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
	except KeyError:
		pass # then we're doing biggest_selection
	tdata_original = tdata
	filename_original = filename

	title = 'Value-added subset: 2D analysis\n\n'# EDIT THIS
	all_sl, angular_radius = statistics(filename,tdata,redshift)
	
	fig, ax = plt.subplots()
	# have to give angular_radius = None for multiple plots in same figure
	SL_vs_n_plot(filename,all_sl,angular_radius,n,ax)

	ax.set_xlim(2,n)
	ax.set_ylim(ylim_lower,0)
	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	plt.title(title,fontsize=16)
	plt.tight_layout()
	# plt.show()
	plt.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_value_added_subset.pdf')

def value_added_selection_2D_flux(equal_width=False):

	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	redshift = False

	filename = 'value_added_selection'# EDIT THIS
	print ('\n!! Filename:', filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	if position_angle: filename += '_PA' # only when using 'position_angle'
	try:
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
	except KeyError:
		pass # then we're doing biggest_selection
	tdata_original = tdata
	filename_original = filename

	title = 'Value-added subset: Flux bins\n\n'# EDIT THIS

	if equal_width: 
		fluxbins = select_flux_bins1(tdata_original)
	else:
		fluxbins = select_flux_bins_cuts_biggest_selection(tdata_original)

	fig, ax = plt.subplots()
	
	for key in sorted(fluxbins):
		tdata = fluxbins[key]
		filename = filename_original + 'flux%s'%key
		all_sl, angular_radius = statistics(filename,tdata,redshift)
		# have to give angular_radius = None for multiple plots in same figure
		# since different subsets have different angular radii per number of nn
		if key == '3':
			SL_vs_n_plot(filename,all_sl,angular_radius,n,ax,label='Bin %s'%key)
			min_sl = np.min(all_sl)
			position = angular_radius[np.argmin(all_sl)]
			print ('Minimum SL: ', min_sl)
			print ('At angular radius: %f degrees'%position)
			print '\n'
		else:
			# hard fix for when n > amount of sources in a bin 
			ending_n = n
			if ending_n > len(tdata):
				print ('n = %i, but this tdata only contains N=%i sources'%(n,len(tdata)))
				ending_n = len(tdata)-1
				print ('Setting n=%i'%ending_n)
			ax.plot(range(1,ending_n+1),np.log10(all_sl),ls='solid',label='Bin %s'%key)
		

	ax.set_xlim(2,n)
	ax.set_ylim(ylim_lower,0)
	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	plt.title(title,fontsize=16)
	plt.tight_layout()

	# plt.show()
	plt.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_value_added_subset_fluxbins.pdf')
	return min_sl,position
	# wait until later

	# fluxbins11 = select_flux_bins11(tdata_original)
	# tdata = fluxbins11
	# filename = filename_original + 'flux11'

def value_added_selection_2D_size(equal_width=False):	

	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	redshift = False

	filename = 'value_added_selection'# EDIT THIS
	print ('\n!! Filename:', filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	if position_angle: filename += '_PA' # only when using 'position_angle'
	try:
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
	except KeyError:
		pass # then we're doing biggest_selection
	tdata_original = tdata
	filename_original = filename

	title = 'Value-added subset: Size bins\n\n'# EDIT THIS
	
	if equal_width:
		sizebins = select_size_bins1(tdata_original)
	else:
		sizebins = select_size_bins_cuts_biggest_selection(tdata_original)

	fig, ax = plt.subplots()
	
	for key in sorted(sizebins):
		tdata = sizebins[key]
		filename = filename_original + 'size%s'%key # important !! (for loading data)

		all_sl, angular_radius = statistics(filename,tdata,redshift)
		# have to give angular_radius = None for multiple plots in same figure
		# since different subsets have different angular radii per number of nn
		if key == '3':
			SL_vs_n_plot(filename,all_sl,angular_radius,n,ax,label='Bin %s'%key)
			min_sl = np.min(all_sl)
			position = angular_radius[np.argmin(all_sl)]
			print ('Minimum SL: ', min_sl)
			print ('At angular radius: %f degrees'%position)
			print '\n'
		else:
			ax.plot(range(1,n+1),np.log10(all_sl),ls='solid',label='Bin %s'%key)

	ax.set_xlim(2,n)
	ax.set_ylim(ylim_lower,0)
	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	plt.title(title,fontsize=16)
	plt.tight_layout()
	# plt.show()
	plt.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_value_added_subset_sizebins.pdf')
	return min_sl,position

def value_added_selection_MG_2D():
	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	#Running all the statistics without redshift
	redshift = False

	filename = 'value_added_selection_MG'  # EDIT THIS
	print ('\n!! Filename:', filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	if position_angle: filename += '_PA' # only when using 'position_angle'
	try:
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
	except KeyError:
		pass # then we're doing biggest_selection
	tdata_original = tdata
	filename_original = filename

	title = 'Connected lobes subset: 2D analysis\n\n' # EDIT THIS
	all_sl, angular_radius = statistics(filename,tdata,redshift)
	
	fig, ax = plt.subplots()
	# have to give angular_radius = None for multiple plots in same figure
	SL_vs_n_plot(filename,all_sl,angular_radius,n,ax)

	ax.set_xlim(2,n)
	ax.set_ylim(ylim_lower,0)

	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	plt.title(title,fontsize=16)
	plt.tight_layout()
	# plt.show()
	plt.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_connected_lobes_subset.pdf')

def value_added_selection_MG_2D_flux(equal_width=False):

	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	redshift = False

	filename = 'value_added_selection_MG'# EDIT THIS
	print ('\n!! Filename:', filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	if position_angle: filename += '_PA' # only when using 'position_angle'
	try:
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
	except KeyError:
		pass # then we're doing biggest_selection
	tdata_original = tdata
	filename_original = filename

	title = 'Connected lobes subset: Flux bins\n\n'# EDIT THIS
	if equal_width:
		fluxbins = select_flux_bins1(tdata_original)
	else:
		fluxbins = select_flux_bins_cuts_biggest_selection(tdata_original)

	fig, ax = plt.subplots()
	
	for key in sorted(fluxbins):
		tdata = fluxbins[key]
		filename = filename_original + 'flux%s'%key
		all_sl, angular_radius = statistics(filename,tdata,redshift)
		# have to give angular_radius = None for multiple plots in same figure
		# since different subsets have different angular radii per number of nn
		if key == '3':
			SL_vs_n_plot(filename,all_sl,angular_radius,n,ax,label='Bin %s'%key)
			min_sl = np.min(all_sl)
			position = angular_radius[np.argmin(all_sl)]
			print ('Minimum SL: ', min_sl)
			print ('At angular radius: %f degrees'%position)
			print '\n'
		else:
						# hard fix for when n > amount of sources in a bin 
			ending_n = n
			if ending_n > len(tdata):
				print ('n = %i, but this tdata only contains N=%i sources'%(n,len(tdata)))
				ending_n = len(tdata)-1
				print ('Setting n=%i'%ending_n)
			ax.plot(range(1,ending_n+1),np.log10(all_sl),ls='solid',label='Bin %s'%key)
		


	ax.set_xlim(2,n)
	ax.set_ylim(-4.5,0)

	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	plt.title(title,fontsize=16)
	plt.tight_layout()
	# plt.show()
	plt.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_connected_lobes_subset_fluxbins.pdf')
	return min_sl,position

	# wait until later

	# fluxbins11 = select_flux_bins11(tdata_original)
	# tdata = fluxbins11
	# filename = filename_original + 'flux11'

def value_added_selection_MG_2D_size(equal_width=False):	

	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	redshift = False

	filename = 'value_added_selection_MG'# EDIT THIS
	print ('\n!! Filename:', filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	if position_angle: filename += '_PA' # only when using 'position_angle'
	try:
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
	except KeyError:
		pass # then we're doing biggest_selection
	tdata_original = tdata
	filename_original = filename

	title = 'Connected lobes subset: Size bins\n\n'# EDIT THIS
	if equal_width:
		sizebins = select_size_bins1(tdata_original)
	else:
		sizebins = select_size_bins_cuts_biggest_selection(tdata_original)

	fig, ax = plt.subplots()
	
	for key in sorted(sizebins):
		tdata = sizebins[key]
		filename = filename_original + 'size%s'%key # important !! (for loading data)

		all_sl, angular_radius = statistics(filename,tdata,redshift)
		# have to give angular_radius = None for multiple plots in same figure
		# since different subsets have different angular radii per number of nn
		if key == '3':
			SL_vs_n_plot(filename,all_sl,angular_radius,n,ax,label='Bin %s'%key)
			min_sl = np.min(all_sl)
			position = angular_radius[np.argmin(all_sl)]
			print ('Minimum SL: ', min_sl)
			print ('At angular radius: %f degrees'%position)
			print '\n'
		else:
			# hard fix for when n > amount of sources in a bin 
			ending_n = n
			if ending_n > len(tdata):
				print ('n = %i, but this tdata only contains N=%i sources'%(n,len(tdata)))
				ending_n = len(tdata)-1
				print ('Setting n=%i'%ending_n)
			ax.plot(range(1,ending_n+1),np.log10(all_sl),ls='solid',label='Bin %s'%key)
		
	ax.set_xlim(2,n)
	ax.set_ylim(ylim_lower,0)
	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	plt.title(title,fontsize=16)
	plt.tight_layout()
	# plt.show()
	plt.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_connected_lobes_subset_sizebins.pdf')
	return min_sl,position

def value_added_selection_NN_2D():
	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	#Running all the statistics without redshift
	redshift = False

	filename = 'value_added_selection_NN'  # EDIT THIS
	print ('\n!! Filename:', filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	if position_angle: filename += '_PA' # only when using 'position_angle'
	try:
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
	except KeyError:
		pass # then we're doing biggest_selection
	tdata_original = tdata
	filename_original = filename

	title = 'Isolated lobes subset: 2D analysis\n\n' # EDIT THIS
	all_sl, angular_radius = statistics(filename,tdata,redshift)
	
	fig, ax = plt.subplots()
	# have to give angular_radius = None for multiple plots in same figure
	SL_vs_n_plot(filename,all_sl,angular_radius,n,ax)

	ax.set_xlim(2,n)
	ax.set_ylim(ylim_lower,0)

	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	plt.title(title,fontsize=16)
	plt.tight_layout()
	# plt.show()
	plt.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_isolated_lobes_subset.pdf')

def value_added_selection_NN_2D_flux(equal_width=False):

	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	redshift = False

	filename = 'value_added_selection_NN'# EDIT THIS
	print ('\n!! Filename:', filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	if position_angle: filename += '_PA' # only when using 'position_angle'
	try:
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
	except KeyError:
		pass # then we're doing biggest_selection
	tdata_original = tdata
	filename_original = filename

	title = 'Isolated lobes subset: Flux bins\n\n'# EDIT THIS
	if equal_width:
		fluxbins = select_flux_bins1(tdata_original)
	else:
		fluxbins = select_flux_bins_cuts_biggest_selection(tdata_original)

	fig, ax = plt.subplots()
	
	for key in sorted(fluxbins):
		tdata = fluxbins[key]
		filename = filename_original + 'flux%s'%key
		all_sl, angular_radius = statistics(filename,tdata,redshift)
		# have to give angular_radius = None for multiple plots in same figure
		# since different subsets have different angular radii per number of nn
		if key == '3':
			SL_vs_n_plot(filename,all_sl,angular_radius,n,ax,label='Bin %s'%key)
			min_sl = np.min(all_sl)
			position = angular_radius[np.argmin(all_sl)]
			print ('Minimum SL: ', min_sl)
			print ('At angular radius: %f degrees'%position)
			print '\n'
		else:
						# hard fix for when n > amount of sources in a bin 
			ending_n = n
			if ending_n > len(tdata):
				print ('n = %i, but this tdata only contains N=%i sources'%(n,len(tdata)))
				ending_n = len(tdata)-1
				print ('Setting n=%i'%ending_n)
			ax.plot(range(1,ending_n+1),np.log10(all_sl),ls='solid',label='Bin %s'%key)
		


	ax.set_xlim(2,n)
	ax.set_ylim(-4.5,0)

	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	plt.title(title,fontsize=16)
	plt.tight_layout()
	# plt.show()
	# EDIT THIS
	plt.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_isolated_lobes_subset_fluxbins.pdf')
	return min_sl,position

	# wait until later

	# fluxbins11 = select_flux_bins11(tdata_original)
	# tdata = fluxbins11
	# filename = filename_original + 'flux11'

def value_added_selection_NN_2D_size(equal_width=False):	

	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	redshift = False

	filename = 'value_added_selection_NN'# EDIT THIS
	print ('\n!! Filename:', filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	if position_angle: filename += '_PA' # only when using 'position_angle'
	try:
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
	except KeyError:
		pass # then we're doing biggest_selection
	tdata_original = tdata
	filename_original = filename

	title = 'Isolated lobes subset: Size bins\n\n'# EDIT THIS
	if equal_width:
		sizebins = select_size_bins1(tdata_original)
	else:
		sizebins = select_size_bins_cuts_biggest_selection(tdata_original)

	fig, ax = plt.subplots()
	
	for key in sorted(sizebins):
		tdata = sizebins[key]
		filename = filename_original + 'size%s'%key # important !! (for loading data)

		if len(tdata) != 0:
			all_sl, angular_radius = statistics(filename,tdata,redshift)

			# have to give angular_radius = None for multiple plots in same figure
			# since different subsets have different angular radii per number of nn
			if key == '3':
				SL_vs_n_plot(filename,all_sl,angular_radius,n,ax,label='Bin %s'%key)
				min_sl = np.min(all_sl)
				position = angular_radius[np.argmin(all_sl)]
				print ('Minimum SL: ', min_sl)
				print ('At angular radius: %f degrees'%position)
				print '\n'
			else:
				# hard fix for when n > amount of sources in a bin 
				ending_n = n
				if ending_n > len(tdata):
					print ('n = %i, but this tdata only contains N=%i sources'%(n,len(tdata)))
					ending_n = len(tdata)-1
					print ('Setting n=%i'%ending_n)
				ax.plot(range(1,ending_n+1),np.log10(all_sl),ls='solid',label='Bin %s'%key)
		else:
			min_sl, position = np.nan, np.nan

	ax.set_xlim(2,n)
	ax.set_ylim(ylim_lower,0)
	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	plt.title(title,fontsize=16)
	plt.tight_layout()
	# plt.show()
	# EDIT THIS
	plt.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_isolated_lobes_subset_sizebins.pdf')
	return min_sl,position

def value_added_compmatch_2D():
	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	#Running all the statistics without redshift
	redshift = False

	filename = 'value_added_compmatch'  # EDIT THIS
	print ('\n!! Filename:', filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	if position_angle: filename += '_PA' # only when using 'position_angle'
	try:
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
	except KeyError:
		pass # then we're doing biggest_selection
	tdata_original = tdata
	filename_original = filename

	title = 'Concluding subset: 2D analysis\n\n' # EDIT THIS
	all_sl, angular_radius = statistics(filename,tdata,redshift)
	
	fig, ax = plt.subplots()
	# have to give angular_radius = None for multiple plots in same figure
	SL_vs_n_plot(filename,all_sl,angular_radius,n,ax)

	ax.set_xlim(2,n)
	ax.set_ylim(ylim_lower,0)
	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	plt.title(title,fontsize=16)
	plt.tight_layout()
	# plt.show()
	plt.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_concluding_subset.pdf')

def value_added_compmatch_2D_flux(equal_width=False):

	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	redshift = False

	filename = 'value_added_compmatch'# EDIT THIS
	print ('\n!! Filename:', filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	if position_angle: filename += '_PA' # only when using 'position_angle'
	try:
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
	except KeyError:
		pass # then we're doing biggest_selection
	tdata_original = tdata
	filename_original = filename

	title = 'Concluding subset: Flux bins\n\n'# EDIT THIS
	if equal_width:
		fluxbins = select_flux_bins1(tdata_original)
	else:
		fluxbins = select_flux_bins_cuts_biggest_selection(tdata_original)
	fig, ax = plt.subplots()
	
	for key in sorted(fluxbins):
		tdata = fluxbins[key]
		filename = filename_original + 'flux%s'%key
		all_sl, angular_radius = statistics(filename,tdata,redshift)
		# have to give angular_radius = None for multiple plots in same figure
		# since different subsets have different angular radii per number of nn
		if key == '3':
			SL_vs_n_plot(filename,all_sl,angular_radius,n,ax,label='Bin %s'%key)
			min_sl = np.min(all_sl)
			position = angular_radius[np.argmin(all_sl)]
			print ('Minimum SL: ', min_sl)
			print ('At angular radius: %f degrees'%position)
			print '\n'
		else:
			# hard fix for when n > amount of sources in a bin 
			ending_n = n
			if ending_n > len(tdata):
				print ('n = %i, but this tdata only contains N=%i sources'%(n,len(tdata)))
				ending_n = len(tdata)-1
				print ('Setting n=%i'%ending_n)
			ax.plot(range(1,ending_n+1),np.log10(all_sl),ls='solid',label='Bin %s'%key)
		
		

	ax.set_xlim(2,n)
	ax.set_ylim(ylim_lower,0)
	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	plt.title(title,fontsize=16)
	plt.tight_layout()
	# plt.show()
	plt.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_concluding_subset_fluxbins.pdf')
	return min_sl,position

	# wait until later

	# fluxbins11 = select_flux_bins11(tdata_original)
	# tdata = fluxbins11
	# filename = filename_original + 'flux11'

def value_added_compmatch_2D_size(equal_width=False):

	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	redshift = False

	filename = 'value_added_compmatch'# EDIT THIS
	print ('\n!! Filename:', filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	if position_angle: filename += '_PA' # only when using 'position_angle'
	try:
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
	except KeyError:
		pass # then we're doing biggest_selection
	tdata_original = tdata
	filename_original = filename

	title = 'Concluding subset: Size bins\n\n'# EDIT THIS
	if equal_width:
		sizebins = select_size_bins1(tdata_original)
	else:
		sizebins = select_size_bins_cuts_biggest_selection(tdata_original)

	fig, ax = plt.subplots()
	
	for key in sorted(sizebins):
		tdata = sizebins[key]
		filename = filename_original + 'size%s'%key # important !! (for loading data)

		all_sl, angular_radius = statistics(filename,tdata,redshift)
		# have to give angular_radius = None for multiple plots in same figure
		# since different subsets have different angular radii per number of nn
		if key == '3':
			SL_vs_n_plot(filename,all_sl,angular_radius,n,ax,label='Bin %s'%key)
			min_sl = np.min(all_sl)
			position = angular_radius[np.argmin(all_sl)]
			print ('Minimum SL: ', min_sl)
			print ('At angular radius: %f degrees'%position)
			print '\n'
		else:
			# hard fix for when n > amount of sources in a bin 
			ending_n = n
			if ending_n > len(tdata):
				print ('n = %i, but this tdata only contains N=%i sources'%(n,len(tdata)))
				ending_n = len(tdata)-1
				print ('Setting n=%i'%ending_n)
			ax.plot(range(1,ending_n+1),np.log10(all_sl),ls='solid',label='Bin %s'%key)
		

	ax.set_xlim(2,n)
	ax.set_ylim(ylim_lower,0)
	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	plt.title(title,fontsize=16)
	plt.tight_layout()
	plt.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_concluding_subset_sizebins.pdf')
	return min_sl,position

def all_subsets_unbinned():
	'''plot 1 figure that has unbinned SL for all subsets  '''
	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	#Running all the statistics without redshift
	redshift = False

	def helper_plot(filename,label):
		print ('\n!! Filename:', filename)
		tdata = Table(fits.open('../%s.fits'%filename)[1].data)
		if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
		if position_angle: filename += '_PA' # only when using 'position_angle'
		try:
			tdata['RA'] = tdata['RA_2']
			tdata['DEC'] = tdata['DEC_2']
		except KeyError:
			pass # then we're doing biggest_selection
		tdata_original = tdata
		filename_original = filename

		title = 'Initial subsample: 2D analysis\n\n'# EDIT THIS
		all_sl, angular_radius = statistics(filename,tdata,redshift)

		min_sl = np.log10(np.min(all_sl))
		position = angular_radius[np.argmin(all_sl)]
		print ('Minimum log10 SL: ', min_sl)
		print ('At angular radius: %f degrees'%position)
		print ('At n_n %i'%(np.argmin(all_sl)+1))
		
		# have to give angular_radius = None for multiple plots in same figure
		ending_n = n
		if ending_n > len(tdata):
			print ('n = %i, but this tdata only contains N=%i sources'%(n,len(tdata)))
			ending_n = len(tdata)-1
			print ('Setting n=%i'%ending_n)
		ax.plot(range(1,ending_n+1),np.log10(all_sl),ls='solid',label='%s'%label)

		# angular radius plot
		ax2.plot(range(2,ending_n+1),angular_radius[2-1:],ls='solid',label='%s'%label) # index 1 corresponds to n=2
		
	fig1, ax = plt.subplots()
	fig2, ax2 = plt.subplots()

	filename = 'biggest_selection'# EDIT THIS
	helper_plot(filename,'Initial sample')
	filename = 'value_added_selection' # EDIT THIS
	helper_plot(filename,'Value-added subset')
	filename = 'value_added_selection_MG'  # EDIT THIS
	helper_plot(filename,'Connected lobes')
	filename = 'value_added_selection_NN'  # EDIT THIS
	helper_plot(filename,'Isolated lobes')
	filename = 'value_added_compmatch'  # EDIT THIS
	helper_plot(filename,'Concluding subset')

	ax.set_xlim(2,n)
	ax.set_ylim(ylim_lower,0)
	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=12)
	title = ''
	ax.set_title(title,fontsize=16)
	fig1.tight_layout()
	# plt.show()
	fig1.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_all_subsets.pdf')
	plt.close(fig1)

	ax2.set_ylabel('Median angular radius (degrees)',fontsize=14)
	ax2.set_xlabel(r'$n_n$',fontsize=14)
	ax2.tick_params(labelsize=12)
	title = ''
	ax2.set_title(title,fontsize=16)
	ax2.legend(fontsize=12)
	fig2.tight_layout()
	fig2.savefig('/data1/osinga/figures/thesis/results/angular_radius_2D_all_subsets.pdf')
	plt.close(fig2)



# four other subsets highest flux bin
def other_subsets_flux(equal_width=False):
	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	redshift = False
	title = 'Subsets: Highest flux bin\n\n'# EDIT THIS

	def helper_plot(filename,label):
		print ('\n!! Filename:', filename)
		tdata = Table(fits.open('../%s.fits'%filename)[1].data)
		if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
		if position_angle: filename += '_PA' # only when using 'position_angle'
		try:
			tdata['RA'] = tdata['RA_2']
			tdata['DEC'] = tdata['DEC_2']
		except KeyError:
			pass # then we're doing biggest_selection
		tdata_original = tdata
		filename_original = filename

		if equal_width:
			fluxbins = select_flux_bins1(tdata_original)
		else:
			fluxbins = select_flux_bins_cuts_biggest_selection(tdata_original)

		for key in sorted(fluxbins):
			if key == '3':
				tdata = fluxbins[key]
				filename = filename_original + 'flux%s'%key
				print ('Number of sources in highest flux bin:')
				print ('%s : %i'%(filename,len(tdata)))
				all_sl, angular_radius = statistics(filename,tdata,redshift)
				# have to give angular_radius = None for multiple plots in same figure
				# since different subsets have different angular radii per number of nn
				ending_n = n
				if ending_n > len(tdata):
					print ('n = %i, but this tdata only contains N=%i sources'%(n,len(tdata)))
					ending_n = len(tdata)-1
					print ('Setting n=%i'%ending_n)

				min_sl = np.min(all_sl)
				position = angular_radius[np.argmin(all_sl)]
				print ('Minimum SL: ', min_sl)
				print ('At angular radius: %f degrees'%position)
				print '\n'
		
				ax.plot(range(1,ending_n+1),np.log10(all_sl),ls='solid',label='%s'%label)
				ax2.plot(range(2,ending_n+1),angular_radius[2-1:],ls='solid',label='%s'%label) # index 1 corresponds to n=2

	fig1, ax = plt.subplots()
	fig2, ax2 = plt.subplots()
	
	filename = 'value_added_selection' # EDIT THIS
	helper_plot(filename,'Value-added subset')
	filename = 'value_added_selection_MG'  # EDIT THIS
	# helper_plot(filename,'Connected lobes')
	# filename = 'value_added_selection_NN'  # EDIT THIS
	helper_plot(filename,'Isolated lobes')
	filename = 'value_added_compmatch'  # EDIT THIS
	helper_plot(filename,'Concluding subset')

	ax.set_xlim(2,n)
	ax.set_ylim(ylim_lower,0)
	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	title = ''
	ax.set_title(title,fontsize=16)
	fig1.tight_layout()
	# plt.show()
	fig1.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_other_subsets_flux3.pdf')
	plt.close(fig1)
	
	ax2.set_ylabel('Median angular radius (degrees)',fontsize=14)
	ax2.set_xlabel(r'$n_n$',fontsize=14)
	ax2.tick_params(labelsize=12)
	title = ''
	ax2.set_title(title,fontsize=16)
	ax2.legend(fontsize=12)
	fig2.tight_layout()
	fig2.savefig('/data1/osinga/figures/thesis/results/angular_radius_2D_other_subsets_flux3.pdf')
	plt.close(fig2)

	# return min_sl,position

# four other subsets highest size bin
def other_subsets_size3(equal_width=False):
	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	redshift = False
	title = 'Subsets: Highest size bin\n\n'# EDIT THIS

	def helper_plot(filename,label):
		print ('\n!! Filename:', filename)
		tdata = Table(fits.open('../%s.fits'%filename)[1].data)
		if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
		if position_angle: filename += '_PA' # only when using 'position_angle'
		try:
			tdata['RA'] = tdata['RA_2']
			tdata['DEC'] = tdata['DEC_2']
		except KeyError:
			pass # then we're doing biggest_selection
		tdata_original = tdata
		filename_original = filename

		if equal_width:
			sizebins = select_size_bins1(tdata_original)
		else:
			sizebins = select_size_bins_cuts_biggest_selection(tdata_original)

		for key in sorted(sizebins):
			if key == '3':
				tdata = sizebins[key]
				filename = filename_original + 'size%s'%key
				print ('Number of sources in highest size bin:')
				print ('%s : %i'%(filename,len(tdata)))
				all_sl, angular_radius = statistics(filename,tdata,redshift)
				# have to give angular_radius = None for multiple plots in same figure
				# since different subsets have different angular radii per number of nn
				ending_n = n
				if ending_n > len(tdata):
					print ('n = %i, but this tdata only contains N=%i sources'%(n,len(tdata)))
					ending_n = len(tdata)-1
					print ('Setting n=%i'%ending_n)

				min_sl = np.min(all_sl)
				position = angular_radius[np.argmin(all_sl)]
				print ('Minimum SL: ', min_sl)
				print ('At angular radius: %f degrees'%position)
				print '\n'
		
				ax.plot(range(1,ending_n+1),np.log10(all_sl),ls='solid',label='%s'%label)
				ax2.plot(range(2,ending_n+1),angular_radius[2-1:],ls='solid',label='%s'%label) # index 1 corresponds to n=2

	plt.rcParams['axes.prop_cycle'] = plt.rcParamsDefault['axes.prop_cycle'][1:]
	fig1, ax = plt.subplots()
	fig2, ax2 = plt.subplots()
	
	filename = 'value_added_selection' # EDIT THIS
	helper_plot(filename,'Value-added subset')
	filename = 'value_added_selection_MG'  # EDIT THIS
	# helper_plot(filename,'Connected lobes')
	# filename = 'value_added_selection_NN'  # EDIT THIS
	helper_plot(filename,'Isolated lobes')
	filename = 'value_added_compmatch'  # EDIT THIS
	helper_plot(filename,'Concluding subset')

	ax.set_xlim(2,n)
	ax.set_ylim(ylim_lower,0)
	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	title = ''
	ax.set_title(title,fontsize=16)
	fig1.tight_layout()
	# plt.show()
	fig1.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_other_subsets_size3.pdf')
	plt.close(fig1)
	
	ax2.set_ylabel('Median angular radius (degrees)',fontsize=14)
	ax2.set_xlabel(r'$n_n$',fontsize=14)
	ax2.tick_params(labelsize=12)
	title = ''
	ax2.set_title(title,fontsize=16)
	ax2.legend(fontsize=12)
	fig2.tight_layout()
	fig2.savefig('/data1/osinga/figures/thesis/results/angular_radius_2D_other_subsets_size3.pdf')
	plt.close(fig2)

# four other subsets lowest size bin
def other_subsets_size0(equal_width=False):
	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	redshift = False
	title = 'Subsets: Lowest size bin\n\n'# EDIT THIS

	def helper_plot(filename,label):
		print ('\n!! Filename:', filename)
		tdata = Table(fits.open('../%s.fits'%filename)[1].data)
		if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
		if position_angle: filename += '_PA' # only when using 'position_angle'
		try:
			tdata['RA'] = tdata['RA_2']
			tdata['DEC'] = tdata['DEC_2']
		except KeyError:
			pass # then we're doing biggest_selection
		tdata_original = tdata
		filename_original = filename

		if equal_width:
			sizebins = select_size_bins1(tdata_original)
		else:
			sizebins = select_size_bins_cuts_biggest_selection(tdata_original)

		for key in sorted(sizebins):
			if key == '0':
				tdata = sizebins[key]
				filename = filename_original + 'size%s'%key
				print ('Number of sources in lowest size bin:')
				print ('%s : %i'%(filename,len(tdata)))
				all_sl, angular_radius = statistics(filename,tdata,redshift)
				# have to give angular_radius = None for multiple plots in same figure
				# since different subsets have different angular radii per number of nn
				ending_n = n
				if ending_n > len(tdata):
					print ('n = %i, but this tdata only contains N=%i sources'%(n,len(tdata)))
					ending_n = len(tdata)-1
					print ('Setting n=%i'%ending_n)

				min_sl = np.min(all_sl)
				position = angular_radius[np.argmin(all_sl)]
				print ('Minimum SL: ', min_sl)
				print ('At angular radius: %f degrees'%position)
				print '\n'
		
				ax.plot(range(1,ending_n+1),np.log10(all_sl),ls='solid',label='%s'%label)
				ax2.plot(range(2,ending_n+1),angular_radius[2-1:],ls='solid',label='%s'%label) # index 1 corresponds to n=2

	fig1, ax = plt.subplots()
	fig2, ax2 = plt.subplots()
	
	# plt.rcParams['axes.prop_cycle'] = plt.rcParams['axes.prop_cycle'][1:]

	filename = 'biggest_selection'
	helper_plot(filename,'Initial sample')

	filename = 'value_added_selection' # EDIT THIS
	helper_plot(filename,'Value-added subset')
	filename = 'value_added_selection_MG'  # EDIT THIS
	helper_plot(filename,'Connected lobes')
	# filename = 'value_added_selection_NN'  # EDIT THIS
	# helper_plot(filename,'Isolated lobes')
	filename = 'value_added_compmatch'  # EDIT THIS
	helper_plot(filename,'Concluding subset')

	ax.set_xlim(2,n)
	ax.set_ylim(ylim_lower,0)
	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	title = ''
	ax.set_title(title,fontsize=16)
	fig1.tight_layout()
	fig1.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_other_subsets_size0.pdf')
	plt.show()
	plt.close(fig1)
	
	ax2.set_ylabel('Median angular radius (degrees)',fontsize=14)
	ax2.set_xlabel(r'$n_n$',fontsize=14)
	ax2.tick_params(labelsize=12)
	title = ''
	ax2.set_title(title,fontsize=16)
	ax2.legend(fontsize=12)
	fig2.tight_layout()
	fig2.savefig('/data1/osinga/figures/thesis/results/angular_radius_2D_other_subsets_size0.pdf')
	plt.close(fig2)

#plot as function of agular dist
def other_subsets_flux_angulardist(equal_width=False):
	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	redshift = False
	title = 'Subsets: Highest flux bin\n\n'# EDIT THIS

	def helper_plot(filename,label):
		print ('\n!! Filename:', filename)
		tdata = Table(fits.open('../%s.fits'%filename)[1].data)
		if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
		if position_angle: filename += '_PA' # only when using 'position_angle'
		try:
			tdata['RA'] = tdata['RA_2']
			tdata['DEC'] = tdata['DEC_2']
		except KeyError:
			pass # then we're doing biggest_selection
		tdata_original = tdata
		filename_original = filename

		if equal_width:
			fluxbins = select_flux_bins1(tdata_original)
		else:
			fluxbins = select_flux_bins_cuts_biggest_selection(tdata_original)

		for key in sorted(fluxbins):
			if key == '3':
				tdata = fluxbins[key]
				filename = filename_original + 'flux%s'%key
				print ('Number of sources in highest flux bin:')
				print ('%s : %i'%(filename,len(tdata)))
				all_sl, angular_radius = statistics(filename,tdata,redshift)
				# have to give angular_radius = None for multiple plots in same figure
				# since different subsets have different angular radii per number of nn
				ending_n = n
				if ending_n > len(tdata):
					print ('n = %i, but this tdata only contains N=%i sources'%(n,len(tdata)))
					ending_n = len(tdata)-1
					print ('Setting n=%i'%ending_n)

				min_sl = np.min(all_sl)
				position = angular_radius[np.argmin(all_sl)]
				print ('Minimum SL: ', min_sl)
				print ('At angular radius: %f degrees'%position)
				print '\n'
				print ('max angular radius:',np.max(angular_radius))
			
				# angular radius index 0 corresponds to s_1, no slicing needed
				ax.plot(angular_radius,np.log10(all_sl),ls='solid',label='%s'%label)
				ax2.plot(range(2,ending_n+1),angular_radius[2-1:],ls='solid',label='%s'%label) # index 1 corresponds to n=2

	plt.rcParams['axes.prop_cycle'] = plt.rcParamsDefault['axes.prop_cycle'][1:]
	fig1, ax = plt.subplots()
	fig2, ax2 = plt.subplots()
	
	# filename = 'biggest_selection'
	# helper_plot(filename,'Initial sample')

	filename = 'value_added_selection' # EDIT THIS
	helper_plot(filename,'Value-added subset')
	filename = 'value_added_selection_MG'  # EDIT THIS
	helper_plot(filename,'Connected lobes')
	filename = 'value_added_selection_NN'  # EDIT THIS
	helper_plot(filename,'Isolated lobes')
	filename = 'value_added_compmatch'  # EDIT THIS
	helper_plot(filename,'Concluding subset')

	# ax.set_xlim(2,n)
	ax.set_ylim(-4.5,0)
	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'Angular radius (degrees)',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	title = ''
	ax.set_title(title,fontsize=16)
	fig1.tight_layout()
	fig1.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_other_subsets_flux3_angradius.pdf')
	# plt.show()
	plt.close(fig1)
	
	ax2.set_ylabel('Median angular radius (degrees)',fontsize=14)
	ax2.set_xlabel(r'$n_n$',fontsize=14)
	ax2.tick_params(labelsize=12)
	title = ''
	ax2.set_title(title,fontsize=16)
	ax2.legend(fontsize=12)
	fig2.tight_layout()
	# fig2.savefig('/data1/osinga/figures/thesis/results/angular_radius_2D_other_subsets_flux3.pdf')
	plt.close(fig2)

	# return min_sl,position

#plot as function of agular dist
def other_subsets_size3_angulardist(equal_width=False):
	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	redshift = False
	title = 'Subsets: Highest size bin\n\n'# EDIT THIS

	def helper_plot(filename,label):
		print ('\n!! Filename:', filename)
		tdata = Table(fits.open('../%s.fits'%filename)[1].data)
		if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
		if position_angle: filename += '_PA' # only when using 'position_angle'
		try:
			tdata['RA'] = tdata['RA_2']
			tdata['DEC'] = tdata['DEC_2']
		except KeyError:
			pass # then we're doing biggest_selection
		tdata_original = tdata
		filename_original = filename

		if equal_width:
			sizebins = select_size_bins1(tdata_original)
		else:
			sizebins = select_size_bins_cuts_biggest_selection(tdata_original)

		for key in sorted(sizebins):
			if key == '3':
				tdata = sizebins[key]
				filename = filename_original + 'size%s'%key
				print ('Number of sources in highest size bin:')
				print ('%s : %i'%(filename,len(tdata)))
				all_sl, angular_radius = statistics(filename,tdata,redshift)
				# have to give angular_radius = None for multiple plots in same figure
				# since different subsets have different angular radii per number of nn
				ending_n = n
				if ending_n > len(tdata):
					print ('n = %i, but this tdata only contains N=%i sources'%(n,len(tdata)))
					ending_n = len(tdata)-1
					print ('Setting n=%i'%ending_n)

				min_sl = np.min(all_sl)
				position = angular_radius[np.argmin(all_sl)]
				print ('Minimum SL: ', min_sl)
				print ('At angular radius: %f degrees'%position)
				print '\n'
		
				# angular radius index 0 corresponds to s_1, no slicing needed
				ax.plot(angular_radius,np.log10(all_sl),ls='solid',label='%s'%label)
				ax2.plot(range(2,ending_n+1),angular_radius[2-1:],ls='solid',label='%s'%label) # index 1 corresponds to n=2

	# to skip the first color and skip the 3rd color
	plt.rcParams['axes.prop_cycle'] = plt.rcParamsDefault['axes.prop_cycle'][1:3].concat(plt.rcParamsDefault['axes.prop_cycle'][4:])

	fig1, ax = plt.subplots()
	fig2, ax2 = plt.subplots()
	
	# filename = 'biggest_selection'
	# helper_plot(filename,'Initial sample')

	filename = 'value_added_selection' # EDIT THIS
	helper_plot(filename,'Value-added subset')
	filename = 'value_added_selection_MG'  # EDIT THIS
	helper_plot(filename,'Connected lobes')
	# THERE IS NO SIZE3 DATA
	# filename = 'value_added_selection_NN'  # EDIT THIS
	# helper_plot(filename,'Isolated lobes')
	filename = 'value_added_compmatch'  # EDIT THIS
	helper_plot(filename,'Concluding subset')

	# ax.set_xlim(2,n)
	ax.set_ylim(ylim_lower,0)
	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'Angular radius (degrees)',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	title = ''
	ax.set_title(title,fontsize=16)
	fig1.tight_layout()
	fig1.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_other_subsets_size3_angradius.pdf')
	# plt.show()
	plt.close(fig1)
	
	ax2.set_ylabel('Median angular radius (degrees)',fontsize=14)
	ax2.set_xlabel(r'$n_n$',fontsize=14)
	ax2.tick_params(labelsize=12)
	title = ''
	ax2.set_title(title,fontsize=16)
	ax2.legend(fontsize=12)
	fig2.tight_layout()
	# fig2.savefig('/data1/osinga/figures/thesis/results/angular_radius_2D_other_subsets_size3.pdf')
	plt.close(fig2)

#plot as function of agular dist
def other_subsets_size0_angulardist(equal_width=False):
	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	redshift = False
	title = 'Subsets: Lowest size bin\n\n'# EDIT THIS

	def helper_plot(filename,label):

		print ('\n!! Filename:', filename)
		tdata = Table(fits.open('../%s.fits'%filename)[1].data)
		if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
		if position_angle: filename += '_PA' # only when using 'position_angle'
		try:
			tdata['RA'] = tdata['RA_2']
			tdata['DEC'] = tdata['DEC_2']
		except KeyError:
			pass # then we're doing biggest_selection
		tdata_original = tdata
		filename_original = filename

		if equal_width:
			sizebins = select_size_bins1(tdata_original)
		else:
			sizebins = select_size_bins_cuts_biggest_selection(tdata_original)

		for key in sorted(sizebins):
			if key == '0':
				tdata = sizebins[key]
				filename = filename_original + 'size%s'%key
				print ('Number of sources in lowest size bin:')
				print ('%s : %i'%(filename,len(tdata)))
				all_sl, angular_radius = statistics(filename,tdata,redshift)
				# have to give angular_radius = None for multiple plots in same figure
				# since different subsets have different angular radii per number of nn
				ending_n = n
				if ending_n > len(tdata):
					print ('n = %i, but this tdata only contains N=%i sources'%(n,len(tdata)))
					ending_n = len(tdata)-1
					print ('Setting n=%i'%ending_n)

				min_sl = np.min(all_sl)
				position = angular_radius[np.argmin(all_sl)]
				print ('Minimum SL: ', min_sl)
				print ('At angular radius: %f degrees'%position)
				print '\n'
				
				# angular radius index 0 corresponds to s_1, no slicing needed
				ax.plot(angular_radius,np.log10(all_sl),ls='solid',label='%s'%label)
				ax2.plot(range(2,ending_n+1),angular_radius[2-1:],ls='solid',label='%s'%label) # index 1 corresponds to n=2

	plt.rcParams['axes.prop_cycle'] = plt.rcParamsDefault['axes.prop_cycle'][1:]
	
	fig1, ax = plt.subplots()
	fig2, ax2 = plt.subplots()

	# filename = 'biggest_selection'
	# helper_plot(filename,'Initial sample')

	filename = 'value_added_selection' # EDIT THIS
	helper_plot(filename,'Value-added subset')
	filename = 'value_added_selection_MG'  # EDIT THIS
	helper_plot(filename,'Connected lobes')
	filename = 'value_added_selection_NN'  # EDIT THIS
	helper_plot(filename,'Isolated lobes')
	filename = 'value_added_compmatch'  # EDIT THIS
	helper_plot(filename,'Concluding subset')

	# ax.set_xlim(2,n)
	ax.set_ylim(ylim_lower,0)
	ax.tick_params(labelsize=12)


	ax.set_xlabel(r'Angular radius (degrees)',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14,loc='lower right')
	title = ''
	ax.set_title(title,fontsize=16)
	fig1.tight_layout()
	fig1.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_other_subsets_size0_angradius.pdf')
	plt.show()
	plt.close(fig1)
	
	ax2.set_ylabel('Median angular radius (degrees)',fontsize=14)
	ax2.set_xlabel(r'$n_n$',fontsize=14)
	ax2.tick_params(labelsize=12)
	title = ''
	ax2.set_title(title,fontsize=16)
	ax2.legend(fontsize=12)
	fig2.tight_layout()
	fig2.savefig('/data1/osinga/figures/thesis/results/angular_radius_2D_other_subsets_size0.pdf')
	plt.close(fig2)



# todo: rewrite this
def last_2_subsets_flux_and_size(equal_width=False):
	assert sys.argv[2] == 'True', ' set sys.argv[2] = True by command line ' 

	redshift = False


	################### value_added_selection_MG
	filename = 'value_added_selection_MG'# EDIT THIS
	print ('\n!! Filename:', filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	if position_angle: filename += '_PA' # only when using 'position_angle'
	try:
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
	except KeyError:
		pass # then we're doing biggest_selection
	tdata_original = tdata
	filename_original = filename

	title = 'Connected and matched lobes: Highest flux bin\n\n'# EDIT THIS
	if equal_width:
		fluxbins = select_flux_bins1(tdata_original)
	else:
		fluxbins = select_flux_bins_cuts_biggest_selection(tdata_original)

	fig, ax = plt.subplots()
	
	for key in sorted(fluxbins):
		tdata = fluxbins[key]
		filename = filename_original + 'flux%s'%key
		all_sl, angular_radius = statistics(filename,tdata,redshift)
		# have to give angular_radius = None for multiple plots in same figure
		# since different subsets have different angular radii per number of nn
		if key == '3':
			ax.plot(angular_radius,np.log10(all_sl),ls='solid',label='Connected lobes')
			min_sl = np.min(all_sl)
			position = angular_radius[np.argmin(all_sl)]
			print ('Minimum SL: ', min_sl)
			print ('At angular radius: %f degrees'%position)
			print '\n'
		else:
			# ax.plot(range(1,n+1),np.log10(all_sl),ls='solid',label='Bin %s'%key)
			pass


	################### 

	################### value_added_selection_compmatch
	filename = 'value_added_compmatch'# EDIT THIS
	print ('\n!! Filename:', filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	if position_angle: filename += '_PA' # only when using 'position_angle'
	try:
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
	except KeyError:
		pass # then we're doing biggest_selection
	tdata_original = tdata
	filename_original = filename

	if equal_width:
		fluxbins = select_flux_bins1(tdata_original)
	else:
		fluxbins = select_flux_bins_cuts_biggest_selection(tdata_original)
	
	for key in sorted(fluxbins):
		tdata = fluxbins[key]
		filename = filename_original + 'flux%s'%key
		all_sl, angular_radius = statistics(filename,tdata,redshift)
		# have to give angular_radius = None for multiple plots in same figure
		# since different subsets have different angular radii per number of nn
		if key == '3':
			ax.plot(angular_radius,np.log10(all_sl),ls='solid',label='Matched lobes')
			min_sl = np.min(all_sl)
			position = angular_radius[np.argmin(all_sl)]
			print ('Minimum SL: ', min_sl)
			print ('At angular radius: %f degrees'%position)
			print '\n'
		else:
			pass		

	ax.set_xlim(0,np.max(angular_radius))
	ax.set_ylim(ylim_lower,0)
	ax.tick_params(labelsize=12)


	ax.set_xlabel('Angular radius (degrees)',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	plt.title(title,fontsize=16)
	plt.tight_layout()
	# plt.show()
	plt.savefig('/data1/osinga/figures/thesis/results/SL_vs_n_2D_other_subsets_flux_size.pdf')
	return min_sl,position




if __name__ == '__main__':
	ylim_lower = -3.82

	# F = open('./min_sl_position_2Dsubsets.csv','w')
	# F.write('subset,log10_SL_flux,pos_flux,log10_SL_size,pos_size')
	# F.write('\n')

	# biggest_selection_2D()
	# min_slf,positionf = biggest_selection_2D_flux()
	# min_sls,positions = biggest_selection_2D_size()
	# F.write('biggest_selection,%.3f,%.3f,%.3f,%.3f'%(np.log10(min_slf),positionf,np.log10(min_sls),positions))
	# F.write('\n')

	# value_added_selection_2D()
	# min_slf,positionf = value_added_selection_2D_flux()
	# min_sls,positions = value_added_selection_2D_size()
	# F.write('value_added_selection,%.3f,%.3f,%.3f,%.3f'%(np.log10(min_slf),positionf,np.log10(min_sls),positions))
	# F.write('\n')

	# value_added_selection_MG_2D()
	# min_slf,positionf = value_added_selection_MG_2D_flux()
	# min_sls,positions = value_added_selection_MG_2D_size()
	# F.write('value_added_selection_MG,%.3f,%.3f,%.3f,%.3f'%(np.log10(min_slf),positionf,np.log10(min_sls),positions))
	# F.write('\n')

	# value_added_selection_NN_2D()
	# min_slf,positionf = value_added_selection_NN_2D_flux()
	# min_slf,positionf = value_added_selection_NN_2D_size()
	# F.write('value_added_selection_NN,%.3f,%.3f,%.3f,%.3f'%(np.log10(min_slf),positionf,np.log10(min_sls),positions))
	# F.write('\n')

	# value_added_compmatch_2D()
	# min_slf,positionf = value_added_compmatch_2D_flux()
	# min_sls,positions = value_added_compmatch_2D_size()
	# F.write('value_added_compmatch,%.3f,%.3f,%.3f,%.3f'%(np.log10(min_slf),positionf,np.log10(min_sls),positions))
	# F.write('\n')

	# F.close()

	# all_subsets_unbinned()

	# other_subsets_size0()
	# other_subsets_size3()
	# other_subsets_flux()

	# other_subsets_size0_angulardist()
	other_subsets_size3_angulardist()
	other_subsets_flux_angulardist()

	# last_2_subsets_size()
	# last_2_subsets_flux_and_size()


	'''
	If there are dimension errors try recalculate=True
	'''
