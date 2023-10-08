'''
The original code from https://github.com/ErikOsinga/MRP1,
developed by Erik Osinga.
'''
import sys
#sys.path.insert(0,'/net/reusel/data1/osinga/anaconda2')

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
		, select_flux_bins11, select_power_bins, select_physical_size_bins,
		select_flux_bins_cuts_biggest_selection, select_size_bins_cuts_biggest_selection)

from astropy.io import ascii

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import pandas as pd

parallel_transport = True
print ('Using parallel_transport = %s' %parallel_transport)

position_angle = 'False' #sys.argv[2]
if position_angle == 'True':
	position_angle = True
else:
	position_angle = False
if position_angle:
	print ('Using position_angle')
else: 
	print ('Using RPA')

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

def Sn_vs_n_test(tdata,Sn_mc,Sn_data,filename,angular_radius,ending_n=180):
	"""
	Make a plot of the SL (Significance level) statistic vs n.
	"""

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
		print ('no writing')
		# Results.write('./data/2D/Statistics_PT'+filename+'_results.fits',overwrite=True)
	else:
		print ('no writing')
		# Results.write('./data/2D/Statistics_'+filename+'_results.fits',overwrite=True)
	
	fig, axarr = plt.subplots(2, sharex=True, gridspec_kw= {'height_ratios':[3, 1]})
	axarr[0].plot(range(starting_n,ending_n+1),np.log10(all_sl))
	# print np.argmin(all_sl)
	axarr[0].set_title('SL vs n for '+filename+'\n\n')
	axarr[0].set_ylabel(r'$\log_{10}$ SL')
	axarr[0].set_xlim(2,n)
	
	if angular_radius is not None:
		ax2 = axarr[0].twiny()
		ax2.set_xlabel('angular_radius (degrees)')
		xticks = axarr[0].get_xticks()
		ax2.set_xticks(xticks)
		print (np.asarray(xticks,dtype='int'))
		xticks2 = np.append(0,angular_radius)[np.asarray(xticks,dtype='int')]
		ax2.set_xticklabels(tick_function(xticks2))
	plt.subplots_adjust(top=0.850)	

	axarr[1].plot(range(starting_n,ending_n+1),all_std)
	axarr[1].set_xlabel('n')
	axarr[1].set_ylabel('sigma')
	axarr[0].set_ylim(-2.5,0)

	if parallel_transport:
		print ('PT')
		# plt.savefig('./figures/2D/SL_vs_n_PT'+filename+'.png',overwrite=True)
	else:
		print ('no PT')
		# plt.savefig('./figures/2D/SL_vs_n_'+filename+'.png',overwrite=True)
	plt.show()
	plt.close()

def Sn_vs_n(tdata,Sn_mc,Sn_data,filename,angular_radius,ending_n=180):
	"""
	Make a plot of the SL (Significance level) statistic vs n.
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
	
	try:
		#fig, axarr = plt.subplots(2, sharex=True, gridspec_kw= {'height_ratios':[3, 1]})
                plt.figure()
                ax = plt.gca()
                ax.plot(range(starting_n,ending_n+1),np.log10(all_sl))
		# print np.argmin(all_sl)
		#ax.set_title('SL vs n for '+filename+'\n\n')
                ax.set_ylabel(r'log(SL)', fontsize=16)
                ax.set_xlim(0,n) #2
                ax.set_ylim(-6.0,0)

                if angular_radius is not None:
                   ax2 = ax.twiny()
                   ax2.set_xlabel('Angular scale (deg)', fontsize=16)
                   xticks = ax.get_xticks()
                   ax2.set_xticks(xticks)
                   print (np.asarray(xticks,dtype='int'))
                   xticks2 = np.append(0,angular_radius)[np.asarray(xticks,dtype='int')]
                   ax2.set_xticklabels(tick_function(xticks2))
                   ax2.tick_params(labelsize=12)
                   ax2.xaxis.set_minor_locator(AutoMinorLocator())
                   ax2.yaxis.set_minor_locator(AutoMinorLocator())
                   ax2.tick_params(which='both', width=2)
                   ax2.tick_params(which='major', length=7)
                   ax2.tick_params(which='minor', length=4, color='k')

                   ax2.tick_params(axis="x", direction="in")
                   ax2.tick_params(axis="y", direction="in")
                   ax2.tick_params(which="minor", axis="x", direction="in")
                   ax2.tick_params(which="minor", axis="y", direction="in")
		#plt.subplots_adjust(top=0.850)	

		#axarr[1].plot(range(starting_n,ending_n+1),all_std)
                ax.set_xlabel('$n$', fontsize=16)
		#axarr[1].set_ylabel('sigma')
                ax.tick_params(labelsize=16)
                #plt.yscale('log')
                #ax1.set_xlim(-1.1,3.1)
                #ax.set_ylim(22,31)
                #a1.set_ylabel('Number', fontsize=16)
                #ax.set_xticks([0,200,400,600,800,1000,1200,1400,1600])
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.tick_params(which='both', width=2)
                ax.tick_params(which='major', length=7)
                ax.tick_params(which='minor', length=4, color='k')

                ax.tick_params(axis="x", direction="in")
                ax.tick_params(axis="y", direction="in")
                ax.tick_params(which="minor", axis="x", direction="in")
                ax.tick_params(which="minor", axis="y", direction="in")
                if parallel_transport:
                        plt.savefig('./figures/2D/SL_vs_n_PT'+filename+'.pdf')#,overwrite=True)
                else:
                        plt.savefig('./figures/2D/SL_vs_n_'+filename+'.pdf')#,overwrite=True)

	except IndexError: # try again, but with xlim on n -= 60
		print ('Index error, setting n-=100')
		plt.close()

		fig, axarr = plt.subplots(2, sharex=True, gridspec_kw= {'height_ratios':[3, 1]})
		axarr[0].plot(range(starting_n,ending_n+1),np.log10(all_sl))
		# print np.argmin(all_sl)
		axarr[0].set_title('SL vs n for '+filename+'\n\n')
		axarr[0].set_ylabel(r'$\log_{10}$ SL')
		axarr[0].set_xlim(2,n-100)
		axarr[0].set_ylim(-2.5,0)
		
		if angular_radius is not None:
			ax2 = axarr[0].twiny()
			ax2.set_xlabel('Median angular radius (degrees)')
			xticks = axarr[0].get_xticks()
			ax2.set_xticks(xticks)
			print (np.asarray(xticks,dtype='int'))
			xticks2 = np.append(0,angular_radius)[np.asarray(xticks,dtype='int')]
			ax2.set_xticklabels(tick_function(xticks2))
		plt.subplots_adjust(top=0.850)	

		axarr[1].plot(range(starting_n,ending_n+1),all_std)
		axarr[1].set_xlabel('n')
		axarr[1].set_ylabel('sigma')

		if parallel_transport:
			plt.savefig('./figures/2D/SL_vs_n_PT'+filename+'.png',overwrite=True)
		else:
			plt.savefig('./figures/2D/SL_vs_n_'+filename+'.png',overwrite=True)




	# plt.show()
	plt.close()

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
        position_angles = np.asarray(tdata['RPA'])

	#convert RAs and DECs to an array that has following layout: [[x1,y1,z1],[x2,y2,z2],etc]
        x = np.cos(np.radians(RAs)) * np.cos(np.radians(DECs))
        y = np.sin(np.radians(RAs)) * np.cos(np.radians(DECs))
        z = np.sin(np.radians(DECs))
        coordinates = np.vstack((x,y,z)).T
	
	#make a KDTree for quick NN searching	
        coordinates_tree = cKDTree(coordinates,leafsize=16)

	# for every source: find n closest neighbours, calculate median dist to furthes
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
        plt.figure()
        ax = plt.gca() 
        plt.plot(n_range,median[starting_n-1:]) # index 1 corresponds to n=2
	#plt.title('Angular radius vs n for ' + filename)
        ax.set_ylabel('Median angular radius (deg)', fontsize=16)
        ax.set_xlabel('$n$', fontsize=16)
        ax.tick_params(labelsize=16)
        #plt.yscale('log')
        #ax1.set_xlim(-1.1,3.1)
        #ax.set_ylim(22,31)
        #a1.set_ylabel('Number', fontsize=16)
        #ax.set_xticks([0,200,400,600,800,1000,1200,1400,1600])
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', width=2)
        ax.tick_params(which='major', length=7)
        ax.tick_params(which='minor', length=4, color='k')
        
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        ax.tick_params(which="minor", axis="x", direction="in")
        ax.tick_params(which="minor", axis="y", direction="in")
   
        plt.savefig('./figures/2D/angular_radius_vs_n_'+filename+'.pdf') #,overwrite=True)
	# plt.savefig('./figures/angular_radius_vs_n_'+filename+'.png',overwrite=True)
        plt.close()

        return median

def statistics(filename,tdata,redshift=False):
	if parallel_transport:
		if n < 998:
			Sn_mc = Table(fits.open('./data/2D/Sn_monte_carlo_PT'+filename+'.fits')[1].data)
		else: 
			Sn_mc = ascii.read('./data/2D/Sn_monte_carlo_PT'+filename+'_%d.csv' % n)

	else:
		if n < 998:
			Sn_mc = Table(fits.open('./data/2D/Sn_monte_carlo_'+filename+'.fits')[1].data)
		else:
			Sn_mc = ascii.read('./data/2D/Sn_monte_carlo_'+filename+'_%d.csv' % n)
	# calculate angular radius for number of nn, 
	angular_radius = angular_radius_vs_n(tdata,filename,n)
	if parallel_transport:
		Sn_data = angular_dispersion_vectorized_n_parallel_transport(tdata,n,redshift)
	else:
		Sn_data = angular_dispersion_vectorized_n(tdata,n,redshift)
	Sn_vs_n(tdata,Sn_mc,Sn_data,filename,angular_radius,n)

n = 1600

if __name__ == '__main__':

	#Running all the statistics without redshift
        redshift = False
        equal_width = False # If equal_width is false use the same flux and size cuts as the initial subsample

        results_dir = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results'
        filename = 'FRIIRGcat_final'#sys.argv[1]
        print(filename)
        tdata1 = pd.read_csv('%s/%s.csv' % (results_dir, filename)) #Table(fits.open('../%s.fits'%filename)[1].data)
        tdata2 = tdata1[tdata1['RA']>=90]
        tdata = tdata2[tdata2['RA']<=270]
        print('data length -->', len(tdata))
	#filename = sys.argv[1]
	#filenamprint (filename)
	#filenamtdata = Table(fits.open('../%s.fits'%filename)[1].data)
	#filenamif position_angle: tdata['RPA'] = tdata['position_angle'] # overwrite to use 'position_angle' only

	#filenamif position_angle: filename += '_PA' # only when using 'position_angle'

	#filenamtry:
	#filenam	tdata['RA'] = tdata['RA_2']
	#filenam	tdata['DEC'] = tdata['DEC_2']
	#filenamexcept KeyError:
	#	pass # then we're doing biggest_selection


        tdata_original = tdata
        filename_original = filename

        statistics(filename,tdata,redshift)


	#if equal_width: 
	#	fluxbins = select_flux_bins1(tdata_original)
	#else:
	#	fluxbins = select_flux_bins_cuts_biggest_selection(tdata_original)
	#for key in fluxbins:
	#	tdata = fluxbins[key]
	#	filename = filename_original + 'flux%s'%key
	#	
	#	statistics(filename,tdata,redshift)

	#if equal_width:
	#	sizebins = select_size_bins1(tdata_original)
	#else:
	#	sizebins = select_size_bins_cuts_biggest_selection(tdata_original)
	#for key in sizebins:
	#	tdata = sizebins[key]
	#	filename = filename_original + 'size%s'%key
	#	
	#	if len(tdata) != 0:
	#		statistics(filename,tdata,redshift)
	#	else:
	#		print ('This bin contains 0 sources:')
	#		print (filename)

