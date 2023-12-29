'''
The original code from https://github.com/ErikOsinga/MRP1,
developed by Erik Osinga.
'''
import sys

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
		,select_power_bins_cuts_VA_selection, select_physical_size_bins_cuts_VA_selection)

from astropy.io import ascii

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import pandas as pd

parallel_transport = True
print ('Using parallel_transport = %s' %parallel_transport)

#position_angle = sys.argv[2]
#if position_angle == 'True':
#	position_angle = True
#else:
position_angle = False
if position_angle:
	print ('Using position_angle')
else: 
	print ('Using RPA')

def tick_function(X):
	return ["%.2f" % z for z in X]

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
        return all_sl  	

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

	plt.plot(n_range,median[starting_n-1:]) # index 1 corresponds to n=2
	plt.title('Angular radius vs n for ' + filename)
	plt.ylabel('Median angular radius (deg)')
	plt.xlabel('n')
	# plt.savefig('/data1/osinga/figures/statistics/show/angular_radius_vs_n_'+filename+'.png',overwrite=True)
	# plt.savefig('./figures/angular_radius_vs_n_'+filename+'.png',overwrite=True)
	plt.close()
		
	return median

def statistics(filename,tdata,redshift=False):
        if parallel_transport:
                if n < 998:
                        Sn_mc = Table(fits.open('./data/Sn_monte_carlo_PT'+filename+'_%s.fits' % n)[1].data)
                else:
                        Sn_mc = ascii.read('./data/Sn_monte_carlo_PT'+filename+'_%s.csv' % n)

        else:
                if n < 998:
                        Sn_mc = Table(fits.open('./data/Sn_monte_carlo_'+filename+'_%s.fits' % n)[1].data)
                else:
                        Sn_mc = ascii.read('./data/Sn_monte_carlo_'+filename+'_%s.csv' % n)

	# calculate angular radius for number of nn, 
        angular_radius = angular_radius_vs_n(tdata,filename,n)
        if parallel_transport:
                Sn_data = angular_dispersion_vectorized_n_parallel_transport(tdata,n,redshift)
        else:
                Sn_data = angular_dispersion_vectorized_n(tdata,n,redshift)
        SL = Sn_vs_n(tdata,Sn_mc,Sn_data,filename,angular_radius,n)
        return SL, angular_radius

n = 1600

if __name__ == '__main__':
#####################################################################################
	#Running all the statistics with redshift
        equal_width = False
        redshift = True 

        results_dir = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results'
        filename = 'FRIIRGcat_final'#sys.argv[1]
        print(filename)
        tdata1 = pd.read_csv('%s/%s.csv' % (results_dir, filename)) #Table(fits.open('../%s.fits'%filename)[1].data)
        
        tdata2 = tdata1[tdata1['RA']>=90]
        tdata3 = tdata2[tdata2['RA']<=270]
        print('data length -->', len(tdata3))
        #tdata_original = tdata
        
        # With redshift
        print ('Using redshift..') ## edited for not actually using z
        z_available = tdata3[tdata3['z'].notnull()]#np.invert(np.isnan(tdata['z_best']))

	#filename = sys.argv[1]
	#print (filename)
	#tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	#if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	#tdata['RA'] = tdata['RA_2']
	#tdata['DEC'] = tdata['DEC_2']

	#print ('Using redshift..')
	#z_available = np.invert(np.isnan(tdata['z_best']))
	#z_zero = tdata['z_best'] == 0#
	## also remove sources with redshift 0, these dont have 3D positions
	#z_available = np.logical_xor(z_available,z_zero)
	#print ('Number of sources with available redshift:', np.sum(z_available))
	## filename += '_only_redshift_sources_' ## edited for not actually using z
	#if position_angle: filename += '_PA' # only when using 'position_angle'
        filename += '_redshift'

        tdata = z_available #tdata[z_available]
        print('data length for redshift -->', len(tdata))
        filename_original = filename
        tdata_original = tdata # use redshift data for power bins and size bins !!
        SL_z, angular_radius_z = statistics(filename,tdata,redshift)


	#Running all the statistics with redshift, but not using redshift
        equal_width = False
        redshift = False

        filename = 'FRIIRGcat_final' #sys.argv[1]
	#print (filename)
	#tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	#if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	#tdata['RA'] = tdata['RA_2']
	#tdata['DEC'] = tdata['DEC_2']

	#print ('Using redshift.. but only for selection')
	#z_available = np.invert(np.isnan(tdata['z_best']))
	#z_zero = tdata['z_best'] == 0#
	## also remove sources with redshift 0, these dont have 3D positions
	#z_available = np.logical_xor(z_available,z_zero)
	#print ('Number of sources with available redshift:', np.sum(z_available))
	#if position_angle: filename += '_PA' # only when using 'position_angle'
	#filename += '_only_redshift_sources_' ## edited for not actually using z

	#tdata_original = tdata
        filename_original = filename

	#tdata = tdata[z_available]

        tdata_original = tdata # use redshift data for power bins and size bins !!
        SL, angular_radius = statistics(filename,tdata,redshift)

        plt.figure()
        ax = plt.gca()
        starting_n = 1
        ending_n = n
        ax.plot(range(starting_n,ending_n+1),np.log(SL_z),'b:', label='3D')
        ax.plot(range(starting_n,ending_n+1),np.log(SL), 'b', label='2D')
        # print np.argmin(all_sl)
        #axarr[0].set_title('SL vs n for '+filename+'\n\n')
        ax.set_ylabel('log(SL)', fontsize=16)
        ax.set_xlim(0,n)#400)#n)
        ax.set_ylim(-6,0)
        
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
        ax.set_xlabel('$n$', fontsize=16)
        ax.tick_params(labelsize=16)
        #plt.yscale('log')
        #ax.set_xlim(0,3.1)
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
        ax.legend()
        if parallel_transport:
                plt.savefig('./figures/SL_vs_n_PT'+filename+'_3D.pdf')#,overwrite=True)
        else:
                plt.savefig('./figures/SL_vs_n_'+filename+'_3D.pdf')#,overwrite=True)

