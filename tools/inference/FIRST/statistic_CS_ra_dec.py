import numpy as np
import pandas as pd
import matplotlib
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import scipy.stats as sta

data_dir = "/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results"

match=pd.read_csv(os.path.join(data_dir,'FIRST_infer_part0-4_th0.1_first_14dec17_matched_cs_nocentre_final.csv'),sep=',')

match1=match[match['int_flux']>0]



plt.figure(600)
#fig,ax=plt.subplots(nrows=1,ncols=2)# , sharex='col', sharey='row')
#fig.delaxes(ax[0][1])
ax1 = plt.gca()#ax[0]#[0]
#ax2 = ax[1][0]
#ax3 = ax[1]#[1]

def gaussian(x, mean, amplitude, standard_deviation):
    return (amplitude * np.exp( - (x - mean)**2 / (2*standard_deviation ** 2)))
#print(np.mean((match['ra_2']-match['RA_1'])*3600))
#samples1 = (match['ra_2']-match['RA_1'])*3600
samples1 = (match['ra_2']-match['RA_1'])*3600*np.cos(match['dec_2']/180.0*np.pi)
mean1 =np.mean(samples1)
sigma1 = np.std(samples1)

print("new ra mean %f" % mean1)
bin_heights1, bin_borders1, _ = ax1.hist(samples1, bins=20, facecolor='#1F77B4', align='mid', histtype='step', edgecolor='#1F77B4', linewidth=2, linestyle='-', label=r'$\Delta_{RA}$')
#bin_heights2, bin_borders2, _ = plt.hist(samples2, bins=50,   facecolor='#FF7F0E', align='mid', histtype='step', edgecolor='#FF7F0E', linewidth=2, linestyle='-', label=r'$\Delta_{DEC}$ mean=%.1f std=%.1f' % (mean2, sigma2))

#bin_heights, bin_borders, _ = plt.hist(samples_new, bins = 40, histtype='step', color='b',linewidth=2, label='histogram')
bin_centers1 = bin_borders1[:-1] + np.diff(bin_borders1) / 2
popt1, _ = curve_fit(gaussian, bin_centers1, bin_heights1, p0=[1., mean1, sigma1])
print("ra fitted para: %s" % popt1)

x_interval_for_fit1 = np.linspace(-5, 5, len(samples1))
ax1.plot(x_interval_for_fit1, gaussian(x_interval_for_fit1, *popt1), '--k', linewidth=2, label=r'fitted $\Delta_{RA}$')
x_data1 = x_interval_for_fit1
coeff1 = popt1
yy=np.linspace(0, popt1[1], 50)
xx=np.ones(yy.shape)*popt1[0]
ax1.plot(xx,yy,'-r')
plt.xlabel(r'$\Delta_{RA}$ (arcsec)', fontsize=16)
#ax1.set_xticklabels([])
ax1.tick_params(labelsize=16)
#plt.yscale('log')
ax1.set_xlim(-5,5)
#ax1.set_ylim(0,500000)
ax1.set_ylabel('Number of sources', fontsize=16)
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(which='both', width=2)
ax1.tick_params(which='major', length=7)
ax1.tick_params(which='minor', length=4, color='k')

ax1.tick_params(axis="x", direction="in")
ax1.tick_params(axis="y", direction="in")
ax1.tick_params(which="minor", axis="x", direction="in")
ax1.tick_params(which="minor", axis="y", direction="in")
#plt.tick_params(labelsize=20)
#labels = ax1.get_xticklabels() + ax1.get_yticklabels()
#[label.set_fontname('Times New Roman') for label in labels]
#plt.tick_params(which='major', length=8,direction='in')
#plt.tick_params(which='minor', length=4,direction='in')
#ax1.tick_params(axis='both', which='both', width=1,direction='in')
#ax1.tick_params(axis = 'x', which = 'major', labelsize = 14, pad=4,direction='in')
#ax1.tick_params(axis = 'y', which = 'major', labelsize = 14, pad=4,direction='in')

plt.tight_layout()

plt.savefig('ra_hist_new.png', dpi=600,format="png")
plt.savefig('ra_hist_new.pdf', dpi=300,format="pdf")

plt.figure(601)
ax3 = plt.gca()

print(np.mean((match['dec_2']-match['DEC_1'])*3600))
samples2 = (match['dec_2']-match['DEC_1'])*3600
mean2 =np.mean(samples2)
sigma2 = np.std(samples2)
print(mean2, sigma2)
bin_heights2, bin_borders2, _ = ax3.hist(samples2, bins=50, facecolor='#1F77B4', align='mid', histtype='step', edgecolor='#1F77B4', linewidth=2, linestyle='-') #, orientation='horizontal') #, label=r'$\Delta_{Dec}$')
y_Prec = sta.norm.pdf( bin_borders2, mean2, sigma2)
bin_centers2 = bin_borders2[:-1] + np.diff(bin_borders2) / 2
popt2, _ = curve_fit(gaussian, bin_centers2, bin_heights2, p0=[1., mean2, sigma2], maxfev = 500000)
#print("dec fitted para: %s" % popt2)

mean,std=norm.fit(bin_borders2)

x_interval_for_fit2 = np.linspace(-5, 5, len(samples2))
#ax3.plot(gaussian(x_interval_for_fit2, mean, np.max(bin_heights2), std*0.1), x_interval_for_fit2, '--k', linewidth=2, label=r'fitted $\Delta_{Dec}$')
ax3.plot(x_interval_for_fit2, gaussian(x_interval_for_fit2, mean, np.max(bin_heights2), std*0.1), '--k', linewidth=2, label=r'fitted $\Delta_{Dec}$')
#coeff1 = popt1
# x_interval_for_fit2 = np.linspace(-5, 5, len(y_Prec))
# plt.plot(x_interval_for_fit2, y_Prec*np.max(bin_heights2) , '--r')
coeff2 = popt2
print(mean, np.max(bin_heights2), std*0.1)
yy=np.linspace(0, np.max(bin_heights2), 50)
xx=np.ones(yy.shape)*mean
plt.plot(xx, yy, '-r')


#plt.ylabel(r'$\Delta_{\rm Dec}$ (arcsec)', fontsize=12)
plt.tick_params(labelsize=16)
#plt.yscale('log')
plt.xlim(-5,5)
# plt.ylim(0,1)
plt.xlabel(r'$\Delta_{Dec}$ (arcsec)', fontsize=16)
plt.ylabel('Number of sources', fontsize=16)
#ax3.set_yticklabels([])
ax3.xaxis.set_minor_locator(AutoMinorLocator())
ax3.yaxis.set_minor_locator(AutoMinorLocator())
ax3.tick_params(which='both', width=2)
ax3.tick_params(which='major', length=7)
ax3.tick_params(which='minor', length=4, color='k')

ax3.tick_params(axis="x", direction="in")
ax3.tick_params(axis="y", direction="in")
ax3.tick_params(which="minor", axis="x", direction="in")
ax3.tick_params(which="minor", axis="y", direction="in")
#plt.tick_params(labelsize=20)
#labels = ax3.get_xticklabels() + ax3.get_yticklabels()
#[label.set_fontname('Times New Roman') for label in labels]
#plt.tick_params(which='major', length=8,direction='in')
#plt.tick_params(which='minor', length=4,direction='in')
#ax3.tick_params(axis='both', which='both', width=1,direction='in')
#ax3.tick_params(axis = 'x', which = 'major', labelsize = 14, pad=4,direction='in')
#ax3.tick_params(axis = 'y', which = 'major', labelsize = 14, pad=4,direction='in')

import matplotlib.ticker as mticker
#labels = ax2.get_yticklabels()
# remove the first and the last labels
#labels[0] = labels[-1] = ""
    # set these new labels
#print(labels)
#ax2.set_yticklabels(labels)
#print(ticks_loc)
plt.tight_layout()
#plt.subplots_adjust(hspace=0)
#plt.subplots_adjust(wspace=0)
plt.savefig('dec_hist_new.png', dpi=600,format="png")
plt.savefig('dec_hist_new.pdf', dpi=300,format="pdf")
