#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.mlab as mlab
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:

data_dir = "/p9550/MWA/GLEAM/blao/hetu_images/deep_learn/inference_sets/hetu_results/final"

#pd.read_csv('../catalogue/DR1_170-231MHz.csv',sep=",")
# DR1=pd.read_csv('../old/catalogue/DR1.csv',sep=",")
#hetu=pd.read_csv(os.path.join(data_dir,'FIRST_all_final.csv'))
match=pd.read_csv(os.path.join(data_dir,'FIRST_hetu_14dec17_matched_cs.csv'),sep=',')
#non_DR1=pd.read_csv('../catalogue/non_match_DR1_170-231MHz.csv',sep=',')
#non_hetu=pd.read_csv('../catalogue/non_match_hetu_170-231MHz.csv',sep=',')


# # DR1 和 hetu 的星表做对比

# In[3]:

major_first = match['FITTED_MAJOR'].values
major_hetu = match['major_1'].values

plt.figure(10)
ax = plt.gca()
#plt.hist(offset_x, bins=33, label=r'$\Delta x$', color='b')
plt.hist(major_hetu/major_first,bins=500, facecolor='#FF7F0E', align='mid', histtype='bar', edgecolor='r', linewidth=0.2, linestyle='-',alpha=0.5)
#plt.hist(offset_y, bins=33,   alpha=0.5, label=r'$\Delta x$', color='r')
#plt.hist(offset_y, bins=33,   facecolor='None', label=r'$\Delta y$', edgecolor='r')
#ax.hist(offset_x,bins=33, facecolor='#FF7F0E', align='mid', histtype='bar', edgecolor='k', linewidth=0.2, linestyle='-',alpha=0.5, label=r'$\Delta y$')
plt.xlabel(r'$Major_{\rm Hetu}/Major_{\rm FIRST}}$', fontsize=18)
plt.tick_params(labelsize=16)
#plt.yscale('log')
plt.xlim(0,2)
plt.ylabel('Number of sources', fontsize=18)
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', width=2)
ax.tick_params(which='major', length=7)
ax.tick_params(which='minor', length=4, color='k')

plt.tick_params(axis="x", direction="in")
plt.tick_params(axis="y", direction="in")
plt.tick_params(which="minor", axis="x", direction="in")
plt.tick_params(which="minor", axis="y", direction="in")
#plt.legend(loc='upper left', fontsize=16)
plt.tight_layout()
plt.savefig('major.png', dpi=600,format="png",bbox_inches = 'tight')

minor_first = match['FITTED_MINOR'].values
minor_hetu = match['minor_1'].values

plt.figure(11)
ax = plt.gca()
#plt.hist(offset_x, bins=33, label=r'$\Delta x$', color='b')
plt.hist(minor_hetu/minor_first,bins=500, facecolor='#FF7F0E', align='mid', histtype='bar', edgecolor='r', linewidth=0.2, linestyle='-',alpha=0.5)
#plt.hist(offset_y, bins=33,   alpha=0.5, label=r'$\Delta x$', color='r')
#plt.hist(offset_y, bins=33,   facecolor='None', label=r'$\Delta y$', edgecolor='r')
#ax.hist(offset_x,bins=33, facecolor='#FF7F0E', align='mid', histtype='bar', edgecolor='k', linewidth=0.2, linestyle='-',alpha=0.5, label=r'$\Delta y$')
plt.xlabel(r'$Minor_{\rm Hetu}/Minor_{\rm FIRST}}$', fontsize=18)
plt.tick_params(labelsize=16)
#plt.yscale('log')
plt.xlim(0.8,1.2)
plt.ylabel('Number of sources', fontsize=18)
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', width=2)
ax.tick_params(which='major', length=7)
ax.tick_params(which='minor', length=4, color='k')

plt.tick_params(axis="x", direction="in")
plt.tick_params(axis="y", direction="in")
plt.tick_params(which="minor", axis="x", direction="in")
plt.tick_params(which="minor", axis="y", direction="in")
#plt.legend(loc='upper left', fontsize=16)
plt.tight_layout()
plt.savefig('minor.png', dpi=600,format="png",bbox_inches = 'tight')

pa_first = match['FITTED_POSANG'].values
pa_hetu = match['pa'].values

plt.figure(12)
ax = plt.gca()
#plt.hist(offset_x, bins=33, label=r'$\Delta x$', color='b')
plt.hist(pa_hetu-pa_first,bins=500, facecolor='#FF7F0E', align='mid', histtype='bar', edgecolor='r', linewidth=0.2, linestyle='-',alpha=0.5)
#plt.hist(offset_y, bins=33,   alpha=0.5, label=r'$\Delta x$', color='r')
#plt.hist(offset_y, bins=33,   facecolor='None', label=r'$\Delta y$', edgecolor='r')
#ax.hist(offset_x,bins=33, facecolor='#FF7F0E', align='mid', histtype='bar', edgecolor='k', linewidth=0.2, linestyle='-',alpha=0.5, label=r'$\Delta y$')
plt.xlabel(r'$\Delta PA$', fontsize=18)
plt.tick_params(labelsize=16)
#plt.yscale('log')
plt.xlim(-90,90)
plt.ylabel('Number of sources', fontsize=18)
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', width=2)
ax.tick_params(which='major', length=7)
ax.tick_params(which='minor', length=4, color='k')

plt.tick_params(axis="x", direction="in")
plt.tick_params(axis="y", direction="in")
plt.tick_params(which="minor", axis="x", direction="in")
plt.tick_params(which="minor", axis="y", direction="in")
#plt.legend(loc='upper left', fontsize=16)
plt.tight_layout()
plt.savefig('pa.png', dpi=600,format="png",bbox_inches = 'tight')



major_first = match['MAJOR_2'].values
major_hetu = match['deconv_major'].values

major_first_new = []
major_hetu_new = []
for m in range(len(major_first)):
    if major_first[m] != 0.0 and (not np.isnan(major_hetu[m])):
       major_first_new.append(major_first[m])
       major_hetu_new.append(major_hetu[m])

major_first_new = np.array(major_first_new)
major_hetu_new = np.array(major_hetu_new)

plt.figure(20)
ax = plt.gca()
#plt.hist(offset_x, bins=33, label=r'$\Delta x$', color='b')
plt.hist(major_hetu_new/major_first_new,bins=10000, facecolor='#FF7F0E', align='mid', histtype='bar', edgecolor='r', linewidth=0.2, linestyle='-',alpha=0.5)
#plt.hist(offset_y, bins=33,   alpha=0.5, label=r'$\Delta x$', color='r')
#plt.hist(offset_y, bins=33,   facecolor='None', label=r'$\Delta y$', edgecolor='r')
#ax.hist(offset_x,bins=33, facecolor='#FF7F0E', align='mid', histtype='bar', edgecolor='k', linewidth=0.2, linestyle='-',alpha=0.5, label=r'$\Delta y$')
plt.xlabel(r'$Major_{\rm Hetu}/Major_{\rm FIRST}}$', fontsize=18)
plt.tick_params(labelsize=16)
#plt.yscale('log')
plt.xlim(0,5)
plt.ylabel('Number of sources', fontsize=18)
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', width=2)
ax.tick_params(which='major', length=7)
ax.tick_params(which='minor', length=4, color='k')

plt.tick_params(axis="x", direction="in")
plt.tick_params(axis="y", direction="in")
plt.tick_params(which="minor", axis="x", direction="in")
plt.tick_params(which="minor", axis="y", direction="in")
#plt.legend(loc='upper left', fontsize=16)
plt.tight_layout()
plt.savefig('major_deconv.png', dpi=600,format="png",bbox_inches = 'tight')

minor_first = match['MINOR_2'].values
minor_hetu = match['deconv_minor'].values

minor_first_new = []
minor_hetu_new = []
for m in range(len(minor_first)):
    if minor_first[m] != 0.0 and (not np.isnan(minor_hetu[m])):
       minor_first_new.append(minor_first[m])
       minor_hetu_new.append(minor_hetu[m])

minor_first_new = np.array(minor_first_new)
minor_hetu_new = np.array(minor_hetu_new)

print("deconv len: %d" % len(minor_first_new))

minor_first_maj = match['MINOR_2'].values
minor_first_min = match['MINOR_2'].values
minor_first_maj_new = []
minor_first_min_new = []
for m in range(len(minor_first_min)):
    if minor_first_maj[m] != 0.0 and minor_first_min[m] != 0.0:
       minor_first_maj_new.append(minor_first_maj[m])

minor_first_maj_new = np.array(minor_first_maj_new)

print("deconv first len: %d" % len(minor_first_maj_new))

plt.figure(21)
ax = plt.gca()
#plt.hist(offset_x, bins=33, label=r'$\Delta x$', color='b')
plt.hist(minor_hetu_new/minor_first_new,bins=10000, facecolor='#FF7F0E', align='mid', histtype='bar', edgecolor='r', linewidth=0.2, linestyle='-',alpha=0.5)
#plt.hist(offset_y, bins=33,   alpha=0.5, label=r'$\Delta x$', color='r')
#plt.hist(offset_y, bins=33,   facecolor='None', label=r'$\Delta y$', edgecolor='r')
#ax.hist(offset_x,bins=33, facecolor='#FF7F0E', align='mid', histtype='bar', edgecolor='k', linewidth=0.2, linestyle='-',alpha=0.5, label=r'$\Delta y$')
plt.xlabel(r'$Minor_{\rm Hetu}/Minor_{\rm FIRST}}$', fontsize=18)
plt.tick_params(labelsize=16)
#plt.yscale('log')
plt.xlim(0,5)
plt.ylabel('Number of sources', fontsize=18)
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', width=2)
ax.tick_params(which='major', length=7)
ax.tick_params(which='minor', length=4, color='k')

plt.tick_params(axis="x", direction="in")
plt.tick_params(axis="y", direction="in")
plt.tick_params(which="minor", axis="x", direction="in")
plt.tick_params(which="minor", axis="y", direction="in")
#plt.legend(loc='upper left', fontsize=16)
plt.tight_layout()
plt.savefig('minor_deconv.png', dpi=600,format="png",bbox_inches = 'tight')

pa_first = match['POSANG'].values
pa_hetu = match['deconv_pa'].values


pa_first_new = []
pa_hetu_new = []
for m in range(len(pa_first)):
    if pa_first[m] != 0.0 and (not np.isnan(pa_hetu[m])):
       pa_first_new.append(pa_first[m])
       pa_hetu_new.append(pa_hetu[m])

pa_first_new = np.array(pa_first_new)
pa_hetu_new = np.array(pa_hetu_new)

plt.figure(22)
ax = plt.gca()
#plt.hist(offset_x, bins=33, label=r'$\Delta x$', color='b')
plt.hist(pa_hetu_new-pa_first_new,bins=100, facecolor='#FF7F0E', align='mid', histtype='bar', edgecolor='r', linewidth=0.2, linestyle='-',alpha=0.5)
#plt.hist(offset_y, bins=33,   alpha=0.5, label=r'$\Delta x$', color='r')
#plt.hist(offset_y, bins=33,   facecolor='None', label=r'$\Delta y$', edgecolor='r')
#ax.hist(offset_x,bins=33, facecolor='#FF7F0E', align='mid', histtype='bar', edgecolor='k', linewidth=0.2, linestyle='-',alpha=0.5, label=r'$\Delta y$')
plt.xlabel(r'$\Delta PA$', fontsize=18)
plt.tick_params(labelsize=16)
#plt.yscale('log')
plt.xlim(-90,90)
plt.ylabel('Number of sources', fontsize=18)
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', width=2)
ax.tick_params(which='major', length=7)
ax.tick_params(which='minor', length=4, color='k')

plt.tick_params(axis="x", direction="in")
plt.tick_params(axis="y", direction="in")
plt.tick_params(which="minor", axis="x", direction="in")
plt.tick_params(which="minor", axis="y", direction="in")
#plt.legend(loc='upper left', fontsize=16)
plt.tight_layout()
plt.savefig('pa_deconv.png', dpi=600,format="png",bbox_inches = 'tight')


match1=match[match['int_flux']>0]


coefficients_flux,covariance_flux=np.polyfit(match['FPEAK'],match['peak_flux']*1000,deg=1,full=False,cov=True)
m_flux=coefficients_flux[0]
c_flux=coefficients_flux[1]
coefficients_err_flux=np.sqrt(np.diag(covariance_flux))
m_err_flux=coefficients_err_flux[0]
c_err_flux=coefficients_err_flux[1]


print(m_flux)
print(m_err_flux)



z_flux=np.polyfit(match1['FPEAK'],match1['peak_flux']*1000,1)
p_flux= np.poly1d(z_flux)

flux_unmatched=match1[match1['peak_flux']*1000>match1['FPEAK']*m_flux+0.9]
flux_unmatched1=match1[match1['peak_flux']*1000<match1['FPEAK']*m_flux-0.9]

flux_unmatched.to_csv(os.path.join(data_dir, 'peak_flux_unmatched_14dec17.csv'),index=0)
#flux_unmatched1.to_csv('../catalogue/flux_unmatched1.csv',index=0)


# In[19]:


fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
x=np.linspace(0,23000,100)
fig.subplots_adjust(top=0.95,bottom=0.1,left=0.14,right=0.965)
plt.plot(match1['FPEAK'],match1['peak_flux']*1000,'o',c='tab:blue')
# plt.plot(flux_unmatched['peak_flux_170_231MHz'],flux_unmatched['peak_flux'],'o',c='tab:pink')
# plt.plot(flux_unmatched1['peak_flux_170_231MHz'],flux_unmatched1['peak_flux'],'o',c='tab:orange')


plt.plot(x,x,'r--')
#plt.plot(x,p_flux(x),'--',color='tab:blue')


plt.tick_params(labelsize=20)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.tick_params(which='major', length=8,direction='in')
plt.tick_params(which='minor', length=4,direction='in')
ax1.tick_params(axis='both', which='both', width=1,direction='in')
ax1.tick_params(axis = 'x', which = 'major', labelsize = 18, pad=4,direction='in')
ax1.tick_params(axis = 'y', which = 'major', labelsize = 18, pad=4,direction='in')

plt.xlabel(r'$S_{FIRST\_{14dec17}}$(mJy)',fontsize=20,labelpad=0)
plt.ylabel(r'$S_{HeTu}$(mJy)',fontsize=20,labelpad=0)
plt.title('peak flux',fontsize=20)
ax1.annotate(r'$\frac{S_{HeTu}}{S_{FIRST\_{14dec17}}}=%.2f \pm %.3f$' % (m_flux,m_err_flux) ,xy=(0.2,0.85),xycoords='figure fraction',color='k',fontsize=20)

#plt.xlim(-0.5,5000)
#plt.ylim(-0.5,5000)

xminors = matplotlib.ticker.AutoMinorLocator(5)
yminors = matplotlib.ticker.AutoMinorLocator(5)
ax1.xaxis.set_minor_locator(xminors)
ax1.yaxis.set_minor_locator(yminors)

plt.savefig('peak_flux.png', dpi=600,format="png",bbox_inches = 'tight')

plt.figure(3)
ax = plt.gca()
#plt.hist(offset_x, bins=33, label=r'$\Delta x$', color='b')
plt.hist(match1['peak_flux']*1000.0/match1['FPEAK'],bins=200, facecolor='#FF7F0E', align='mid', histtype='bar', edgecolor='r', linewidth=0.2, linestyle='-',alpha=0.5)
#plt.hist(offset_y, bins=33,   alpha=0.5, label=r'$\Delta x$', color='r')
#plt.hist(offset_y, bins=33,   facecolor='None', label=r'$\Delta y$', edgecolor='r')
#ax.hist(offset_x,bins=33, facecolor='#FF7F0E', align='mid', histtype='bar', edgecolor='k', linewidth=0.2, linestyle='-',alpha=0.5, label=r'$\Delta y$')
plt.xlabel(r'$S_{Hetu}/S_{FIRST\_{14dec17}}$', fontsize=18)
plt.tick_params(labelsize=16)
plt.yscale('log')
plt.xlim(0,20)
plt.ylabel('Number of sources', fontsize=18)
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', width=2)
ax.tick_params(which='major', length=7)
ax.tick_params(which='minor', length=4, color='k')

plt.tick_params(axis="x", direction="in")
plt.tick_params(axis="y", direction="in")
plt.tick_params(which="minor", axis="x", direction="in")
plt.tick_params(which="minor", axis="y", direction="in")
#plt.legend(loc='upper left', fontsize=16)
plt.tight_layout()
plt.savefig('peak_flux_hetu_first.png', dpi=600,format="png",bbox_inches = 'tight')
# In[20]:


coefficients_flux,covariance_flux=np.polyfit(match1['FINT'],match1['int_flux']*1000,deg=1,full=False,cov=True)
m_flux=coefficients_flux[0]
c_flux=coefficients_flux[1]
coefficients_err_flux=np.sqrt(np.diag(covariance_flux))
m_err_flux=coefficients_err_flux[0]
c_err_flux=coefficients_err_flux[1]


print(m_flux)
print(m_err_flux)

ax.tick_params(which='minor', length=4, color='k')

plt.tick_params(axis="x", direction="in")
plt.tick_params(axis="y", direction="in")
plt.tick_params(which="minor", axis="x", direction="in")
plt.tick_params(which="minor", axis="y", direction="in")
plt.legend(loc='upper left', fontsize=16)
plt.tight_layout()

# In[20]:


coefficients_flux,covariance_flux=np.polyfit(match1['FINT'],match1['int_flux']*1000,deg=1,full=False,cov=True)
m_flux=coefficients_flux[0]
c_flux=coefficients_flux[1]
coefficients_err_flux=np.sqrt(np.diag(covariance_flux))
m_err_flux=coefficients_err_flux[0]
c_err_flux=coefficients_err_flux[1]


print(m_flux)
print(m_err_flux)

z_flux=np.polyfit(match['FINT'],match['int_flux']*1000,1)
p_flux= np.poly1d(z_flux)

int_flux_unmatched=match[match['int_flux']*1000>match['FINT']+0.9]
int_flux_unmatched1=match[match['int_flux']*1000<match['FINT']-0.9]

int_flux_unmatched.to_csv(os.path.join(data_dir, 'int_flux_unmatched_14dec17.csv'),index=0)
#int_flux_unmatched1.to_csv('../catalogue/int_flux_unmatched1.csv',index=0)


# In[21]:


fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
x=np.linspace(0,23000,100)
fig.subplots_adjust(top=0.95,bottom=0.1,left=0.14,right=0.965)
plt.plot(match1['FINT'],match1['int_flux']*1000,'o',c='tab:blue')
# plt.plot(int_flux_unmatched['int_flux_170_231MHz'],int_flux_unmatched['int_flux'],'o',c='tab:pink')
# plt.plot(int_flux_unmatched1['int_flux_170_231MHz'],int_flux_unmatched1['int_flux'],'o',c='tab:orange')



plt.plot(x,x,'r--')
# plt.plot(x,p_flux(x),'--',color='tab:green',alpha=1)


plt.tick_params(labelsize=20)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.tick_params(which='major', length=8,direction='in')
plt.tick_params(which='minor', length=4,direction='in')
ax1.tick_params(axis='both', which='both', width=1,direction='in')
ax1.tick_params(axis = 'x', which = 'major', labelsize = 18, pad=4,direction='in')
ax1.tick_params(axis = 'y', which = 'major', labelsize = 18, pad=4,direction='in')

plt.xlabel(r'$S_{FIRST\_{14dec17}}$(mJy)',fontsize=20,labelpad=0)
plt.ylabel(r'$S_{HeTu}$(mJy)',fontsize=20,labelpad=0)
plt.title('int flux',fontsize=20)
ax1.annotate(r'$\frac{S_{HeTu}}{S_{FIRST\_{14dec17}}}=%.2f \pm %.3f$' % (m_flux,m_err_flux) ,xy=(0.2,0.85),xycoords='figure fraction',color='k',fontsize=20)

#plt.xlim(-0.5,5000)
#plt.ylim(-0.5,5000)

xminors = matplotlib.ticker.AutoMinorLocator(5)
yminors = matplotlib.ticker.AutoMinorLocator(5)
ax1.xaxis.set_minor_locator(xminors)
ax1.yaxis.set_minor_locator(yminors)

plt.savefig('int_flux.png',dpi=600, format="png",bbox_inches = 'tight')

#plt.figure(5)
fig,ax=plt.subplots(nrows=2,ncols=1,sharey=True)
def gaussian(x, mean, amplitude, standard_deviation):
    return (amplitude * np.exp( - (x - mean)**2 / (2*standard_deviation ** 2)))
#ax = plt.gca()
ax1 = ax[0]#plt.subplot(211)
ax2 = ax[1]#plt.subplot(212)
#plt.hist(offset_x, bins=33, label=r'$\Delta x$', color='b')
#plt.hist(match1['centre_ra']-match1['RA_2'],bins=200, facecolor='None', align='mid', histtype='bar', edgecolor='r', linewidth=0.2, linestyle='-',alpha=0.5)
print(np.mean((match1['ra_1']-match1['RA_2'])*3600))
print(np.mean((match1['dec_1']-match1['DEC_2'])*3600))
#plt.hist((match1['centre_ra']-match1['RA_2'])*3600, bins=50, facecolor='#1F77B4', align='mid', histtype='step', edgecolor='#1F77B4', linewidth=0.2, linestyle='-',alpha=0.5, label=r'$\Delta\alpha$')
#plt.hist((match1['centre_dec']-match1['DEC_2'])*3600, bins=50,   facecolor='#FF7F0E', align='mid', histtype='step', edgecolor='#FF7F0E', linewidth=0.2, linestyle='-',alpha=0.5, label=r'$\Delta\delta$')
#ax.hist(offset_x,bins=33, facecolor='#FF7F0E', align='mid', histtype='bar', edgecolor='k', linewidth=0.2, linestyle='-',alpha=0.5, label=r'$\Delta y$')
samples1 = (match1['ra_1']-match1['RA_2'])*3600
samples2 = (match1['dec_1']-match1['DEC_2'])*3600
#samples = (match['peak_flux']*1000.0)/match['FPEAK']
#samples_1 = samples1[samples1 <=0.75]
#samples1 = samples_1[samples_1 >=-0.75]
#samples_2 = samples2[samples2 <=0.75]
#samples2 = samples_2[samples_2 >=-0.75]
mean1 =np.mean(samples1)
sigma1 = np.std(samples1)
mean2 =np.mean(samples2)
sigma2 = np.std(samples2)

bin_heights1, bin_borders1, _ = ax1.hist(samples1, bins=50, facecolor='#1F77B4', align='mid', histtype='step', edgecolor='#1F77B4', linewidth=2, linestyle='-', label=r'$\Delta_{RA}$ mean=%.1f std=%.1f' % (mean1, sigma1))
bin_heights2, bin_borders2, _ = ax2.hist(samples2, bins=50,   facecolor='#FF7F0E', align='mid', histtype='step', edgecolor='#FF7F0E', linewidth=2, linestyle='-', label=r'$\Delta_{DEC}$ mean=%.1f std=%.1f' % (mean2, sigma2))

#bin_heights, bin_borders, _ = plt.hist(samples_new, bins = 40, histtype='step', color='b',linewidth=2, label='histogram')
bin_centers1 = bin_borders1[:-1] + np.diff(bin_borders1) / 2
bin_centers2 = bin_borders2[:-1] + np.diff(bin_borders2) / 2
popt1, _ = curve_fit(gaussian, bin_centers1, bin_heights1, p0=[1., mean1, sigma1])
popt2, _ = curve_fit(gaussian, bin_centers2, bin_heights2, p0=[1., mean2, sigma2],  maxfev=500000)
print("ra fitted para: %s" % popt1)
print("dec fitted para: %s" % popt2)

x_interval_for_fit1 = np.linspace(-0.75, 0.75, len(samples1))
x_interval_for_fit2 = np.linspace(-0.75, 0.75, len(samples2))
ax1.plot(x_interval_for_fit1, gaussian(x_interval_for_fit1, *popt1), '--r', linewidth=2, label=r'gaussian fit hist $\Delta_{RA}$')
ax2.plot(x_interval_for_fit2, gaussian(x_interval_for_fit2, *popt2), '--b', linewidth=2, label=r'gaussian fit hist $\Delta_{DEC}$')
x_data1 = x_interval_for_fit1
coeff1 = popt1
coeff2 = popt2

#plt.hist((match1['centre_ra']-match1['RA_2'])*3600, bins=50, facecolor='#1F77B4', align='mid', histtype='step', edgecolor='#1F77B4', linewidth=0.2, linestyle='-',alpha=0.5, label=r'$\Delta\alpha$')
#plt.hist((match1['centre_dec']-match1['DEC_2'])*3600, bins=50,   facecolor='#FF7F0E', align='mid', histtype='step', edgecolor='#FF7F0E', linewidth=0.2, linestyle='-',alpha=0.5, label=r'$\Delta\delta$')

#ax1.get_shared_x_axes().join(ax1, ax2)

ax1.set_xticklabels([])
ax2.set_xlabel(r'Separation (asec)', fontsize=12)
ax1.tick_params(labelsize=12)
ax2.tick_params(labelsize=12)
#plt.yscale('log')
plt.xlim(-0.75,0.75)
ax1.set_ylabel('Number of sources', fontsize=12)
ax2.set_ylabel('Number of sources', fontsize=12)
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(which='both', width=2)
ax1.tick_params(which='major', length=7)
ax1.tick_params(which='minor', length=4, color='k')
ax2.tick_params(which='both', width=2)
ax2.tick_params(which='major', length=7)
ax2.tick_params(which='minor', length=4, color='k')


ax1.tick_params(axis="x", direction="in")
ax1.tick_params(axis="y", direction="in")
ax1.tick_params(which="minor", axis="x", direction="in")
ax1.tick_params(which="minor", axis="y", direction="in")
ax1.legend(loc='upper left', fontsize=10)

ax2.tick_params(axis="x", direction="in")
ax2.tick_params(axis="y", direction="in")
ax2.tick_params(which="minor", axis="x", direction="in")
ax2.tick_params(which="minor", axis="y", direction="in")
ax2.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.savefig('ra_dec_separation.png', dpi=600,format="png",bbox_inches = 'tight')

import scipy
from scipy.optimize import curve_fit
import pandas as pd
from scipy.optimize import leastsq
def chi_squared(a, t1, t2):
    global Time
    global Y_tilde
    global Y_fn
    x = len(Time)
    chi_sq = 0
    y = [a*exp(-Time[i]/t1) + (10000-a)*exp(-Time[i]/t2) for i in range(x)]
    Y_fn = y
    #print "y Y_tilde chi_sq"
    for i in range(x):
        term = (y[i] - Y_tilde[i])*(y[i] - Y_tilde[i])/Y_tilde[i]
        chi_sq += term
        
        #print Time[i], y[i], Y_tilde[i], chi_sq

    return 10**chi_sq
def cauchy(x, m, s):
    return ((1/(np.pi*s))*(s**2/((x-m)**2+s**2)))
def gaussian(x, mean, amplitude, standard_deviation):
    return (amplitude * np.exp( - (x - mean)**2 / (2*standard_deviation ** 2)))
def fit_function(x, A, beta, B, mu, sigma):
    return 10**(A * np.exp(-x/beta) + B * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)))

#Skew-Laplace
def skewLaplace(x,a,b,mu):
    """Return the value of the Skew-Laplace PDF"""
    if x <= mu:
        return ((1/(a+b))*np.exp((x-mu)/a))
        #return 10**(a*b/(a+b)*np.exp(-a*(mu-x)))
    else:
        return ((1/(a+b))*np.exp((mu-x)/b))
        #return 10**(a*b/(a+b)*np.exp(-b*(x-mu)))
    #return 10**np.piecewise(x, [x < mu, x >= mu], [lambda x:(a*b/(a+b)*np.exp(-a*mu-x)), \
           #lambda x:(a*b/(a+b)*np.exp(-b*(x-mu)))])
def calc_chisquare(y_data, model, y_sigma=None, verbose=False):
    y_fitdata = model(x_data, *coeff)
    residuals = y_data - y_fitdata
    if type(y_sigma) == np.ndarray:
        chisquare = (pow(residuals, 2.)/pow(y_sigma,2.)).sum()
    else:
        chisquare = pow(residuals, 2.).sum()
    dof = len(y_data) - len(coeff)
    chisquare_nu = chisquare/dof
    #CDF is defined as the area under the right hand tail (from some value of y to infinity) of the Chi square probability density function.
    cdf = scipy.special.chdtrc(dof,chisquare)
    #estimating root-mean-squared errors i.e. rms of residuals:
    #RMSE is the measure of scatter in the data from the about the model. It should be small for better fit.
    RMSE = np.sqrt(np.mean(np.square(residuals)))
    #estimating R-squared (i.e. Coefficient of Determination) parameter:
    Rsquared = 1.0 - (np.var(residuals) / np.var(y_data))
    #The value of R^2 lies between 0 and 1. The zero value means no correlation between data and model (bad fit), while 1 value means a perfect fit of model to the data.For better fit, it should be close to unity.
    if verbose:
        print("chisquare/dof = {0:.2f} / {1:d} = {2:.2f}".format(chisquare, dof, chisquare_nu))
        #print("\nCDF = {:10.5f}%".format(100.*cdf))
        print('RMSE: {0:.2f}'.format(RMSE))
        print('R-squared: {0:.2f}'.format(Rsquared))
    if cdf < 0.05 :
        print("\nNOTE: This does not appear to be a great fit, so the")
        print("      parameter uncertainties may underestimated.")
    elif cdf > 0.95 :
        print("\nNOTE: This fit seems better than expected, so the")
        print("      data uncertainties may have been overestimated.")
    return chisquare_nu
#vectorize so you can use func with array
skewLaplace = np.vectorize(skewLaplace)
from matplotlib import rcParams
config = {
     "font.family":'Times New Roman',  # 
#    "font.size": 18,
#     "mathtext.fontset":'stix',
}
#rcParams.update(config)
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

plt.figure(6)
samples = (match['peak_flux']*1000.0)/match['FPEAK']
samples_1 = samples[samples <=2]
samples_new = samples_1[samples_1 >=0]
mean =np.mean(samples_new)
sigma = np.std(samples_new)

bin_heights, bin_borders, _ = plt.hist(samples_new, bins = 40, histtype='step', color='b',linewidth=2, label='histogram')
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., mean, sigma])
print("flux fitted para: %s" % popt)
x_interval_for_fit = np.linspace(0, 2, len(samples_new))
plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), '--r', linewidth=2, label='fit')
x_data = x_interval_for_fit 
coeff = popt
ax=plt.gca()
try:
    y_errors
    chisquare_nu = calc_chisquare(samples_new, gaussian, y_sigma=y_errors, verbose=True)
except NameError:
    chisquare_nu = calc_chisquare(samples_new, gaussian, y_sigma=None, verbose=True)
#calc_chisquare(samples_new, gaussian(x_interval_for_fit, *popt))
plt.xlim((0,2))
#plt.ylim((0,14000))
ax.set_xlabel(r'$S_{Hetu}/S_{FIRST\_{14dec17}}$', fontdict={'family' : 'Times New Roman', 'size'   : 18})
ax.tick_params(labelsize=16)
ax.set_ylabel('Number of sources', fontsize=18)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', width=2)
ax.tick_params(which='major', length=7)
ax.tick_params(which='minor', length=4, color='k')
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
ax.tick_params(which="minor", axis="x", direction="in")
ax.tick_params(which="minor", axis="y", direction="in")
plt.tight_layout()
#plt.rcParams["font.family"] = "Times New Roman"
#import matplotlib as mpl; print(mpl.font_manager.get_cachedir())
plt.savefig('match_cs_hist_flux.png')

# In[15]:


#coefficients_flux,covariance_flux=np.polyfit(match1['a'],match1['major'],deg=1,full=False,cov=True)
#m_flux=coefficients_flux[0]
#c_flux=coefficients_flux[1]
#coefficients_err_flux=np.sqrt(np.diag(covariance_flux))
#m_err_flux=coefficients_err_flux[0]
#c_err_flux=coefficients_err_flux[1]
#
#
#print(m_flux)
#print(m_err_flux)
#
#z_flux=np.polyfit(match['a'],match['major'],1)
#p_flux= np.poly1d(z_flux)
#
#
## In[16]:
#
#
#fig = plt.figure(figsize=(6,6))
#ax1 = fig.add_subplot(111)
#x=np.linspace(0,300,300)
#fig.subplots_adjust(top=0.95,bottom=0.1,left=0.14,right=0.965)
#plt.plot(match1['a'],match1['major'],'o',c='tab:blue')
#
#
#
#plt.plot(x,x,'r--')
#plt.plot(x,p_flux(x),'--',color='tab:green',alpha=1)
#
#
#plt.tick_params(labelsize=20)
#labels = ax1.get_xticklabels() + ax1.get_yticklabels()
#[label.set_fontname('Times New Roman') for label in labels]
#plt.tick_params(which='major', length=8,direction='in')
#plt.tick_params(which='minor', length=4,direction='in')
#ax1.tick_params(axis='both', which='both', width=1,direction='in')
#ax1.tick_params(axis = 'x', which = 'major', labelsize = 18, pad=4,direction='in')
#ax1.tick_params(axis = 'y', which = 'major', labelsize = 18, pad=4,direction='in')
#
#plt.xlabel(r'$a_{DR1}$(Jy)',fontsize=20,labelpad=0)
#plt.ylabel(r'$a_{HeTu}$(Jy)',fontsize=20,labelpad=0)
#plt.title('Bmajor',fontsize=20)
#ax1.annotate(r'$\frac{a_{HeTu}}{a_{DR1}}=%.2f \pm %.3f$' % (m_flux,m_err_flux) ,xy=(0.2,0.85),xycoords='figure fraction',color='k',fontsize=20)
#
## plt.xlim(-0.5,11)
#plt.ylim(-0.5,2000)
#
#xminors = matplotlib.ticker.AutoMinorLocator(5)
#yminors = matplotlib.ticker.AutoMinorLocator(5)
#ax1.xaxis.set_minor_locator(xminors)
#ax1.yaxis.set_minor_locator(yminors)
#
#plt.savefig('../image/matched_major.png', format="png",dpi=600,bbox_inches = 'tight')
#
#
## In[17]:
#
#
#coefficients_flux,covariance_flux=np.polyfit(match1['b'],match1['minor'],deg=1,full=False,cov=True)
#m_flux=coefficients_flux[0]
#c_flux=coefficients_flux[1]
#coefficients_err_flux=np.sqrt(np.diag(covariance_flux))
#m_err_flux=coefficients_err_flux[0]
#c_err_flux=coefficients_err_flux[1]
#
#
#print(m_flux)
#print(m_err_flux)
#
#z_flux=np.polyfit(match['b'],match['minor'],1)
#p_flux= np.poly1d(z_flux)
#
#
## In[18]:
#
#
#fig = plt.figure(figsize=(6,6))
#ax1 = fig.add_subplot(111)
#x=np.linspace(0,200,300)
#fig.subplots_adjust(top=0.95,bottom=0.1,left=0.14,right=0.965)
#plt.plot(match1['b'],match1['minor'],'o',c='tab:blue')
#
#
#
#plt.plot(x,x,'r--')
#plt.plot(x,p_flux(x),'--',color='tab:green',alpha=1)
#
#
#plt.tick_params(labelsize=20)
#labels = ax1.get_xticklabels() + ax1.get_yticklabels()
#[label.set_fontname('Times New Roman') for label in labels]
#plt.tick_params(which='major', length=8,direction='in')
#plt.tick_params(which='minor', length=4,direction='in')
#ax1.tick_params(axis='both', which='both', width=1,direction='in')
#ax1.tick_params(axis = 'x', which = 'major', labelsize = 18, pad=4,direction='in')
#ax1.tick_params(axis = 'y', which = 'major', labelsize = 18, pad=4,direction='in')
#
#plt.xlabel(r'$b_{DR1}$(Jy)',fontsize=20,labelpad=0)
#plt.ylabel(r'$b_{HeTu}$(Jy)',fontsize=20,labelpad=0)
#plt.title('Bminor',fontsize=20)
#ax1.annotate(r'$\frac{a_{HeTu}}{a_{DR1}}=%.2f \pm %.3f$' % (m_flux,m_err_flux) ,xy=(0.2,0.85),xycoords='figure fraction',color='k',fontsize=20)
#
## plt.xlim(-0.5,11)
#plt.ylim(-0.5,500)
#
#xminors = matplotlib.ticker.AutoMinorLocator(5)
#yminors = matplotlib.ticker.AutoMinorLocator(5)
#ax1.xaxis.set_minor_locator(xminors)
#ax1.yaxis.set_minor_locator(yminors)
#
#plt.savefig('../image/matched_minor.png', format="pdf",dpi=600,bbox_inches = 'tight')
#
#
## In[17]:
#
#
#hetu1=hetu[hetu['peak_flux']<1]
#DR2=DR1[DR1['peak_flux_170_231MHz']<1]
#plt.hist(DR2['peak_flux_170_231MHz'],bins=1000,facecolor='tab:blue',edgecolor='tab:gray',alpha=0.8,label='DR1 200_231MHz',stacked=True)
#plt.hist(hetu1['peak_flux'],bins=1000,facecolor='tab:orange',edgecolor='tab:gray',alpha=0.8,label='HeTu 170_231MHz',stacked=True)
## hetu['peak_flux'].plot.hist(bins=200)
#plt.title('peak flux',fontsize=20)
#plt.xlim(0,0.2)
#plt.legend()
#plt.savefig('../image/peak_flux_hist.pdf', format="pdf",dpi=600,bbox_inches = 'tight')
#
#
## In[18]:
#
#
#non_DR2=non_DR1[non_DR1['peak_flux_170_231MHz']<1]
#
#plt.hist(DR2['peak_flux_170_231MHz'],bins=1000,facecolor='tab:blue',edgecolor='tab:gray',alpha=0.8,label='DR1 200_231MHz',stacked=True)
#plt.hist(non_DR2['peak_flux_170_231MHz'],bins=1000,facecolor='tab:orange',edgecolor='tab:gray',alpha=0.8,label='DR1 not matched',stacked=True)
#
#plt.title('peak flux',fontsize=20)
#t1=hetu['label'].value_counts()
#t2=non_hetu['label'].value_counts()
#
#
## In[86]:
#
#
#plt.axes(yscale='log')
#plt.hist(hetu['label'],facecolor='tab:blue',edgecolor='tab:gray',alpha=0.8,label='HeTu all',stacked=True)
#plt.hist(non_hetu['label'],facecolor='tab:orange',edgecolor='tab:gray',alpha=0.8,label='HeTu not matched',stacked=True)
#plt.title('label',fontsize=20)
#plt.legend()
#plt.savefig('../image/hetu_matched_label_hist.png', format="pdf",dpi=600,bbox_inches = 'tight')
#
#
## In[87]:
#
#
#
#plt.hist(non_hetu1['peak_flux'],bins=1000,facecolor='tab:orange',edgecolor='tab:gray',alpha=0.8,label='HeTu not matched',stacked=True)
#plt.hist(non_DR2['peak_flux_170_231MHz'],bins=1000,facecolor='tab:blue',edgecolor='tab:gray',alpha=0.8,label='DR1 not matched',stacked=True)
## plt.axes(yscale='log')
#
## hetu['peak_flux'].plot.hist(bins=200)
#plt.title('peak flux',fontsize=20)
#plt.xlim(0,0.2)
#plt.legend()
#plt.savefig('../image/hetu_matched_hist_1.pdf', format="pdf",dpi=600,bbox_inches = 'tight')
#
#
## In[88]:
#
#
#plt.hist(hetu['centre_ra'],bins=100,facecolor='tab:olive',alpha=0.5,label='HeTu all',stacked=True)
#plt.hist(DR1['ra_170_231MHz'],bins=100,facecolor='tab:pink',alpha=0.5,label='DR1 all',stacked=True)
#
#plt.hist(non_hetu['centre_ra'],bins=100,facecolor='tab:orange',alpha=0.5,label='HeTu not matched',stacked=True)
#plt.hist(non_DR1['ra_170_231MHz'],bins=100,facecolor='tab:blue',alpha=0.5,label='DR1 not matched',stacked=True)
## plt.axes(yscale='log')
#
## hetu['peak_flux'].plot.hist(bins=200)
#plt.title('RA',fontsize=20)
## plt.xlim(0,0.2)
#plt.legend()
#plt.savefig('../image/RA_hist.pdf', format="pdf",dpi=600,bbox_inches = 'tight')
#
#
## In[89]:
#
#
#
#plt.hist(DR1['dec_170_231MHz'],bins=100,facecolor='tab:pink',alpha=0.5,label='DR1 all',stacked=True)
#plt.hist(hetu['centre_dec'],bins=100,facecolor='tab:olive',alpha=0.5,label='HeTu all',stacked=True)
#plt.hist(non_hetu['centre_dec'],bins=100,facecolor='tab:orange',alpha=0.5,label='HeTu not matched',stacked=True)
#plt.hist(non_DR1['dec_170_231MHz'],bins=100,facecolor='tab:blue',alpha=0.5,label='DR1 not matched',stacked=True)
## plt.axes(yscale='log')
#
## hetu['peak_flux'].plot.hist(bins=200)
#plt.title('DEC',fontsize=20)
## plt.xlim(0,0.2)
#plt.legend()
#plt.savefig('../image/DEC_hist.pdf', format="pdf",dpi=600,bbox_inches = 'tight')
#

# In[ ]:





# In[ ]:




