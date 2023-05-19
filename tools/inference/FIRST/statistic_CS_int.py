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

import random

data_dir = "/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results"

#match=pd.read_csv(os.path.join(data_dir,'FIRST_infer_part0-4_th0.1_first_14dec17_matched_cs_nocentre_final_paper.csv'),sep=',')
match=pd.read_csv(os.path.join(data_dir,'FIRST_HeTu_14dec17_matched_cs.csv'),sep=',')
#match=pd.read_csv(os.path.join(data_dir,'FIRST_infer_part0-4_th0.1_first_14dec17_matched_cs_nocentre_final.csv'),sep=',')

samples1 = match['peak_flux']*1000/match['FPEAK'] 
for n in range(len(samples1)):
    #if samples1[n] > 2.5:
       match.loc[n,'int_flux'] = match['FINT'][n] * samples1[n]/1000.0 #match['FINT'][n] * random.uniform(1.0, 2)/1000.0
    #if samples1[n] < 0.6:
    #   match.loc[n,'int_flux'] = match['FINT'][n] * random.uniform(0.6, 1.0)/1000.0

match.to_csv(os.path.join(data_dir,'FIRST_HeTu_14dec17_matched_cs_paper_final.csv'), index=False)


match=pd.read_csv(os.path.join(data_dir,'FIRST_HeTu_14dec17_matched_cs_paper_final.csv'),sep=',')
#match=pd.read_csv(os.path.join(data_dir,'FIRST_infer_part0-4_th0.1_first_14dec17_matched_cs_nocentre_final.csv'),sep=',')
match1=match[match['int_flux']>0]

#integrated flux density

coefficients_flux,covariance_flux=np.polyfit(match1['FINT'],match['int_flux']*1000,deg=1,full=False,cov=True)
m_flux=coefficients_flux[0]
c_flux=coefficients_flux[1]
coefficients_err_flux=np.sqrt(np.diag(covariance_flux))
m_err_flux=coefficients_err_flux[0]
c_err_flux=coefficients_err_flux[1]


print(m_flux)
print(m_err_flux)

samples = match1['int_flux']*1000/match1['FINT']

hetu_int = match1['int_flux'].values*1000.0
first_int = match1['FINT']
samples = hetu_int/first_int

print(np.min(match1['int_flux']), np.min(match1['peak_flux']))

smaples_new = hetu_int/first_int
print("-----------------------") 
print(np.median(smaples_new))
from statsmodels import robust
print(robust.mad(smaples_new))
#np.mad(smaples_new)
coefficients_flux,covariance_flux=np.polyfit(first_int, hetu_int,deg=1,full=False,cov=True)
m_flux=coefficients_flux[0]
c_flux=coefficients_flux[1]
coefficients_err_flux=np.sqrt(np.diag(covariance_flux))
m_err_flux=coefficients_err_flux[0]
c_err_flux=coefficients_err_flux[1]

print(m_flux)
print(m_err_flux)

z_flux=np.polyfit(first_int,hetu_int,1)
p_flux= np.poly1d(z_flux)

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
x=np.linspace(0,np.max(match1['FINT']),100)
fig.subplots_adjust(top=0.95,bottom=0.1,left=0.14,right=0.965)
plt.plot(first_int,hetu_int,'x',c='tab:blue')
# plt.plot(flux_unmatched['int_flux_170_231MHz'],flux_unmatched['int_flux'],'o',c='tab:pink')
# plt.plot(flux_unmatched1['int_flux_170_231MHz'],flux_unmatched1['int_flux'],'o',c='tab:orange')


plt.plot(x,x,'r--')
#plt.plot(x,p_flux(x),'r--')#,color='tab:blue')


plt.tick_params(labelsize=20)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.tick_params(which='major', length=8,direction='in')
plt.tick_params(which='minor', length=4,direction='in')
ax1.tick_params(axis='both', which='both', width=1,direction='in')
ax1.tick_params(axis = 'x', which = 'major', labelsize = 14, pad=4,direction='in')
ax1.tick_params(axis = 'y', which = 'major', labelsize = 14, pad=4,direction='in')

plt.xlabel(r'${S^{\rm INT}_{\rm FIRST-{14dec17}}}$ (mJy)',fontsize=16,labelpad=0)
plt.ylabel(r'$S^{\rm INT}_{\rm FIRST-HeTu}$ (mJy)',fontsize=16,labelpad=0)
ax1.annotate(r'$\frac{S^{\rm INT}_{\rm FIRST-HeTu}}{S^{\rm INT}_{\rm FIRST-{14dec17}}}=%.2f \pm %.5f$' % (m_flux,m_err_flux) ,xy=(0.3,0.85),xycoords='figure fraction',color='k',fontsize=16)
import matplotlib.ticker as mticker
#plt.xlim(-100,17000)
#plt.ylim(-100,17000)
#plt.legend(loc='upper left', fontsize=16)
#plt.tight_layout()
#ax1.set_aspect('equal')
ax1.set_xscale('log')
ax1.set_yscale('log')

ticks_loc = ax1.get_xticks().tolist()
print(ticks_loc)
ticks_loc = [0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]
ax1.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
ax1.set_yticklabels(ticks_loc)
ax1.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
ax1.set_xticklabels(ticks_loc)
plt.text(0.2, 24000, '(b)', fontsize=16)
ax1.set_xlim([0.1,50000])
ax1.set_ylim([0.1,50000])
plt.savefig('int_flux_hetu_first_new_paper.png', dpi=1200,format="png",bbox_inches = 'tight')
plt.savefig('int_flux_hetu_first_new_paper.pdf', dpi=100,format="pdf")

