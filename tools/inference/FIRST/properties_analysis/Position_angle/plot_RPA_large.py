import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

result_file = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FRIIRGcat_final.csv'
hetu_csv = pd.read_csv(result_file)
hetu_csv = hetu_csv[hetu_csv['LAS']>=80]
RPAs = hetu_csv['RPA'].values
ras = hetu_csv['RA'].values
decs = hetu_csv['DEC'].values

RPAs_N = []
RPAs_S = []
for mm in range(len(RPAs)):
    #ra: 6h-18h, 90d-270d
    if ras[mm] >= 90 and ras[mm] <= 270:
       RPAs_N.append(RPAs[mm])
    else:
       RPAs_S.append(RPAs[mm])

print(len(RPAs_N))
fig = plt.figure()
ax1 = plt.gca()
#ax1.hist(RPAs, bins=20, facecolor='k', align='mid', histtype='step', edgecolor='k', linewidth=2, linestyle='-')
#ax1.hist(RPAs_N, bins=20, facecolor='r', align='mid', histtype='step', edgecolor='r', linewidth=2, linestyle='-')
ax1.hist(RPAs_N, bins=20, align='mid', histtype='step', linewidth=2, linestyle='-')
plt.tick_params(labelsize=16)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.tick_params(which='major', length=8,direction='in')
plt.tick_params(which='minor', length=4,direction='in')
ax1.tick_params(axis='both', which='both', width=1,direction='in')
ax1.tick_params(axis = 'x', which = 'major', labelsize = 14, pad=4,direction='in')
ax1.tick_params(axis = 'y', which = 'major', labelsize = 14, pad=4,direction='in')
ax1.set_ylabel('Number', fontsize=20) 
ax1.set_xlabel('Position angle (degrees)', fontsize=20)
plt.tight_layout()
plt.savefig('RPAs_new.pdf')

N=20
theta=np.linspace(0,np.pi,N+1)
plt.figure(2)
#bins = np.linspace(0, np.pi, N + 1)
n, bins, _ = plt.hist(RPAs_N, bins=N, align='mid', histtype='step', linewidth=2, linestyle='-')
plt.clf()
ax=plt.subplot(111,projection='polar')
print(bins[1:N])
width = np.pi / N
bars=ax.bar(bins[:N]/180.0*np.pi,n,width=width,bottom=0.0)
#print(theta)
#ax.set_xticks([0, 30, 60, 90, 120, 150 , 180])
#ax.set_rticks([0.5, 1, 1.5, 2])
ax.set_yticklabels([0,2,4,6,8,10,12,14])
ax.set_thetamin(0)
ax.set_thetamax(180)
label_position=ax.get_rlabel_position()
print(label_position)
print(ax.get_rmax())
ax.text(np.radians(-25),ax.get_rmax()/2.,r'Number ($\times 10$)',
        rotation=0.0,ha='center',va='center', fontsize=12)
#ax.set_xlabel(r'Number ()', fontsize=16, labelpad=-50)
ax.set_title('Radio position angle', fontsize=12, pad=-40)
#ax.tick_params(labelsize=14)
#plt.tight_layout()
#plt.show()
ax.margins(0)
plt.savefig('RPAs_polar_new.pdf')
