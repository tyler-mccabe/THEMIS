import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import io
import math as m

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

path = '/Volumes/Seagate/NASA/Summer2018/THEMIS/'
date = '07_14_2008/'
tplot = 'TPLOT_save_file_THB.tplot'

file = scipy.io.readsav(path+date+tplot,python_dict=True)
dq = file['dq']
dh = dq['dh']

##############
# Indicies of Interest
##############

index1 = 37 ## Magnetic Field Mag
index2 = 29 ## Magnetic Field Vector
index3 = 70 ## Electron Density Burst
index4 = 71 ## Electron Temp Burst
index5 = 85 ## Ion Density Burst
index6 = 86 ## Ion Temp Burst
## Plot fit params later

##############
# Magnetic Field Magnitude Plot
##############
t1 = dh[index1].x[0] - 1215993600
y1 = dh[index1].y[0]
min_index1 = np.asarray([])
max_index1 = np.asarray([])
for i in range(len(y1)-1):
    if m.isnan(y1[i]) == True and m.isnan(y1[i+1]) == False:
        min_index1 = np.append(min_index1,i)
    if m.isnan(y1[i]) == False and m.isnan(y1[i+1]) == True:
        max_index1 = np.append(max_index1,i+1)
# print('min',min_index)
# print('max',max_index)
# for i in range(len(min_index)):
#     min = int(min_index[i])
#     max = int(max_index[i])
#     print(t1[min],t1[max])
##############
# Magnetic Field Vector Plot
##############
t2 = dh[index2].x[0] - 1215993600
y2_x = dh[index2].y[0][0] #GSE Coordinates
y2_y = dh[index2].y[0][1] #GSE Coordinates
y2_z = dh[index2].y[0][2] #GSE Coordinates
min_index2 = np.asarray([])
max_index2 = np.asarray([])
for i in range(len(y2_x)-1):
    if m.isnan(y2_x[i]) == True and m.isnan(y2_x[i+1]) == False:
        min_index2 = np.append(min_index2,i)
    if m.isnan(y2_x[i]) == False and m.isnan(y2_x[i+1]) == True:
        max_index2 = np.append(max_index2,i+1)
##############
# Electron Density
##############
t3 = dh[index3].x[0] - 1215993600
y3 = dh[index3].y[0]
min_index3 = np.asarray([])
max_index3 = np.asarray([])
for i in range(len(y3)-1):
    if m.isnan(y3[i]) == True and m.isnan(y3[i+1]) == False:
        min_index3 = np.append(min_index3,i)
    if m.isnan(y3[i]) == False and m.isnan(y3[i+1]) == True:
        max_index3 = np.append(max_index3,i+1)
##############
# Electron Temperature
##############
t4 = dh[index4].x[0] - 1215993600
y4 = dh[index4].y[0]
min_index4 = np.asarray([])
max_index4 = np.asarray([])
for i in range(len(y4)-1):
    if m.isnan(y4[i]) == True and m.isnan(y4[i+1]) == False:
        min_index4 = np.append(min_index4,i)
    if m.isnan(y4[i]) == False and m.isnan(y4[i+1]) == True:
        max_index4 = np.append(max_index4,i+1)
##############
# Ion Density
##############
t5 = dh[index5].x[0] - 1215993600
y5 = dh[index5].y[0]
min_index5 = np.asarray([])
max_index5 = np.asarray([])
for i in range(len(y5)-1):
    if m.isnan(y5[i]) == True and m.isnan(y5[i+1]) == False:
        min_index5 = np.append(min_index5,i)
    if m.isnan(y5[i]) == False and m.isnan(y5[i+1]) == True:
        max_index5 = np.append(max_index5,i+1)
##############
# Ion Temperature
##############
t6 = dh[index6].x[0] - 1215993600
y6 = dh[index6].y[0]
min_index6 = np.asarray([])
max_index6 = np.asarray([])
for i in range(len(y6)-1):
    if m.isnan(y6[i]) == True and m.isnan(y6[i+1]) == False:
        min_index6 = np.append(min_index6,i)
    if m.isnan(y6[i]) == False and m.isnan(y6[i+1]) == True:
        max_index6 = np.append(max_index6,i+1)
##############
# Interpolate Times
##############        
y1 = np.interp(t5,t1,y1)
y2_x = np.interp(t5,t2,y2_x)
y2_y = np.interp(t5,t2,y2_y)
y2_z = np.interp(t5,t2,y2_z)
y3 = np.interp(t5,t3,y3)
y4 = np.interp(t5,t4,y4)
y6 = np.interp(t5,t6,y6)
##############
# Overall Plot Structure
##############
# names = ['1st','2nd','3rd','4th','5th','6th']
# name = names[i]
for i in range(6):
    fig = plt.figure(figsize=(5.5,6.5))
    ax1 = fig.add_subplot(6,1,1)
    ax2 = fig.add_subplot(6,1,2)
    ax3 = fig.add_subplot(6,1,3)
    ax4 = fig.add_subplot(6,1,4)
    ax5 = fig.add_subplot(6,1,5)
    ax6 = fig.add_subplot(6,1,6)
    
    start = int(min_index1[i])
    xmin = t1[start]-3.9
    stop = int(max_index1[i])
    xmax = t1[stop]+3.9
    
    time = np.asarray([])
    for t in range(len(t1)):
        if t1[t] >= xmin and t1[t] <= xmax:
            time = np.append(time,t1[t])
    begin = list(t1).index(np.nanmin(time))  
    end = list(t1).index(np.nanmax(time))     
    ax1.plot(t5,y1,'k',lw=0.5)
    ax1.set_ylabel('$|\mathbf{B}_{o}| (nT)$')
    # plt.setp(ax1.get_xticklabels(), visible=False) #Share x-axis
    ax1.set_xlim(xmin,xmax)
    ax1.set_ylim(np.nanmin(y1[begin:end]), np.nanmax(y1[begin:end]))

    ax2.plot(t5,y2_x,'r',lw=0.5,label='X')
    ax2.plot(t5,y2_y,'g',lw=0.5,label='Y')
    ax2.plot(t5,y2_z,'b',lw=0.5, label ='Z')
    # ax2.legend(loc='lower left', frameon=True,prop={'size': 6})
    ax2.set_ylabel('$\mathbf{B}_{o} (nT)$')
    ax2.set_xlim(xmin,xmax)
    mins = np.asarray([np.nanmin(y2_x[begin:end]),np.nanmin(y2_y[begin:end]),np.nanmin(y2_z[begin:end])])
    maxs = np.asarray([np.nanmax(y2_x[begin:end]),np.nanmax(y2_y[begin:end]),np.nanmax(y2_z[begin:end])])
    min_y = np.nanmin(mins)
    max_y = np.nanmax(maxs)
    ax2.set_ylim(min_y, max_y)
    
    ax3.plot(t5,y3,'k',lw=0.5)
    ax3.set_ylabel('$n_{e}\ (cm^{-3})$')
    ax3.set_xlim(xmin,xmax)
    ax3.set_ylim(np.nanmin(y3[begin:end]), np.nanmax(y3[begin:end]))
    
    ax4.plot(t5,y4,'k',lw=0.5)
    ax4.set_ylabel('$T_{e}$ (eV)')
    ax4.set_xlim(xmin,xmax)
    ax4.set_ylim(np.nanmin(y4[begin:end]), np.nanmax(y4[begin:end]))
    
    ax5.plot(t5,y5,'k',lw=0.5)
    ax5.set_ylabel('$n_{i}\ (cm^{-3})$')
    ax5.set_xlim(xmin,xmax)
    ax5.set_ylim(np.nanmin(y5[begin:end]), np.nanmax(y5[begin:end]))
    
    ax6.plot(t5,y6,'k',lw=0.5)
    ax6.set_ylabel('$T_{i}$ (eV)')
    ax6.set_xlabel('Time From Start of Date (s)')
    ax6.set_xlim(xmin,xmax)
    ax6.set_ylim(np.nanmin(y6[begin:end]), np.nanmax(y6[begin:end]))
    
    # fig.suptitle('Themis C: 2008-07-14 - %s Area of Interest'%name)
    fig.suptitle('Themis B: 2008-07-14')
    fig.subplots_adjust(left=0.15, bottom=0.1, right=0.97, top=0.94, wspace=0.2, hspace=0.38)
    # plt.savefig(path + 'Themis_2008_07_14_%s.png'%name)
    # plt.clf()
    plt.show()