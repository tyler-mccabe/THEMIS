import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import io
import math as m
import datetime

## This code reads in a .tplot file and plots 6 parameters in 6 subplots
## Must change date, tplot, indicies of interest, save directory and plot Title


'''
PROGRESS: AT THIS POINT, THE AUGUST 19TH THEMIS FILE DOES NOT HAVE 
DATA BREAKS SEPARATED BY NANS. NEED TO UPDATE CODE SO THAT IF 
MIN_INDEX1/MAX_INDEX 1 IS EMPTY, THEN SEPARATE BASED ON A JUMP 
FORWARD IN TIME. ie: if t[i+1] - t[i] > 100s
'''

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

path = '/Volumes/Seagate/NASA/Summer2018/THEMIS/'
date = '08_19_2008/'
tplot = 'TPLOT_save_file_THC.tplot'
savedir = 'Plots/THC/'
plot_title = 'Themis C: ' + date[:-1]

file = scipy.io.readsav(path+date+tplot,python_dict=True)
dq = file['dq']
dh = dq['dh']

##############
# Indicies of Interest
##############
## Remember, these are one less than the IDL index
index1 = 41 ## Magnetic Field Mag      (fgh_mag)
index2 = 29 ## Magnetic Field Vector   (fgh_gse)
index3 = 74 ## Electron Density Burst  (peeb_density)
index4 = 75 ## Electron Temp Burst     (peib_avgtemp)
index5 = 89 ## Ion Density Burst       (peib_density)
index6 = 90 ## Ion Temp Burst          (peib_avgtemp)
## Plot fit params later

##############
# Magnetic Field Magnitude Plot
##############
t1 = dh[index1].x[0]# - 1215993600
y1 = dh[index1].y[0]

##############
# Magnetic Field Vectors
##############
t2 = dh[index2].x[0]# - 1215993600
y2_x = dh[index2].y[0][0] #GSE Coordinates
y2_y = dh[index2].y[0][1] #GSE Coordinates
y2_z = dh[index2].y[0][2] #GSE Coordinates
##############
# Electron Density
##############
t3 = dh[index3].x[0]# - 1215993600
y3 = dh[index3].y[0]

##############
# Electron Temperature
##############
t4 = dh[index4].x[0]# - 1215993600
y4 = dh[index4].y[0]

##############
# Ion Density
##############
t5 = dh[index5].x[0]# - 1215993600
y5 = dh[index5].y[0]

##############
# Ion Temperature
##############
t6 = dh[index6].x[0]# - 1215993600
y6 = dh[index6].y[0]

##############
# Get slices to plot
############## 
min_index1 = np.asarray([])
max_index1 = np.asarray([])
for i in range(len(y1)-1):
    if m.isnan(y1[i]) == True and m.isnan(y1[i+1]) == False:
        min_index1 = np.append(min_index1,i)
    if m.isnan(y1[i]) == False and m.isnan(y1[i+1]) == True:
        max_index1 = np.append(max_index1,i+1)
if len(min_index1) > len(max_index1):
    max_index1 = np.append(max_index1,len(y1)-1)
print('min 1 ', min_index1)
print('max 1 ', max_index1)

##############
# Overall Plot Structure
##############

def unix_to_utc(unix_time_array):
    '''Takes array of tick labels in unix time
    and converts them into readable utc
    Make sure to import datetime'''
    result = [None]*(len(unix_time_array))
    for i in range(len(unix_time_array)):
        result[i] = datetime.datetime.utcfromtimestamp(unix_time_array[i]
        ).strftime('%H:%M:%S')
    print("result type:",type(result))
    return result

def get_begin_end_indicies(time_array,start_time,stop_time):
    '''Get slice indicies for each of the variables to plot'''
    time = np.asarray([])
    for t in range(len(time_array)):
        if time_array[t] >= start_time and time_array[t] <= stop_time:
            time = np.append(time,time_array[t])
    begin_index = list(time_array).index(np.nanmin(time))
    end_index = list(time_array).index(np.nanmax(time))
    return begin_index,end_index

for i in range(6):

    print('Plot ',i+1)

    fig = plt.figure(figsize=(5.5,6.5))
    ax1 = fig.add_subplot(6,1,1)
    ax2 = fig.add_subplot(6,1,2)
    ax3 = fig.add_subplot(6,1,3)
    ax4 = fig.add_subplot(6,1,4)
    ax5 = fig.add_subplot(6,1,5)
    ax6 = fig.add_subplot(6,1,6)
    
    start = int(min_index1[i]) #+ 1
    xmin = t1[start]# - 10 # 3.9
    stop = int(max_index1[i]) #- 1
    xmax = t1[stop]# + 10 # 3.9

    begin, end = get_begin_end_indicies(t1, xmin, xmax)
    ax1.plot(t1,y1,'k',lw=0.5)
    ax1.set_ylabel('$|\mathbf{B}_{o}| (nT)$')
    ax1.set_xlim(xmin,xmax)
    ax1.set_ylim(np.nanmin(y1[begin:end])*0.8, np.nanmax(y1[begin:end])*1.1)
    ax1.set_xticklabels(unix_to_utc(ax1.get_xticks()))    
    plt.setp(ax1.get_xticklabels(), visible=False) #Share x-axis

    begin, end = get_begin_end_indicies(t2, xmin, xmax)
    ax2.plot(t2,y2_x,'r',lw=0.5,label='X')
    ax2.plot(t2,y2_y,'g',lw=0.5,label='Y')
    ax2.plot(t2,y2_z,'b',lw=0.5, label ='Z')
    # ax2.legend(loc='lower left', frameon=True,prop={'size': 6})
    ax2.set_ylabel('$\mathbf{B}_{o} (nT)$')
    ax2.set_xlim(xmin,xmax)
    mins = np.asarray([np.nanmin(y2_x[begin:end]),np.nanmin(y2_y[begin:end]),np.nanmin(y2_z[begin:end])])
    maxs = np.asarray([np.nanmax(y2_x[begin:end]),np.nanmax(y2_y[begin:end]),np.nanmax(y2_z[begin:end])])
    min_y = np.nanmin(mins)
    max_y = np.nanmax(maxs) * 1.1
    ax2.set_ylim(min_y, max_y) 
    ax2.set_xticklabels(unix_to_utc(ax2.get_xticks()))    
    plt.setp(ax2.get_xticklabels(), visible=False) #Share x-axis

    begin, end = get_begin_end_indicies(t3, xmin, xmax)
    ax3.plot(t3,y3,'k',lw=0.5)
    ax3.set_ylabel('$n_{e}\ (cm^{-3})$')
    ax3.set_xlim(xmin,xmax)
    ax3.set_ylim(np.nanmin(y3[begin:end])*0.8, np.nanmax(y3[begin:end])*1.1)
    ax3.set_xticklabels(unix_to_utc(ax3.get_xticks()))    
    plt.setp(ax3.get_xticklabels(), visible=False) #Share x-axis

    begin, end = get_begin_end_indicies(t4, xmin, xmax)
    # ax4.semilogy(t5,y4,'k',lw=0.5)    
    ax4.plot(t4,y4,'k',lw=0.5)
    ax4.set_ylabel('$T_{e}$ (eV)')
    ax4.set_xlim(xmin,xmax)
    ax4.set_ylim(np.nanmin(y4[begin:end])*0.8, np.nanmax(y4[begin:end])*1.1)
    diff = np.nanmax(y4[begin:end]) - np.nanmin(y4[begin:end])
    if diff >= 1000:
        ax4.cla()
        ax4.semilogy(t4,y4,'k',lw=0.5)
        ax4.set_ylabel('$T_{i}$ (eV)')
        ax4.set_xlabel('Time From Start of Date (s)')
        ax4.set_xlim(xmin,xmax)
        ax4.set_ylim(np.nanmin(y4[begin:end])*0.8, np.nanmax(y4[begin:end])*1.1)
    else:
        pass
    ax4.set_xticklabels(unix_to_utc(ax4.get_xticks()))    
    plt.setp(ax4.get_xticklabels(), visible=False) #Share x-axis

    begin, end = get_begin_end_indicies(t5, xmin, xmax)
    ax5.plot(t5,y5,'k',lw=0.5)
    ax5.set_ylabel('$n_{i}\ (cm^{-3})$')
    ax5.set_xlim(xmin,xmax)
    ax5.set_ylim(np.nanmin(y5[begin:end])*0.8, np.nanmax(y5[begin:end])*1.1)
    ax5.set_xticklabels(unix_to_utc(ax5.get_xticks()))
    plt.setp(ax5.get_xticklabels(), visible=False) #Share x-axis

    begin, end = get_begin_end_indicies(t6, xmin, xmax)
    ax6.plot(t6,y6,'k',lw=0.5)
    ax6.set_ylabel('$T_{i}$ (eV)')
    ax6.set_xlabel('Time From Start of Date (s)')
    ax6.set_xlim(xmin,xmax)
    ax6.set_ylim(np.nanmin(y6[begin:end])*0.8, np.nanmax(y6[begin:end])*1.1)
    diff = np.nanmax(y6[begin:end]) - np.nanmin(y6[begin:end])
    if diff >= 1000:
        ax6.cla()
        ax6.semilogy(t6,y6,'k',lw=0.5)
        ax6.set_ylabel('$T_{i}$ (eV)')
        ax6.set_xlabel('Time From Start of Date (s)')
        ax6.set_xlim(xmin,xmax)
        ax6.set_ylim(np.nanmin(y6[begin:end])*0.8, np.nanmax(y6[begin:end])*1.1)
    else:
        pass
    ax6.set_xticklabels(unix_to_utc(ax6.get_xticks()))

    fig.suptitle(plot_title)
    fig.subplots_adjust(left=0.15, bottom=0.1, right=0.97, top=0.94, wspace=0.2, hspace=0.18) #hspace = 0.38 with labels
    plt.savefig(path + date + savedir + 'figure_%d.png'%(i+1))
    # plt.show() 
    plt.clf()
