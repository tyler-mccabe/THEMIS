import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import io
import math as m
import datetime

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

themis = 'THC'
# slice_by = 'efw' ## Choose variable to create slices by

path = '/Users/tylermccabe/Documents/NASA/Summer2018/THEMIS/'
date = '09_08_2008/'
tplot = themis + '.tplot'
savedir = 'Plots/' + themis + '/'
plot_title = themis + ': ' + date[:-1]

file = scipy.io.readsav(path+date+tplot,python_dict=True)
dq = file['dq']
dh = dq['dh']

##############
# Indicies of Interest
##############
def get_index(string_handle):
    for i in range(len(dq)):
        string = dq[i][0].decode()
        if string[4:] == string_handle:
            print(string,i)
            index = i
    return index
index1 = get_index('fgh_mag')      ## Magnetic Field Mag 
index2 = get_index('fgh_gse')  ## Magnetic Field Vector
index3 = get_index('peeb_velocity_gse')  ## Electron Velocity
index4 = get_index('peib_velocity_gse')  ## Ion Velocity      
# print('\n')

##############
# Magnetic Field Magnitude
##############
t1 = dh[index1].x[0]
y1 = dh[index1].y[0]

##############
# Magnetic Field Vector
##############
t2 = dh[index2].x[0]
y2_x = dh[index2].y[0][0] #GSE
y2_y = dh[index2].y[0][1]
y2_z = dh[index2].y[0][2]

##############
# Electron Velocity
##############
t3 = dh[index3].x[0]
y3_x = dh[index3].y[0][0] #GSE
y3_y = dh[index3].y[0][1]
y3_z = dh[index3].y[0][2]

##############
# Ion Velocity
##############
t4 = dh[index4].x[0]
y4_x = dh[index4].y[0][0] #GSE
y4_y = dh[index4].y[0][1]
y4_z = dh[index4].y[0][2]

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

for i in range(len(min_index1)):

    print('Plot ',i+1)

    fig = plt.figure(figsize=(6.5,6.5))
    ax1 = fig.add_subplot(4,1,1)
    ax2 = fig.add_subplot(4,1,2,sharex=ax1)
    ax3 = fig.add_subplot(4,1,3,sharex=ax1)
    ax4 = fig.add_subplot(4,1,4,sharex=ax1)
    
    start = int(min_index1[i]) 
    xmin = t1[start]
    stop = int(max_index1[i]) 
    xmax = t1[stop]

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

    try:
        begin, end = get_begin_end_indicies(t3, xmin, xmax)    
    except:
    	ValueError
    ax3.plot(t3,y3_x,'r',lw=0.5,label='X')
    ax3.plot(t3,y3_y,'g',lw=0.5,label='Y')
    ax3.plot(t3,y3_z,'b',lw=0.5, label ='Z')
    ax3.set_ylabel('$\mathbf{v}_{e}$ (km/s)')
    ax3.set_xlim(xmin,xmax)
    mins = np.asarray([np.nanmin(y3_x[begin:end]),np.nanmin(y3_y[begin:end]),np.nanmin(y3_z[begin:end])])
    maxs = np.asarray([np.nanmax(y3_x[begin:end]),np.nanmax(y3_y[begin:end]),np.nanmax(y3_z[begin:end])])
    min_y = np.nanmin(mins)
    max_y = np.nanmax(maxs) * 1.1
    ax3.set_ylim(min_y, max_y) 
    ax3.set_xticklabels(unix_to_utc(ax3.get_xticks()))    
    plt.setp(ax3.get_xticklabels(), visible=False) #Share x-axis

    try:
    	begin, end = get_begin_end_indicies(t4, xmin, xmax)
    except:
        ValueError
    ax4.plot(t4,y4_x,'r',lw=0.5,label='X')
    ax4.plot(t4,y4_y,'g',lw=0.5,label='Y')
    ax4.plot(t4,y4_z,'b',lw=0.5, label ='Z')
    ax4.set_ylabel('$\mathbf{v}_{i}$ (km/s)')
    ax4.set_xlim(xmin,xmax)
    mins = np.asarray([np.nanmin(y4_x[begin:end]),np.nanmin(y4_y[begin:end]),np.nanmin(y4_z[begin:end])])
    maxs = np.asarray([np.nanmax(y4_x[begin:end]),np.nanmax(y4_y[begin:end]),np.nanmax(y4_z[begin:end])])
    min_y = np.nanmin(mins)
    max_y = np.nanmax(maxs) * 1.1
    ax4.set_ylim(min_y, max_y) 
    ax4.set_xticklabels(unix_to_utc(ax4.get_xticks()))    
    plt.setp(ax4.get_xticklabels(), rotation=30, horizontalalignment='right')

    fig.suptitle(plot_title)
    fig.subplots_adjust(left=0.20, bottom=0.1, right=0.97, top=0.94, wspace=0.2, hspace=0.18) #hspace = 0.38 with labels
    plt.savefig(path + date + savedir + 'Velocity_figure_%d.png'%(i))
    # plt.show() 
    plt.clf()
    plt.close()