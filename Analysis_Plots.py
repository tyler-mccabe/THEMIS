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

path = '/Users/tylermccabe/Documents/NASA/Summer2018/THEMIS/'
date = '07_14_2008/'
tplot = 'TPLOT_save_file_THC_new.tplot'
savedir = 'Plots/THC/'
plot_title = 'Themis C: ' + date[:-1]

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
            print(string)
            index = i
    return index

index1 = get_index('fgh_mag')      ## Magnetic Field Mag 
index2 = get_index('efp_l1_cal_tdas_corrected_rmspikes_dsl')  ## DC-coupled electric field vectors  
index3 = get_index('efw_l1_cal_tdas_corrected_rmspikes_dsl')  ## AC-coupled electric field vectors
index4 = get_index('scp_l1_cal_NoCutoff_dsl') ## DC-coupled magnetic field vectors from the search coil magnetometer data 
index5 = get_index('scw_l1_cal_NoCutoff_dsl') ## AC-coupled magnetic field vectors from the burst search coil magnetometer data    
index6 = get_index('peeb_red_chisq_good') ## Reduced Chi-Squared values         


##############
# Magnetic Field Magnitude Plot
##############
t1 = dh[index1].x[0]
y1 = dh[index1].y[0]

##############
# EFP Plot
##############
t2 = dh[index2].x[0]
y2_x = dh[index2].y[0][0] #DSL Coordinates
y2_y = dh[index2].y[0][1] #DSL Coordinates
y2_z = dh[index2].y[0][2] #DSL Coordinates

##############
# EFW Plot
##############
t3 = dh[index3].x[0]
y3_x = dh[index3].y[0][0] #DSL Coordinates
y3_y = dh[index3].y[0][1] #DSL Coordinates
y3_z = dh[index3].y[0][2] #DSL Coordinates

##############
# SCP Plot
##############
t4 = dh[index4].x[0]
y4_x = dh[index4].y[0][0] #DSL Coordinates
y4_y = dh[index4].y[0][1] #DSL Coordinates
y4_z = dh[index4].y[0][2] #DSL Coordinates

##############
# SCW Plot
##############
t5 = dh[index5].x[0]
y5_x = dh[index5].y[0][0] #DSL Coordinates
y5_y = dh[index5].y[0][1] #DSL Coordinates
y5_z = dh[index5].y[0][2] #DSL Coordinates

##############
# Reduced Chi-Squared plot
##############
t6 = dh[index6].x[0]
y6_x = dh[index6].y[0][0] #Parallel
y6_y = dh[index6].y[0][1] #Orthogonal
y6_z = dh[index6].y[0][2] #Anti-parallel

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

for i in range(len(min_index1)):
    x = int(min_index1[i])
    y = int(max_index1[i])
    print(x,y,t1[x]-1.2191e9,t1[y]-1.2191e9)
if len(min_index1) < 1:
    for i in range(len(t1)-1):
        if (t1[i+1] - t1[i]) >= 200:
            min_index1 = np.append(min_index1,i)
        else:
            pass
elif len(min_index1) != 6:
    min_index1 = np.asarray([0])
    max_index1 = np.asarray([])
    for i in range(len(t1)-1):
        if (t1[i+1]-t1[i]) >= 200:
            min_index1 = np.append(min_index1,i+1)
            max_index1 = np.append(max_index1,i)
    max_index1 = np.append(max_index1,len(t1)-1)
else:
    pass

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

    fig = plt.figure(figsize=(5.5,6.5))
    ax1 = fig.add_subplot(6,1,1)
    ax2 = fig.add_subplot(6,1,2)
    ax3 = fig.add_subplot(6,1,3)
    ax4 = fig.add_subplot(6,1,4)
    ax5 = fig.add_subplot(6,1,5)
    ax6 = fig.add_subplot(6,1,6)
    
    start = int(min_index1[i]) 
    xmin = t1[start]
    stop = int(max_index1[i]) 
    xmax = t1[stop]

    begin, end = get_begin_end_indicies(t1, xmin, xmax)
    ax1.plot(t1,y1,'k',lw=0.1)
    ax1.set_ylabel('$|\mathbf{B}_{o}| (nT)$',fontsize=8)
    ax1.set_xlim(xmin,xmax)
    ax1.set_ylim(np.nanmin(y1[begin:end])*0.8, np.nanmax(y1[begin:end])*1.1)
    ax1.set_xticklabels(unix_to_utc(ax1.get_xticks()))    
    plt.setp(ax1.get_xticklabels(), visible=False) #Share x-axis

    begin, end = get_begin_end_indicies(t2, xmin, xmax)
    ax2.plot(t2,y2_x,'r',lw=0.5,label='X')
    ax2.plot(t2,y2_y,'g',lw=0.5,label='Y')
    ax2.plot(t2,y2_z,'b',lw=0.5, label ='Z')
    ax2.set_ylabel('EFP (mV/m)',fontsize=8)
    mins = np.asarray([np.nanmin(y2_x[begin:end]),np.nanmin(y2_y[begin:end]),np.nanmin(y2_z[begin:end])])
    maxs = np.asarray([np.nanmax(y2_x[begin:end]),np.nanmax(y2_y[begin:end]),np.nanmax(y2_z[begin:end])])
    min_y = np.nanmin(mins)
    max_y = np.nanmax(maxs) * 1.1
    ax2.set_ylim(min_y, max_y) 
    ax2.set_xticklabels(unix_to_utc(ax2.get_xticks()))    
    plt.setp(ax2.get_xticklabels(), visible=False) #Share x-axis
    
    begin, end = get_begin_end_indicies(t3, xmin, xmax)    
    ax3.plot(t3,y3_x,'r',lw=0.1,label='X')
    ax3.plot(t3,y3_y,'g',lw=0.1,label='Y')
    ax3.plot(t3,y3_z,'b',lw=0.1,label='Z')
    ax3.set_ylabel('EFW (mV/m)',fontsize=8)
    mins = np.asarray([np.nanmin(y3_x[begin:end]),np.nanmin(y3_y[begin:end]),np.nanmin(y3_z[begin:end])])
    maxs = np.asarray([np.nanmax(y3_x[begin:end]),np.nanmax(y3_y[begin:end]),np.nanmax(y3_z[begin:end])])
    min_y = np.nanmin(mins)
    max_y = np.nanmax(maxs) * 1.1
    ax3.set_ylim(min_y, max_y) 
    ax3.set_xticklabels(unix_to_utc(ax3.get_xticks()))    
    plt.setp(ax3.get_xticklabels(), visible=False) #Share x-axis

    begin, end = get_begin_end_indicies(t4, xmin, xmax)    
    ax4.plot(t4,y4_x,'r',lw=0.5,label='X')
    ax4.plot(t4,y4_y,'g',lw=0.5,label='Y')
    ax4.plot(t4,y4_z,'b',lw=0.5,label='Z')
    ax4.set_ylabel('SCP (nT)',fontsize=8)
    mins = np.asarray([np.nanmin(y4_x[begin:end]),np.nanmin(y4_y[begin:end]),np.nanmin(y4_z[begin:end])])
    maxs = np.asarray([np.nanmax(y4_x[begin:end]),np.nanmax(y4_y[begin:end]),np.nanmax(y4_z[begin:end])])
    min_y = np.nanmin(mins)
    max_y = np.nanmax(maxs) * 1.1
    ax4.set_ylim(min_y, max_y) 
    ax4.set_xticklabels(unix_to_utc(ax4.get_xticks()))
    plt.setp(ax4.get_xticklabels(), visible=False) #Share x-axis
    
    begin, end = get_begin_end_indicies(t5, xmin, xmax)
    ax5.plot(t5,y5_x,'r',lw=0.5,label='X')
    ax5.plot(t5,y5_y,'g',lw=0.5,label='Y')
    ax5.plot(t5,y5_z,'b',lw=0.5,label='Z')
    ax5.set_ylabel('SCW (nT)',fontsize=8)
    mins = np.asarray([np.nanmin(y5_x[begin:end]),np.nanmin(y5_y[begin:end]),np.nanmin(y5_z[begin:end])])
    maxs = np.asarray([np.nanmax(y5_x[begin:end]),np.nanmax(y5_y[begin:end]),np.nanmax(y5_z[begin:end])])
    min_y = np.nanmin(mins)
    max_y = np.nanmax(maxs) * 1.1
    ax5.set_ylim(min_y, max_y) 
    ax5.set_xticklabels(unix_to_utc(ax5.get_xticks()))
    plt.setp(ax5.get_xticklabels(), visible=False) #Share x-axis
    
    begin, end = get_begin_end_indicies(t6, xmin, xmax)
    ax6.semilogy(t6,y6_x,'ro',markersize=0.75,label='Parallel')
    ax6.semilogy(t6,y6_y,'go',markersize=0.75,label='Orthogonal')
    ax6.semilogy(t6,y6_z,'bo',markersize=0.75,label='Anti-Parallel')
    ax6.set_ylabel('Red $ \chi ^{2}$',fontsize=8)
    ax6.set_xlabel('Time From Start of Date (s)')
    mins = np.asarray([np.nanmin(y6_x[begin:end]),np.nanmin(y6_y[begin:end]),np.nanmin(y6_z[begin:end])])
    maxs = np.asarray([np.nanmax(y6_x[begin:end]),np.nanmax(y6_y[begin:end]),np.nanmax(y6_z[begin:end])])
    min_y = np.nanmin(mins)
    max_y = np.nanmax(maxs) * 1.1
    ax6.set_ylim(min_y, max_y) 
    ax6.set_xticklabels(unix_to_utc(ax6.get_xticks()))
    
    fig.suptitle(plot_title)
    fig.subplots_adjust(left=0.15, bottom=0.1, right=0.97, top=0.94, wspace=0.2, hspace=0.18) #hspace = 0.38 with labels
    plt.savefig(path + date + savedir + 'Analysis_figure_%d.png'%(i+1))
    plt.clf()
    # plt.show()
