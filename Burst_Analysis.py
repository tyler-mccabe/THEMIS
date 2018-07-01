import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

# EFW and SCW burst analysis
path = '/Users/Tyler/Documents/NASA/Summer2018/TestIDL/'
file1 = 'dh_42.txt'
file2 = 'dh_249.txt'
file3 = 'dh_243.txt'
file4 = 'dh_252.txt'
file5 = 'dh_254.txt'

f1 = path + file1
f2 = path + file2
f3 = path + file3
f4 = path + file4
f5 = path + file5

f1 = np.loadtxt(f1) # Mag field magnitude -thc_fgh_mag Magnitude of the thc_fgh_gse 3-vectors
f2 = np.loadtxt(f2) # DC-coupled electric field vectors
f3 = np.loadtxt(f3) # AC-coupled electric field vectors
f4 = np.loadtxt(f4) # DC-coupled magnetic field vectors from the search coil magnetometer data
f5 = np.loadtxt(f5) # AC-coupled magnetic field vectors from the burst search coil magnetometer data

##############
# Magnetic Field Magnitude Plot
##############
t1 = f1[0,:]
y1 = f1[1,:]

##############
# EFP Plot
##############
t2 = f2[0,:]
y2_x = f2[1,:]
y2_y = f2[2,:]
y2_z = f2[3,:]

##############
# EFW Plot
##############
t3 = f3[0,:]
y3_x = f3[1,:]
y3_y = f3[2,:]
y3_z = f3[3,:]

##############
# SCP Plot
##############
t4 = f4[0,:]
y4_x = f4[1,:]
y4_y = f4[2,:]
y4_z = f4[3,:]

##############
# SCW Plot
##############
t5 = f5[0,:]
y5_x = f5[1,:]
y5_y = f5[2,:]
y5_z = f5[3,:]

xmin1 = np.asarray([44696.7,44740.5,47986.8,47999.8,55259.8,55571.8,72144.8,72266.8,79062.6,79095,79661,79699,81446.8,81486.5])
xmax1 = np.asarray([44704.9,44748,47994.2,48008,55266.8,55580,72152.4,72275.1,79071,79103.2,79667,79707.2,81454.8,81495])
names = ['1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th','11th','12th','13th','14th']
for i in range(len(xmin1)):
    xmin = xmin1[i]
    xmax = xmax1[i]
    name = names[i]
    fig = plt.figure(figsize=(7,6.5))
    ax1 = fig.add_subplot(5,1,1)
    ax2 = fig.add_subplot(5,1,2)
    ax3 = fig.add_subplot(5,1,3)
    ax4 = fig.add_subplot(5,1,4)
    ax5 = fig.add_subplot(5,1,5)
    
    ax1.plot(t1,y1,'k',lw=0.5)
    ax1.set_ylabel('$|\mathbf{B}_{o}| (nT)$',fontsize=8)
    # plt.setp(ax1.get_xticklabels(), visible=False) #Share x-axis
    ax1.set_xlim(xmin,xmax)
    t1_new = np.asarray([])
    for t in range(len(t1)):
        if t1[t] >= xmin and t1[t] <= xmax:
            t1_new = np.append(t1_new,t1[t])   
    start = list(t1).index(np.nanmin(t1_new))
    stop = list(t1).index(np.nanmax(t1_new))
    ax1.set_ylim(np.nanmin(y1[start:stop]), np.nanmax(y1[start:stop]))
    
    ax2.plot(t2,y2_x,'r',lw=0.5,label='X')
    ax2.plot(t2,y2_y,'g',lw=0.5,label='Y')
    ax2.plot(t2,y2_z,'b',lw=0.5, label ='Z')
    # ax2.legend(loc='lower left', frameon=True,prop={'size': 6})
    ax2.set_ylabel('EFP (mV/m)',fontsize=8)
    # plt.setp(ax2.get_xticklabels(), visible=False) #Share x-axis
    ax2.set_xlim(xmin,xmax)
    t2_new = np.asarray([])
    for t in range(len(t2)):
        if t2[t] >= xmin and t2[t] <= xmax:
            t2_new = np.append(t2_new,t2[t])   
    start = list(t2).index(np.nanmin(t2_new))
    stop = list(t2).index(np.nanmax(t2_new))
    mins = np.asarray([np.nanmin(y2_x[start:stop]),np.nanmin(y2_y[start:stop]),np.nanmin(y2_z[start:stop])])
    maxs = np.asarray([np.nanmax(y2_x[start:stop]),np.nanmax(y2_y[start:stop]),np.nanmax(y2_z[start:stop])])
    min_y = np.nanmin(mins)
    max_y = np.nanmax(maxs)
    ax2.set_ylim(min_y, max_y)
    
    ax3.plot(t3,y3_x,'r',lw=0.1,label='X')
    ax3.plot(t3,y3_y,'g',lw=0.1,label='Y')
    ax3.plot(t3,y3_z,'b',lw=0.1,label='Z')
    ax3.set_ylabel('EFW (mV/m)',fontsize=8)
    # plt.setp(ax3.get_xticklabels(), visible=False) #Share x-axis
    ax3.set_xlim(xmin,xmax)
    t3_new = np.asarray([])
    for t in range(len(t3)):
        if t3[t] >= xmin and t3[t] <= xmax:
            t3_new = np.append(t3_new,t3[t])   
    start = list(t3).index(np.nanmin(t3_new))
    stop = list(t3).index(np.nanmax(t3_new))
    mins = np.asarray([np.nanmin(y3_x[start:stop]),np.nanmin(y3_y[start:stop]),np.nanmin(y3_z[start:stop])])
    maxs = np.asarray([np.nanmax(y3_x[start:stop]),np.nanmax(y3_y[start:stop]),np.nanmax(y3_z[start:stop])])
    min_y = np.nanmin(mins)
    max_y = np.nanmax(maxs)
    ax3.set_ylim(min_y, max_y)
    
    ax4.plot(t4,y4_x,'r',lw=0.5,label='X')
    ax4.plot(t4,y4_y,'g',lw=0.5,label='Y')
    ax4.plot(t4,y4_z,'b',lw=0.5,label='Z')
    # ax4.legend(loc='best', frameon=True,prop={'size': 6})
    ax4.set_ylabel('SCP (nT)',fontsize=8)
    # plt.setp(ax4.get_xticklabels(), visible=False) #Share x-axis
    ax4.set_xlim(xmin,xmax)
    t4_new = np.asarray([])
    for t in range(len(t4)):
        if t4[t] >= xmin and t4[t] <= xmax:
            t4_new = np.append(t4_new,t4[t])   
    start = list(t4).index(np.nanmin(t4_new))
    stop = list(t4).index(np.nanmax(t4_new))
    mins = np.asarray([np.nanmin(y4_x[start:stop]),np.nanmin(y4_y[start:stop]),np.nanmin(y4_z[start:stop])])
    maxs = np.asarray([np.nanmax(y4_x[start:stop]),np.nanmax(y4_y[start:stop]),np.nanmax(y4_z[start:stop])])
    min_y = np.nanmin(mins)
    max_y = np.nanmax(maxs)
    ax4.set_ylim(min_y, max_y)    
    
    ax5.plot(t5,y5_x,'r',lw=0.1,label='X')
    ax5.plot(t5,y5_y,'g',lw=0.1,label='Y')
    ax5.plot(t5,y5_z,'b',lw=0.1,label='Z')
    # ax5.legend(loc='lower left', frameon=True,prop={'size': 6})
    ax5.set_ylabel('SCW (nT)',fontsize=8)
    # plt.setp(ax5.get_xticklabels(), visible=False) #Share x-axis
    ax5.set_xlabel('Time From Start of Date (s)')
    ax5.set_xlim(xmin,xmax)
    t5_new = np.asarray([])
    for t in range(len(t5)):
        if t5[t] >= xmin and t5[t] <= xmax:
            t5_new = np.append(t5_new,t5[t])   
    start = list(t5).index(np.nanmin(t5_new))
    stop = list(t5).index(np.nanmax(t5_new))
    mins = np.asarray([np.nanmin(y5_x[start:stop]),np.nanmin(y5_y[start:stop]),np.nanmin(y5_z[start:stop])])
    maxs = np.asarray([np.nanmax(y5_x[start:stop]),np.nanmax(y5_y[start:stop]),np.nanmax(y5_z[start:stop])])
    min_y = np.nanmin(mins)
    max_y = np.nanmax(maxs)
    ax5.set_ylim(min_y, max_y)
    
    fig.suptitle('Themis C: 2008-07-14 Burst - %s Area of Interest'%name)
    fig.subplots_adjust(left=0.12, bottom=0.08, right=0.96, top=0.9, wspace=0.2, hspace=0.2)
    plt.savefig(path + 'Bursts/' + 'Themis_2008_07_14_Burst_Analysis_%s.png'%name)
    # plt.savefig(path + 'Themis_2008_07_14_Analysis_%s.pdf'%name)
plt.show()