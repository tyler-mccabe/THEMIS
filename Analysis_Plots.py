import numpy as np
import matplotlib.pyplot as plt

path = '/Users/Tyler/Documents/NASA/Summer2018/TestIDL/'
file1 = 'dh_42.txt'
file2 = 'dh_249.txt'
file3 = 'dh_243.txt'
file4 = 'dh_252.txt'
file5 = 'dh_254.txt'
file6 = 'dh_284_Red_Chi-Squared_value_each_fit.txt'

f1 = path + file1
f2 = path + file2
f3 = path + file3
f4 = path + file4
f5 = path + file5
f6 = path + file6
f1 = np.loadtxt(f1) # Mag field magnitude -thc_fgh_mag Magnitude of the thc_fgh_gse 3-vectors
f2 = np.loadtxt(f2) # DC-coupled electric field vectors
f3 = np.loadtxt(f3) # AC-coupled electric field vectors
f4 = np.loadtxt(f4) # DC-coupled magnetic field vectors from the search coil magnetometer data
f5 = np.loadtxt(f5) # AC-coupled magnetic field vectors from the burst search coil magnetometer data
f6 = np.loadtxt(f6) # Reduced Chi-Squared values

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

##############
# Reduced Chi-Squared plot
##############
t6 = f6[0,:]
y6_par = f6[1,:]
y6_orth = f6[2,:]
y6_anti = f6[3,:]

##############
# Overall Plot Structure
##############

# xmin1 = np.asarray([44350,47440,55020,71870,78800,81170,55257.5])
# xmax1 = np.asarray([45110,48100,55670,72530,80130,81860,55272])
xmin1 = np.asarray([44695,44741,47987,48000,55260,55572,72145,72266,79062,79092,79661,79697,81446,81487])
xmax1 = np.asarray([44707,44752,47994,48008,55268,55582,72153,72277,79076,79109,79671,79711,81458,81496])
# names = ['1st','2nd','3rd','4th','5th','6th','Close up']
names = ['1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th','11th','12th','13th','14th']
for i in range(len(xmin1)):
    xmin = xmin1[i]
    xmax = xmax1[i]
    name = names[i]
    fig = plt.figure(figsize=(5,6))
    ax1 = fig.add_subplot(6,1,1)
    ax2 = fig.add_subplot(6,1,2)
    ax3 = fig.add_subplot(6,1,3)
    ax4 = fig.add_subplot(6,1,4)
    ax5 = fig.add_subplot(6,1,5)
    ax6 = fig.add_subplot(6,1,6)
    
    ax1.plot(t1,y1,'k',lw=0.1)
    ax1.set_ylabel('$|\mathbf{B}_{o}| (nT)$',fontsize=8)
    plt.setp(ax1.get_xticklabels(), visible=False) #Share x-axis
    ax1.set_xlim(xmin,xmax)
    if i == 2:
        ax1.set_ylim(0,9)
    if i == 3:
        ax1.set_ylim(0,20)
    if i == 4:
        ax1.set_ylim(0,28)
    if i == 5 or i == 6:
        ax1.set_ylim(0,10)
    
    ax2.plot(t2,y2_x,'r',lw=0.1,label='X')
    ax2.plot(t2,y2_y,'g',lw=0.1,label='Y')
    ax2.plot(t2,y2_z,'b',lw=0.1, label ='Z')
    # ax2.legend(loc='lower left', frameon=True,prop={'size': 6})
    ax2.set_ylabel('EFP (mV/m)',fontsize=8)
    plt.setp(ax2.get_xticklabels(), visible=False) #Share x-axis
    ax2.set_xlim(xmin,xmax)
    if i == 2 or i == 6:
        ax2.set_ylim(-30,25)
    if i == 5:
        ax2.set_ylim(-25,20)
    
    ax3.plot(t3,y3_x,'r',lw=0.1,label='X')
    ax3.plot(t3,y3_y,'g',lw=0.1,label='Y')
    ax3.plot(t3,y3_z,'b',lw=0.1,label='Z')
    ax3.set_ylabel('EFW (mV/m)',fontsize=8)
    plt.setp(ax3.get_xticklabels(), visible=False) #Share x-axis
    ax3.set_xlim(xmin,xmax)
    if i == 2 or i == 5 or i == 6:
        ax3.set_ylim(-10,10)
    if i == 3:
        ax3.set_ylim(-50,50)
    
    ax4.plot(t4,y4_x,'r',lw=0.1,label='X')
    ax4.plot(t4,y4_y,'g',lw=0.1,label='Y')
    ax4.plot(t4,y4_z,'b',lw=0.1,label='Z')
    # ax4.legend(loc='best', frameon=True,prop={'size': 6})
    ax4.set_ylabel('SCP (nT)',fontsize=8)
    plt.setp(ax4.get_xticklabels(), visible=False) #Share x-axis
    ax4.set_xlim(xmin,xmax)
    if i == 0 or i ==1:
        ax4.set_ylim(-25,20)
    if i == 2:
        ax4.set_ylim(-3,5)
    if i == 3 or i == 6:
        ax4.set_ylim(-3,3)
    if i == 4:
        ax4.set_ylim(-10,10)
    if i == 5:
        ax4.set_ylim(-2,2.5)
    
    ax5.plot(t5,y5_x,'r',lw=0.1,label='X')
    ax5.plot(t5,y5_y,'g',lw=0.1,label='Y')
    ax5.plot(t5,y5_z,'b',lw=0.1,label='Z')
    # ax5.legend(loc='lower left', frameon=True,prop={'size': 6})
    ax5.set_ylabel('SCW (nT)',fontsize=8)
    plt.setp(ax5.get_xticklabels(), visible=False) #Share x-axis
    ax5.set_xlim(xmin,xmax)
    if i == 0 or i == 2 or i == 3:
        ax5.set_ylim(-0.06,0.06)
    if i == 4 or i == 6:
        ax5.set_ylim(-0.1,0.1)
    if i == 5:
        ax5.set_ylim(-0.07,0.07)
    
    ax6.semilogy(t6,y6_par,'ro',markersize=0.75,label='Parallel')
    ax6.semilogy(t6,y6_orth,'go',markersize=0.75,label='Orthogonal')
    ax6.semilogy(t6,y6_anti,'bo',markersize=0.75,label='Anti-Parallel')
    ax6.set_ylabel('Red $ \chi ^{2}$',fontsize=8)
    ax6.set_xlabel('Time From Start of Date (s)')
    ax6.set_xlim(xmin,xmax)
    
    # fig.suptitle('Themis C: 2008-07-14 - %s Area of Interest'%name)
    fig.suptitle('Themis C: 2008-07-14 Burst - %s Area of Interest'%name)
    # fig.subplots_adjust(left=0.15, bottom=0.1, right=0.96, top=0.93, wspace=0.2, hspace=0.08)
    fig.subplots_adjust(left=0.16, bottom=0.08, right=0.96, top=0.93, wspace=0.2, hspace=0.2)
    # plt.tight_layout()
    # plt.savefig(path + 'Themis_2008_07_14_Analysis_%s.png'%name)
    # plt.savefig(path + 'Themis_2008_07_14_Analysis_%s.pdf'%name)
plt.show()