import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import io

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

# plt.ion()
# plt.cla()

file = '/Users/Tyler/Documents/NASA/Summer2018/THEMIS/07_14_2008/TPLOT_save_file_THC_FGM-ALL_EESA-IESA-Moments_EFI-Cal-Cor_SCM-Cal-Cor_ESA-SST-Spectra_eVDF_fit_results_2008-07-14_0000x00-2359x59.tplot'
f = scipy.io.readsav(file,python_dict=True) 

index = 133

dq = f['dq']
dh = dq['dh']

t = dh[index-1].x[0]
# print(t.shape)
t = t - 1215993600

##########
## Get Data
##########
time = dh[272].x[0]   # Power Law A
# print('time',time.shape)
A_par = dh[272].y[0][0]#[700] # 1330
A_orth = dh[272].y[0][1]
A_anti = dh[272].y[0][2]
B_par = dh[273].y[0][0]#[700]
B_orth = dh[273].y[0][1]
B_anti = dh[273].y[0][2]
C_par = dh[274].y[0][0]#[700]
C_orth = dh[274].y[0][1] 
C_anti = dh[274].y[0][2]
chi_sq_par_good = dh[283].y[0][0]
chi_sq_orth_good = dh[283].y[0][1]
chi_sq_anti_good = dh[283].y[0][2]

flux_par = dh[index-1].y[0]  #(20,1362)
flux_orth = dh[135].y[0]    
flux_anti = dh[138].y[0]
energy_par = dh[index-1].v[0]   #(20,1362)
energy_orth = dh[135].v[0]
energy_anti = dh[138].v[0]

flux_par_new = np.zeros((20,1330))
flux_orth_new = np.zeros((20,1330))
flux_anti_new = np.zeros((20,1330))
energy_par_new = np.zeros((20,1330))
energy_orth_new = np.zeros((20,1330))
energy_anti_new = np.zeros((20,1330))

##########
## Good Fits
##########
t_good = dh[292].x[0] - 1215993600
t_good = np.around(t_good,decimals=2)
t = np.around(t,decimals=2) #1362
times = np.intersect1d(t_good,t) #1330

for i in range(len(times)):
    x = times[i]
    ind = list(t).index(x)
    flux_par_new[:,i] = flux_par[:,ind]
    flux_orth_new[:,i] = flux_orth[:,ind]
    flux_anti_new[:,i] = flux_anti[:,ind]
    energy_par_new[:,i] = energy_par[:,ind]
    energy_orth_new[:,i] = energy_orth[:,ind]
    energy_anti_new[:,i] = energy_anti[:,ind]
# print(np.shape(flux_par_new))
# print(np.shape(energy_anti_new))

# print('times',len(times))
flags = dh[292].y[0][0] #1330
# print('flags',len(flags))
good_indicies = np.asarray([])
for i in range(len(flags)):
    x = flags[i]
    if x == 1.0:
        good_indicies = np.append(good_indicies,i)  # good contains all of the time indicies that correspond to a 'good' fit
times_good = np.asarray([])  # contains times corresponding to 'good' fit
energy_par_good = np.zeros((20,711))
energy_orth_good = np.zeros((20,711))
energy_anti_good = np.zeros((20,711))
flux_par_good = np.zeros((20,711))
flux_orth_good = np.zeros((20,711))
flux_anti_good = np.zeros((20,711))
A_par_good = np.asarray([])
B_par_good = np.asarray([])
C_par_good = np.asarray([])
chi_par_good = np.asarray([])
A_orth_good = np.asarray([])
B_orth_good = np.asarray([])
C_orth_good = np.asarray([])
chi_orth_good = np.asarray([])
A_anti_good = np.asarray([])
B_anti_good = np.asarray([])
C_anti_good = np.asarray([])
chi_anti_good = np.asarray([])
for i in range(len(good_indicies)):
    x = int(good_indicies[i])
    times_good = np.append(times_good,t_good[x])
    energy_par_good[:,i] = energy_par_new[:,x]
    energy_orth_good[:,i] = energy_orth_new[:,x]
    energy_anti_good[:,i] = energy_anti_new[:,x]
    flux_par_good[:,i] = flux_par_new[:,x]
    flux_orth_good[:,i] = flux_orth_new[:,x]
    flux_anti_good[:,i] = flux_anti_new[:,x]
    A_par_good = np.append(A_par_good,A_par[x])
    A_orth_good = np.append(A_orth_good,A_orth[x])
    A_anti_good = np.append(A_anti_good,A_anti[x])
    B_par_good = np.append(B_par_good,B_par[x])
    B_orth_good = np.append(B_orth_good,B_orth[x])
    B_anti_good = np.append(B_anti_good,B_anti[x])
    C_par_good = np.append(C_par_good,C_par[x])
    C_orth_good = np.append(C_orth_good,C_orth[x])
    C_anti_good = np.append(C_anti_good,C_anti[x])
    chi_par_good = np.append(chi_par_good,chi_sq_par_good[x])
    chi_orth_good = np.append(chi_orth_good,chi_sq_orth_good[x])
    chi_anti_good = np.append(chi_anti_good,chi_sq_anti_good[x])

##########
## Put Time into UTC
##########
hours = times_good/3600.
# print(hours)
int_hours = np.floor(hours)
minutes = (hours - int_hours)*60.
# print(minutes)
int_minutes = np.floor(minutes)
seconds = (minutes - int_minutes)*60
# print(seconds)
time_utc = []
for i in range(len(times_good)):
    time_utc = np.append(time_utc,'%d:%d:%.2f'%(int_hours[i],int_minutes[i],seconds[i]))
# print(time_utc)


minE = 50 #eV
maxE = 5*10**3 #eV
E = np.linspace(minE,maxE,711)
# for i in range(len(times_good)):
# # for i in range(10):
#     fig = plt.figure(figsize=(6,5))
#     ax = fig.add_subplot(1,1,1)
#     ax.loglog(energy_par_good[:,i],flux_par_good[:,i],'ro',markersize=2,label='Red. $\chi ^{2}$ = %.3f'%chi_par_good[i])
#     ax.loglog(energy_orth_good[:,i],flux_orth_good[:,i],'go',markersize=2,label='Red. $\chi ^{2}$ = %.3f'%chi_orth_good[i])
#     ax.loglog(energy_anti_good[:,i],flux_anti_good[:,i],'bo',markersize=2,label='Red. $\chi ^{2}$ = %.3f'%chi_anti_good[i])
#     ax.axvline(1./np.absolute(C_par_good[i]),c='r',ls='--',lw = 1)
#     ax.axvline(1./np.absolute(C_orth_good[i]),c='g',ls='--',lw = 1)
#     ax.axvline(1./np.absolute(C_anti_good[i]),c='b',ls='--',lw = 1)
#     f_par = A_par_good[i]*E**B_par_good[i]*np.exp(C_par_good[i]*E)
#     f_orth = A_orth_good[i]*E**B_orth_good[i]*np.exp(C_orth_good[i]*E)
#     f_anti = A_anti_good[i]*E**B_anti_good[i]*np.exp(C_anti_good[i]*E)
#     ax.loglog(E,f_par,'r')
#     ax.loglog(E,f_orth,'g')
#     ax.loglog(E,f_anti,'b')
#     ax.set_title('Fit index %d: Time = %s'%(i,time_utc[i]))
#     ax.set_xlabel('Energy (eV)')
#     ax.set_ylabel('Flux (#$cm^{-2}s^{-1}sr^{-1}eV^{-1}$)')
#     ax.legend(loc='lower left',frameon=True)
#     fig.subplots_adjust(left=0.16, bottom=0.15, right=0.96, top=0.91, wspace=0.2, hspace=0.09)
#     plt.savefig('/Users/Tyler/Documents/NASA/Summer2018/THEMIS/07_14_2008/Plots/Fits_with_data/Fit_%d.png'%i)

good_fits = np.asarray([8,12,16,18,19,22,30,49,77,95,105,116,117,122,123,124,127,129,183,203])
good_fits1 = np.asarray([222,260,273,316,334,335,336,337,344,345,348,353,390,393,397,400,407])
good_fits2 = np.asarray([413,415,416,418,464,471,472,478,519,541,577,638,647,653,655,674])

good_fits = np.append(good_fits,good_fits1)
good_fits = np.append(good_fits,good_fits2)
# print(np.shape(good_fits))

t1 = dh[41].x[0] - 1215993600
B_mag = dh[41].y[0]
t2 = dh[29].x[0] - 1215993600
B_x = dh[29].y[0][0]
B_y = dh[29].y[0][1]
B_z = dh[29].y[0][2]
t3 = dh[89].x[0] - 1215993600
n_i = dh[89].y[0]
t4 = dh[103].x[0] - 1215993600
vi_x = dh[103].y[0][0]
vi_y = dh[103].y[0][1]
vi_z = dh[103].y[0][2]
t5 = dh[75].x[0] - 1215993600
T_e = dh[75].y[0]
t6 = dh[90].x[0] - 1215993600
T_i = dh[90].y[0]
# for i in range(5):
# for i in range(len(good_fits)):
for i in range(len(times_good)):
    # x = good_fits[i]
    xmin = times_good[i] - 15 ## Change back to x
    xmax = times_good[i] + 15 ## Change back to x
    fig = plt.figure(figsize=(5,6))
    ax1 = fig.add_subplot(6,1,1)
    ax2 = fig.add_subplot(6,1,2)
    ax3 = fig.add_subplot(6,1,3)
    ax4 = fig.add_subplot(6,1,4)
    ax5 = fig.add_subplot(6,1,5)
    ax6 = fig.add_subplot(6,1,6)
    
    ax1.plot(t1,B_mag,'k',lw=0.5)
    ax1.set_ylabel('$|\mathbf{B}_{o}| (nT)$')
    # plt.setp(ax1.get_xticklabels(), visible=False) #Share x-axis
    ax1.set_xlim(xmin,xmax)
    t1_new = np.asarray([])
    for t in range(len(t1)):
        if t1[t] >= xmin and t1[t] <= xmax:
            t1_new = np.append(t1_new,t1[t])   
    start = list(t1).index(np.nanmin(t1_new))
    stop = list(t1).index(np.nanmax(t1_new))
    ax1.set_ylim(np.nanmin(B_mag[start:stop]), np.nanmax(B_mag[start:stop]))
    ax1.axvline(times_good[x],c='k',ls='--',lw=0.5)
    
    ax2.plot(t2,B_x,'r',lw=0.5,label='X')
    ax2.plot(t2,B_y,'g',lw=0.5,label='Y')
    ax2.plot(t2,B_z,'b',lw=0.5, label ='Z')
    # ax2.legend(loc='lower left', frameon=True,prop={'size': 6})
    ax2.set_ylabel('$\mathbf{B}_{o} (nT)$')
    # plt.setp(ax2.get_xticklabels(), visible=False) #Share x-axis
    ax2.set_xlim(xmin,xmax)
    t2_new = np.asarray([])
    for t in range(len(t2)):
        if t2[t] >= xmin and t2[t] <= xmax:
            t2_new = np.append(t2_new,t2[t])   
    start = list(t2).index(np.nanmin(t2_new))
    stop = list(t2).index(np.nanmax(t2_new))
    mins = np.asarray([np.nanmin(B_x[start:stop]),np.nanmin(B_y[start:stop]),np.nanmin(B_z[start:stop])])
    maxs = np.asarray([np.nanmax(B_x[start:stop]),np.nanmax(B_y[start:stop]),np.nanmax(B_z[start:stop])])
    min_y = np.nanmin(mins)
    max_y = np.nanmax(maxs)
    ax2.set_ylim(min_y, max_y)
    ax2.axvline(times_good[x],c='k',ls='--',lw=0.5)
    
    # ax3.semilogy(t3,n_i,'k',lw=0.5)
    ax3.plot(t3,n_i,'k',lw=0.5)
    ax3.set_ylabel('$n_{i}\ (cm^{-3})$')
    # plt.setp(ax3.get_xticklabels(), visible=False) #Share x-axis
    ax3.set_xlim(xmin,xmax)
    t3_new = np.asarray([])
    for t in range(len(t3)):
        if t3[t] >= xmin and t3[t] <= xmax:
            t3_new = np.append(t3_new,t3[t])   
    start = list(t3).index(np.nanmin(t3_new))
    stop = list(t3).index(np.nanmax(t3_new))
    ax3.set_ylim(np.nanmin(n_i[start:stop]), np.nanmax(n_i[start:stop]))
    ax3.axvline(times_good[x],c='k',ls='--',lw=0.5)
    
    ax4.plot(t4,vi_x,'r',lw=0.5,label='X')
    ax4.plot(t4,vi_y,'g',lw=0.5,label='Y')
    ax4.plot(t4,vi_z,'b',lw=0.5,label='Z')
    ax4.legend(loc='best', frameon=True,prop={'size': 6})
    ax4.set_ylabel('$\mathbf{V}_{bulk} $(km/s)')
    # plt.setp(ax4.get_xticklabels(), visible=False) #Share x-axis
    ax4.set_xlim(xmin,xmax)
    t4_new = np.asarray([])
    for t in range(len(t4)):
        if t4[t] >= xmin and t4[t] <= xmax:
            t4_new = np.append(t4_new,t4[t])   
    start = list(t4).index(np.nanmin(t4_new))
    stop = list(t4).index(np.nanmax(t4_new))
    mins = np.asarray([np.nanmin(vi_x[start:stop]),np.nanmin(vi_y[start:stop]),np.nanmin(vi_z[start:stop])])
    maxs = np.asarray([np.nanmax(vi_x[start:stop]),np.nanmax(vi_y[start:stop]),np.nanmax(vi_z[start:stop])])
    min_y = np.nanmin(mins)
    max_y = np.nanmax(maxs)
    ax4.set_ylim(min_y, max_y)
    ax4.axvline(times_good[x],c='k',ls='--',lw=0.5)
    
    # ax5.semilogy(t5,T_e,'k',lw=0.5)
    ax5.plot(t5,T_e,'k',lw=0.5)
    ax5.set_ylabel('$T_{e}$ (eV)')
    ax5.set_xlim(xmin,xmax)
    t5_new = np.asarray([])
    for t in range(len(t5)):
        if t5[t] >= xmin and t5[t] <= xmax:
            t5_new = np.append(t5_new,t5[t])   
    start = list(t5).index(np.nanmin(t5_new))
    stop = list(t5).index(np.nanmax(t5_new))
    ax5.set_ylim(np.nanmin(T_e[start:stop]), np.nanmax(T_e[start:stop]))
    ax5.axvline(times_good[x],c='k',ls='--',lw=0.5)
    
    # ax6.semilogy(t6,T_i,'k',lw=0.5)
    ax6.plot(t6,T_i,'k',lw=0.5)
    ax6.set_ylabel('$T_{i}$ (eV)')
    ax6.set_xlabel('Time From Start of Date (s)')
    ax6.set_xlim(xmin,xmax)
    t6_new = np.asarray([])
    for t in range(len(t6)):
        if t6[t] >= xmin and t6[t] <= xmax:
            t6_new = np.append(t6_new,t6[t])   
    start = list(t6).index(np.nanmin(t6_new))
    stop = list(t6).index(np.nanmax(t6_new))
    ax6.set_ylim(np.nanmin(T_i[start:stop]), np.nanmax(T_i[start:stop]))
    ax6.axvline(times_good[x],c='k',ls='--',lw=0.5)
    
    fig.suptitle('Themis C: 2008-07-14 - Fit With Data Index %d'%x)
    # fig.suptitle('Themis C: 2008-07-14 Plot 2 - %s Area of Interest'%name)
    fig.subplots_adjust(left=0.2, bottom=0.1, right=0.96, top=0.93, wspace=0.2, hspace=0.08)
    # plt.savefig('/Users/Tyler/Documents/NASA/Summer2018/THEMIS/07_14_2008/Plots/Fits_with_data/Select_fits/Index_%d.png'%x)
    plt.savefig('/Users/Tyler/Documents/NASA/Summer2018/THEMIS/07_14_2008/Plots/Fits_with_data/Remaining_fits/Index_%d.png'%x)

## Magnetic Field Magnitude
t1 = dh[41].x[0] - 1215993600
y1 = dh[41].y[0]
## EFP Plot
t2 = dh[248].x[0] - 1215993600
y2_x = dh[248].y[0][0]
y2_y = dh[248].y[0][1]
y2_z = dh[248].y[0][2]
## EFW Plot
t3 = dh[242].x[0] - 1215993600
y3_x = dh[242].y[0][0]
y3_y = dh[242].y[0][1]
y3_z = dh[242].y[0][2]
## SCP Plot
t4 = dh[251].x[0] - 1215993600
y4_x = dh[251].y[0][0]
y4_y = dh[251].y[0][1]
y4_z = dh[251].y[0][2]
## SCW Plot
t5 = dh[253].x[0] - 1215993600
y5_x = dh[253].y[0][0]
y5_y = dh[253].y[0][1]
y5_z = dh[253].y[0][2]

# for i in range(5):
# for i in range(len(good_fits)):
#     x = good_fits[i]
#     xmin = times_good[x] - 2
#     xmax = times_good[x] + 2
#     fig = plt.figure(figsize=(7,6.5))
#     ax1 = fig.add_subplot(5,1,1)
#     ax2 = fig.add_subplot(5,1,2)
#     ax3 = fig.add_subplot(5,1,3)
#     ax4 = fig.add_subplot(5,1,4)
#     ax5 = fig.add_subplot(5,1,5)
#     
#     ax1.plot(t1,y1,'k',lw=0.5)
#     ax1.set_ylabel('$|\mathbf{B}_{o}| (nT)$',fontsize=8)
#     # plt.setp(ax1.get_xticklabels(), visible=False) #Share x-axis
#     ax1.set_xlim(xmin,xmax)
#     t1_new = np.asarray([])
#     for t in range(len(t1)):
#         if t1[t] >= xmin and t1[t] <= xmax:
#             t1_new = np.append(t1_new,t1[t])   
#     start = list(t1).index(np.nanmin(t1_new))
#     stop = list(t1).index(np.nanmax(t1_new))
#     ax1.set_ylim(np.nanmin(y1[start:stop]), np.nanmax(y1[start:stop]))
#     ax1.axvline(times_good[x],c='k',ls='--',lw=0.5)
#     
#     ax2.plot(t2,y2_x,'r',lw=0.5,label='X')
#     ax2.plot(t2,y2_y,'g',lw=0.5,label='Y')
#     ax2.plot(t2,y2_z,'b',lw=0.5, label ='Z')
#     # ax2.legend(loc='lower left', frameon=True,prop={'size': 6})
#     ax2.set_ylabel('EFP (mV/m)',fontsize=8)
#     # plt.setp(ax2.get_xticklabels(), visible=False) #Share x-axis
#     ax2.set_xlim(xmin,xmax)
#     t2_new = np.asarray([])
#     for t in range(len(t2)):
#         if t2[t] >= xmin and t2[t] <= xmax:
#             t2_new = np.append(t2_new,t2[t])   
#     start = list(t2).index(np.nanmin(t2_new))
#     stop = list(t2).index(np.nanmax(t2_new))
#     mins = np.asarray([np.nanmin(y2_x[start:stop]),np.nanmin(y2_y[start:stop]),np.nanmin(y2_z[start:stop])])
#     maxs = np.asarray([np.nanmax(y2_x[start:stop]),np.nanmax(y2_y[start:stop]),np.nanmax(y2_z[start:stop])])
#     min_y = np.nanmin(mins)
#     max_y = np.nanmax(maxs)
#     ax2.set_ylim(min_y, max_y)
#     ax2.axvline(times_good[x],c='k',ls='--',lw=0.5)
#     
#     ax3.plot(t3,y3_x,'r',lw=0.1,label='X')
#     ax3.plot(t3,y3_y,'g',lw=0.1,label='Y')
#     ax3.plot(t3,y3_z,'b',lw=0.1,label='Z')
#     ax3.set_ylabel('EFW (mV/m)',fontsize=8)
#     # plt.setp(ax3.get_xticklabels(), visible=False) #Share x-axis
#     ax3.set_xlim(xmin,xmax)
#     t3_new = np.asarray([])
#     # for t in range(len(t3)):
#     #     if t3[t] >= xmin and t3[t] <= xmax:
#     #         t3_new = np.append(t3_new,t3[t])   
#     # start = list(t3).index(np.nanmin(t3_new))
#     # stop = list(t3).index(np.nanmax(t3_new))
#     # mins = np.asarray([np.nanmin(y3_x[start:stop]),np.nanmin(y3_y[start:stop]),np.nanmin(y3_z[start:stop])])
#     # maxs = np.asarray([np.nanmax(y3_x[start:stop]),np.nanmax(y3_y[start:stop]),np.nanmax(y3_z[start:stop])])
#     # min_y = np.nanmin(mins)
#     # max_y = np.nanmax(maxs)
#     # ax3.set_ylim(min_y, max_y)
#     ax3.axvline(times_good[x],c='k',ls='--',lw=0.5)
#     
#     ax4.plot(t4,y4_x,'r',lw=0.5,label='X')
#     ax4.plot(t4,y4_y,'g',lw=0.5,label='Y')
#     ax4.plot(t4,y4_z,'b',lw=0.5,label='Z')
#     # ax4.legend(loc='best', frameon=True,prop={'size': 6})
#     ax4.set_ylabel('SCP (nT)',fontsize=8)
#     # plt.setp(ax4.get_xticklabels(), visible=False) #Share x-axis
#     ax4.set_xlim(xmin,xmax)
#     t4_new = np.asarray([])
#     for t in range(len(t4)):
#         if t4[t] >= xmin and t4[t] <= xmax:
#             t4_new = np.append(t4_new,t4[t])   
#     start = list(t4).index(np.nanmin(t4_new))
#     stop = list(t4).index(np.nanmax(t4_new))
#     mins = np.asarray([np.nanmin(y4_x[start:stop]),np.nanmin(y4_y[start:stop]),np.nanmin(y4_z[start:stop])])
#     maxs = np.asarray([np.nanmax(y4_x[start:stop]),np.nanmax(y4_y[start:stop]),np.nanmax(y4_z[start:stop])])
#     min_y = np.nanmin(mins)
#     max_y = np.nanmax(maxs)
#     ax4.set_ylim(min_y, max_y)  
#     ax4.axvline(times_good[x],c='k',ls='--',lw=0.5)  
#     
#     ax5.plot(t5,y5_x,'r',lw=0.1,label='X')
#     ax5.plot(t5,y5_y,'g',lw=0.1,label='Y')
#     ax5.plot(t5,y5_z,'b',lw=0.1,label='Z')
#     # ax5.legend(loc='lower left', frameon=True,prop={'size': 6})
#     ax5.set_ylabel('SCW (nT)',fontsize=8)
#     # plt.setp(ax5.get_xticklabels(), visible=False) #Share x-axis
#     ax5.set_xlabel('Time From Start of Date (s)')
#     ax5.set_xlim(xmin,xmax)
#     # t5_new = np.asarray([])
#     # for t in range(len(t5)):
#     #     if t5[t] >= xmin and t5[t] <= xmax:
#     #         t5_new = np.append(t5_new,t5[t])   
#     # start = list(t5).index(np.nanmin(t5_new))
#     # stop = list(t5).index(np.nanmax(t5_new))
#     # mins = np.asarray([np.nanmin(y5_x[start:stop]),np.nanmin(y5_y[start:stop]),np.nanmin(y5_z[start:stop])])
#     # maxs = np.asarray([np.nanmax(y5_x[start:stop]),np.nanmax(y5_y[start:stop]),np.nanmax(y5_z[start:stop])])
#     # min_y = np.nanmin(mins)
#     # max_y = np.nanmax(maxs)
#     # ax5.set_ylim(min_y, max_y)
#     ax5.axvline(times_good[x],c='k',ls='--',lw=0.5)
#     
#     fig.suptitle('Themis C: 2008-07-14 - Fit With Data Index %d'%x)
#     fig.subplots_adjust(left=0.12, bottom=0.08, right=0.96, top=0.9, wspace=0.2, hspace=0.2)
#     plt.savefig('/Users/Tyler/Documents/NASA/Summer2018/THEMIS/07_14_2008/Plots/Fits_with_data/Select_fits/Detail/Index_%d_detail.png'%x)