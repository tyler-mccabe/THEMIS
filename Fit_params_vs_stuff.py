import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

path = '/Users/Tyler/Documents/NASA/Summer2018/THEMIS/07_14_2008/'
file1 = 'dh_274_Fit_param_B.txt'
file2 = 'dh_275_Fit_param_C.txt'
file3 = 'dh_104_ion_bulk_velocity.txt'
file4 = 'dh_90_ion_density_burst.txt'
file5 = 'dh_42.txt' # Magnetic Field Magnitude
file6 = 'dh_75_electron_density_burst.txt'
file7 = 'dh_101_electron_bulk_velocity_gse.txt'
file8 = 'dh_284_Red_Chi-Squared_value_each_fit.txt'
file9 = 'dh_30_fgh_gse.txt'
file10 = 'dh_76_electron_temp_burst.txt'
file11 = 'dh_91_ion_temp_burst.txt'

f1 = np.loadtxt(path + file1)
f2 = np.loadtxt(path + file2)
f3 = np.loadtxt(path + file3)
f4 = np.loadtxt(path + file4)
f5 = np.loadtxt(path + file5)
f6 = np.loadtxt(path + file6)
f7 = np.loadtxt(path + file7)
f8 = np.loadtxt(path + file8)
f9 = np.loadtxt(path + file9)
f10 = np.loadtxt(path + file10)
f11 = np.loadtxt(path + file11)

##########
## Fit Param B
##########
t1 = f1[0,:]
B_par = f1[1,:]
B_orth = f1[2,:]
B_anti = f1[3,:]

##########
## Fit Param C
##########
t2 = f2[0,:]
C_par = f2[1,:]
C_orth = f2[2,:]
C_anti = f2[3,:]

##########
## Ion Bulk Velocity
##########
t3 = f3[0,:]
v1_x = f3[1,:] #GSE Coordinates
v1_y = f3[2,:] #GSE Coordinates
v1_z = f3[3,:] #GSE Coordinates

##########
## Ion Density
##########
t4 = f4[0,:]
n = f4[1,:]

##########
## Magnetic Field Magnitude
##########
t5 = f5[0,:]
B1 = f5[1,:]

##########
## Electron Density
##########
t6 = f6[0,:]
n1 = f6[1,:]

##########
## Electron Bulk Velocity
##########
t7 = f7[0,:]
v2_x = f7[1,:] #GSE Coordinates
v2_y = f7[2,:] #GSE Coordinates
v2_z = f7[3,:] #GSE Coordinates

##########
## Reduced Chi-Squared
##########
t8 = f8[0,:]
Chi_par = f8[1,:]
Chi_orth = f8[2,:]
Chi_anti = f8[3,:]

##########
## Magnetic Feld Vector
##########
t9 = f9[0,:]
B1_x = f9[1,:]
B1_y = f9[2,:]
B1_z = f9[3,:]

##########
## Electron Temperature
##########
t10 = f10[0,:]
T_e1 = f10[1,:]

##########
## Ion Temperature
##########
t11 = f11[0,:]
T_i1 = f11[1,:]

##########
## Interpolation
##########
# print(np.shape(t1))   (1330,)
# print(np.shape(t2))   (1330,)
# print(np.shape(t3))   (2710,) 
# print(np.shape(t4))   (2710,)
# print(np.shape(t5))   (1084440,)

time = t1
vi_x = np.interp(time,t3,v1_x)
vi_y = np.interp(time,t3,v1_y)
vi_z = np.interp(time,t3,v1_z)
n_i = np.interp(time,t4,n)
B_mag = np.interp(time,t5,B1)
n_e = np.interp(time,t6,n1)
ve_x = np.interp(time,t7,v2_x)
ve_y = np.interp(time,t7,v2_y)
ve_z = np.interp(time,t7,v2_z)
B_x = np.interp(time,t9,B1_x)
B_y = np.interp(time,t9,B1_y)
B_z = np.interp(time,t9,B1_z)
T_e = np.interp(time,t10,T_e1)
T_i = np.interp(time,t11,T_i1)

##########
## Calculate Mach Number (From Truemann 2009 page 411)
##########
# print(n_e)
mass_e = 9.1e-28 #g
rho_e = n_e * mass_e
M_e = np.absolute(ve_x*1e5)/(B_mag*1e-5)*np.sqrt(rho_e)
print(M_e)

##########
## Plotting
##########
B_par_abs = np.absolute(B_par)
B_orth_abs = np.absolute(B_orth)
B_anti_abs = np.absolute(B_anti)
C_par_abs = np.absolute(C_par)
C_orth_abs = np.absolute(C_orth)
C_anti_abs = np.absolute(C_anti)
Eo_par = 1./C_par_abs
Eo_orth = 1./C_orth_abs
Eo_anti = 1./C_anti_abs

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

# ax1.loglog(B_anti_abs,T_i,'bo',markersize=1,label='Anti-Parallel')
# ax1.loglog(B_orth_abs,T_i,'go',markersize=1,label='Orthogonal')
# ax1.loglog(B_par_abs,T_i,'ro',markersize=1,label='Parallel')
ax1.semilogy(T_e,ve_x,'ro',markersize=1)
ax1.semilogy(T_e,ve_y,'go',markersize=1)
ax1.semilogy(T_e,ve_z,'bo',markersize=1)
ax1.set_ylabel('Electron Velocity')
ax1.set_xlabel('Electron Temperature')

# ax2.loglog(Eo_par,T_i,'ro',markersize=1,label='Parallel')
# ax2.loglog(Eo_orth,T_i,'go',markersize=1,label='Orthogonal')
# ax2.loglog(Eo_anti,T_i,'bo',markersize=1,label='Anti-Parallel')
ax2.semilogy(T_i,vi_x,'ro',markersize=1)
ax2.semilogy(T_i,vi_y,'go',markersize=1)
ax2.semilogy(T_i,vi_z,'bo',markersize=1)
ax2.set_xlabel('Ion Temperature')
ax2.set_ylabel('Ion Velocity')

ax3.loglog(T_e,Chi_par,'ro',markersize=1,label='Parallel')
ax3.loglog(T_e,Chi_orth,'go',markersize=1,label='Orthogonal')
ax3.loglog(T_e,Chi_anti,'bo',markersize=1,label='Anti-Parallel')
ax3.set_ylabel('Reduced $\chi^{2}$')
ax3.set_xlabel('Electron Temperature')

ax4.loglog(T_i,Chi_par,'ro',markersize=1,label='Parallel')
ax4.loglog(T_i,Chi_orth,'go',markersize=1,label='Orthogonal')
ax4.loglog(T_i,Chi_anti,'bo',markersize=1,label='Anti-Parallel')
ax4.set_ylabel('Reduced $\chi^{2}$')
ax4.set_xlabel('Ion Temperature')
plt.tight_layout()

##########
## Determine Lowest Chi Squared
##########
# low_Chi_par = np.asarray([])
# low_Chi_orth = np.asarray([])
# low_Chi_anti = np.asarray([])
# par_times = np.asarray([])
# orth_times = np.asarray([])
# anti_times = np.asarray([])
# for i in range(len(time)):
#     if Chi_par[i] <= 10.0:
#         low_Chi_par = np.append(low_Chi_par,Chi_par[i])
#         par_times = np.append(par_times,time[i])
#     if Chi_orth[i] <= 10.0:
#         low_Chi_orth = np.append(low_Chi_orth,Chi_orth[i])
#         orth_times = np.append(orth_times,time[i])
#     if Chi_anti[i] <= 10.0:
#         low_Chi_anti = np.append(low_Chi_anti,Chi_anti[i])
#         anti_times = np.append(anti_times,time[i])
# print('Number of parallel fits: %d'%len(par_times))
# print('Number of orthogonal fits: %d'%len(orth_times))
# print('Number of anti-parallel fits: %d'%len(anti_times))        
# 
# for i in range(len(orth_times)):
#     fig = plt.figure()
#     ax1 = fig.add_subplot(7,1,1)
#     ax2 = fig.add_subplot(7,1,2)
#     ax3 = fig.add_subplot(7,1,3)
#     ax4 = fig.add_subplot(7,1,4)
#     ax5 = fig.add_subplot(7,1,5)
#     ax6 = fig.add_subplot(7,1,6)
#     ax7 = fig.add_subplot(7,1,7)
#     
#     xmin = orth_times[i] - 20
#     xmax = orth_times[i] + 20
#     
#     time_new = np.asarray([])
#     for t in range(len(time)):
#         if time[t] >= xmin and time[t] <= xmax:
#             time_new = np.append(time_new,time[t])
#     start = list(time).index(np.nanmin(time_new))
#     stop = list(time).index(np.nanmax(time_new))
#     
#     ax1.plot(time,B_mag,'k',lw=1)
#     ax1.set_ylabel('$|\mathbf{B}_{o}| (nT)$',fontsize=7)
#     plt.setp(ax1.get_xticklabels(), visible=False) #Share x-axis
#     ax1.set_xlim(xmin,xmax)
#     ax1.set_ylim(np.nanmin(B_mag[start:stop]), np.nanmax(B_mag[start:stop]))
#     
#     ax2.plot(time,B_x,'r',lw=1,label='X')
#     ax2.plot(time,B_y,'g',lw=1,label='Y')
#     ax2.plot(time,B_z,'b',lw=1,label ='Z')
#     # ax2.legend(loc='lower left', frameon=True,prop={'size': 6})
#     ax2.set_ylabel('$\mathbf{B}_{o} (nT)$',fontsize=7)
#     plt.setp(ax2.get_xticklabels(), visible=False) #Share x-axis
#     ax2.set_xlim(xmin,xmax)
#     mins = np.asarray([np.nanmin(B_x[start:stop]),np.nanmin(B_y[start:stop]),np.nanmin(B_z[start:stop])])
#     maxs = np.asarray([np.nanmax(B_x[start:stop]),np.nanmax(B_y[start:stop]),np.nanmax(B_z[start:stop])])
#     min_y = np.nanmin(mins)
#     max_y = np.nanmax(maxs)
#     ax2.set_ylim(min_y, max_y)
# 
#     ax3.semilogy(time,n_i,'k',lw=1)
#     ax3.set_ylabel('$n_{i}\ (cm^{-3})$',fontsize=7)
#     plt.setp(ax3.get_xticklabels(), visible=False) #Share x-axis
#     ax3.set_xlim(xmin,xmax)
#     ax3.set_ylim(np.nanmin(n_i[start:stop]), np.nanmax(n_i[start:stop]))
#     
#     ax4.plot(time,ve_x,'r',lw=1,label='X')
#     ax4.plot(time,ve_y,'g',lw=1,label='Y')
#     ax4.plot(time,ve_z,'b',lw=1,label='Z')
#     # ax4.legend(loc='best', frameon=True,prop={'size': 6})
#     ax4.set_ylabel('Electron $\mathbf{V}_{bulk} $(km/s)',fontsize=7)
#     plt.setp(ax4.get_xticklabels(), visible=False) #Share x-axis
#     ax4.set_xlim(xmin,xmax)
#     mins = np.asarray([np.nanmin(ve_x[start:stop]),np.nanmin(ve_y[start:stop]),np.nanmin(ve_z[start:stop])])
#     maxs = np.asarray([np.nanmax(ve_x[start:stop]),np.nanmax(ve_y[start:stop]),np.nanmax(ve_z[start:stop])])
#     min_y = np.nanmin(mins)
#     max_y = np.nanmax(maxs)
#     ax4.set_ylim(min_y, max_y)
#     
#     ax5.plot(time,vi_x,'r',lw=1,label='X')
#     ax5.plot(time,vi_y,'g',lw=1,label='Y')
#     ax5.plot(time,vi_z,'b',lw=1,label='Z')
#     # ax5.legend(loc='best', frameon=True,prop={'size': 6})
#     ax5.set_ylabel('Ion $\mathbf{V}_{bulk} $(km/s)',fontsize=7)
#     plt.setp(ax5.get_xticklabels(), visible=False) #Share x-axis
#     ax5.set_xlim(xmin,xmax)
#     mins = np.asarray([np.nanmin(vi_x[start:stop]),np.nanmin(vi_y[start:stop]),np.nanmin(vi_z[start:stop])])
#     maxs = np.asarray([np.nanmax(vi_x[start:stop]),np.nanmax(vi_y[start:stop]),np.nanmax(vi_z[start:stop])])
#     min_y = np.nanmin(mins)
#     max_y = np.nanmax(maxs)
#     ax5.set_ylim(min_y, max_y)
#     
#     ax6.plot(time,T_e,'k',lw=1)
#     ax6.set_ylabel('$T_{e}$ (eV)',fontsize=7)
#     plt.setp(ax6.get_xticklabels(), visible=False) #Share x-axis
#     ax6.set_xlim(xmin,xmax)
#     ax6.set_ylim(np.nanmin(T_e[start:stop]), np.nanmax(T_e[start:stop]))
#     
#     
#     ax7.plot(time,T_i,'k',lw=1)
#     ax7.set_ylabel('$T_{i}$ (eV)',fontsize=7)
#     ax7.set_xlabel('Time From Start of Date (s)')
#     ax7.set_xlim(xmin,xmax)
#     ax7.set_ylim(np.nanmin(T_i[start:stop]), np.nanmax(T_i[start:stop]))
#     
#     fig.subplots_adjust(left=0.1, bottom=0.09, right=0.95, top=0.96, wspace=0.2, hspace=0.12)
# 
plt.show()