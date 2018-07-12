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

path = '/Users/tylermccabe/Documents/NASA/Summer2018/THEMIS/'
date = '09_16_2008/'

tplot = themis + '.tplot'
savedir = 'Plots/' + themis + '/Fits/'
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
index0 = get_index('peeb_fit_status_good')   
index1 = get_index('peeb_fit_ener_spec_para_good')  
index2 = get_index('peeb_fit_ener_spec_perp_good')  
index3 = get_index('peeb_fit_ener_spec_anti_good')
index4 = get_index('peeb_amplitudes_good') # Fit param A
index5 = get_index('peeb_powerlaws_good') # Fit param B
index6 = get_index('peeb_ener_cutoffs_good') # 1/|C|
index7 = get_index('peeb_red_chisq_good')  

print('\n')
print('Fit Status            ',dh[index0].y[0].shape)
print('Parallel spec good    ',dh[index1].y[0].shape)
print('Perp spec good        ',dh[index2].y[0].shape)
print('Anti para spec good   ',dh[index3].y[0].shape)
print('Fit param A           ',dh[index4].y[0].shape)
print('Fit param B           ',dh[index5].y[0].shape)
print('Cutoff Energy         ',dh[index6].y[0].shape)
print('Red Chi^2             ',dh[index7].y[0].shape, '\n')

##########
## Get Data
##########
## For spectra data, v = energy y = flux ##

## Fit Status
t0 = dh[index0].x[0]
y0 = dh[index0].y[0]
## Parallel Energy Spectra
t1 = dh[index1].x[0]
v1 = dh[index1].y[0] 
y1 = dh[index1].v[0] 
## Orthogonal Energy Spectra
t2 = dh[index2].x[0]
v2 = dh[index2].y[0]
y2 = dh[index2].v[0]
## Anti-Parallel Energy Spectra
t3 = dh[index3].x[0]
v3 = dh[index3].y[0]
y3 = dh[index3].v[0]
## Amplitude: Fit Parameter A
t4 = dh[index4].x[0]
y4 = dh[index4].y[0] # y1[0,1,2] = dh[index].y[0][0,1,2] for para,perp...etc
## Powerlaw: Fit Parameter B
t5 = dh[index5].x[0]
y5 = dh[index5].y[0] # y1[0,1,2] = dh[index].y[0][0,1,2] for para,perp...etc
## Cutoff Energy: 1/|C|
t6 = dh[index6].x[0]
y6 = -1.0/(dh[index6].y[0]) # y1[0,1,2] = dh[index].y[0][0,1,2]
## Reduced Chi^2
t7 = dh[index7].x[0]
y7 = dh[index7].y[0] # y1[0,1,2] = dh[index].y[0][0,1,2]

if np.size(t0) != np.size(t1):
    print('Fit times and data times not the same shape')
    exit()

def nan_equal(a,b):
    try:
        np.testing.assert_equal(a,b)
    except AssertionError:
        return False
    return True
same = False
while same == False:
    same = nan_equal(t1,t2)
    if same == False:
        print('Error: t1 and t2 not equal')
        exit()
    same = nan_equal(t2,t3)
    if same == False:
        print('Error: t2 and t3 not equal') # if t1=t2 and t2=t3 then t1=t3
        exit()
    same = nan_equal(t0,t4)
    if same == False:
        print('Error: t0 and t4 not equal')
        exit()
    same = nan_equal(t0,t5)
    if same == False:
        print('Error: t0 and t5 not equal')
        exit()
    same = nan_equal(t0,t6)
    if same == False:
        print('Error: t0 and t6 not equal')
        exit()
    same = nan_equal(t0,t7)
    if same == False:
        print('Error: t0 and t7 not equal')
        exit()
print('All data and fit parameter time arrays equal','\n')

##########
## Good Fits
##########
### FIT FLAGS ARE 3D VECTORS SO THAT EACH DIRECTION CORRESPONDS TO THE PARALLEL/ORTHO/ANTI ###

good_indicies_par = np.asarray([])
good_indicies_orth = np.asarray([])
good_indicies_anti = np.asarray([])
for i in range(len(y0[0])):
    if y0[0][i] >= 1.0:
        good_indicies_par = np.append(good_indicies_par,i)  # good contains all of the time indicies that correspond to a 'good' fit
    if y0[1][i] >= 1.0:
        good_indicies_orth = np.append(good_indicies_orth,i)
    if y0[2][i] >= 1.0:
        good_indicies_anti = np.append(good_indicies_anti,i)
print('par',len(good_indicies_par))
print('orth',len(good_indicies_orth))
print('anti',len(good_indicies_anti))
good_indicies = np.union1d(good_indicies_par,good_indicies_orth)
good_indicies = np.union1d(good_indicies,good_indicies_anti)
print('good_indicies',len(good_indicies),'\n')
# print(np.max(good_indicies))
# print(np.min(good_indicies),np.max(good_indicies))
# print(np.min(good_indicies_par),np.max(good_indicies_par))
# print(np.min(good_indicies_orth),np.max(good_indicies_orth))
# print(np.min(good_indicies_anti),np.max(good_indicies_anti))


# for i in range(len())

time = np.asarray([])
for num in good_indicies:
    x = int(num)
    time = np.append(time,t0[x])

rows = np.size(y2[:,0])
y1_good = np.zeros((rows,np.size(good_indicies)))
y2_good = np.zeros((rows,np.size(good_indicies)))
y3_good = np.zeros((rows,np.size(good_indicies)))
v1_good = np.zeros((rows,np.size(good_indicies)))
v2_good = np.zeros((rows,np.size(good_indicies)))
v3_good = np.zeros((rows,np.size(good_indicies)))
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
    x = int(good_indicies[i]) ##n x = index of good fit
    A_par_good = np.append(A_par_good,y4[0][x])
    A_orth_good = np.append(A_orth_good,y4[1][x])
    A_anti_good = np.append(A_anti_good,y4[2][x])
    B_par_good = np.append(B_par_good,y5[0][x])
    B_orth_good = np.append(B_orth_good,y5[1][x])
    B_anti_good = np.append(B_anti_good,y5[2][x])
    C_par_good = np.append(C_par_good,y6[0][x])
    C_orth_good = np.append(C_orth_good,y6[1][x])
    C_anti_good = np.append(C_anti_good,y6[2][x])
    chi_par_good = np.append(chi_par_good,y7[0][x])
    chi_orth_good = np.append(chi_orth_good,y7[1][x])
    chi_anti_good = np.append(chi_anti_good,y7[2][x])
    y1_good[:,i] = y1[:,x]
    y2_good[:,i] = y2[:,x]
    y3_good[:,i] = y3[:,x]
    v1_good[:,i] = v1[:,x]
    v2_good[:,i] = v2[:,x]
    v3_good[:,i] = v3[:,x]
    # if i == 115:
    #     print(x)

##########
## Put Time into UTC
##########
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
time_utc = unix_to_utc(time)

# exit()

##########
## Create Plot
##########
minE = 20 #eV
maxE = 5*10**3 #eV
E = np.linspace(minE,maxE,len(time))
print('Making',np.size(time),'plots','\n')
for i in range(len(time)):
    print('Plot %d'%i)
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(1,1,1)
    if m.isnan(chi_par_good[i]) == False:
        ax.loglog(y1_good[:,i],v1_good[:,i],'ro',markersize=2,label='Red. $\chi ^{2}$ = %.3f'%chi_par_good[i])
    else:
        ax.loglog(y1_good[:,i],v1_good[:,i],'ro',markersize=2)
    if m.isnan(chi_orth_good[i]) == False:
        ax.loglog(y2_good[:,i],v2_good[:,i],'go',markersize=2,label='Red. $\chi ^{2}$ = %.3f'%chi_orth_good[i])
    else:
        ax.loglog(y2_good[:,i],v2_good[:,i],'go',markersize=2)
    if m.isnan(chi_anti_good[i]) == False:
        ax.loglog(y3_good[:,i],v3_good[:,i],'bo',markersize=2,label='Red. $\chi ^{2}$ = %.3f'%chi_anti_good[i])
    else:
        ax.loglog(y3_good[:,i],v3_good[:,i],'bo',markersize=2)
    ax.axvline(1./np.absolute(C_par_good[i]),c='r',ls='--',lw = 1)
    ax.axvline(1./np.absolute(C_orth_good[i]),c='g',ls='--',lw = 1)
    ax.axvline(1./np.absolute(C_anti_good[i]),c='b',ls='--',lw = 1)
    f_par = A_par_good[i]*E**B_par_good[i]*np.exp(C_par_good[i]*E)
    f_orth = A_orth_good[i]*E**B_orth_good[i]*np.exp(C_orth_good[i]*E)
    f_anti = A_anti_good[i]*E**B_anti_good[i]*np.exp(C_anti_good[i]*E)
    ax.loglog(E,f_par,'r')
    ax.loglog(E,f_orth,'g')
    ax.loglog(E,f_anti,'b')
    ax.set_title(plot_title + ' Time = %s'%(time_utc[i]))
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Flux (#$cm^{-2}s^{-1}sr^{-1}eV^{-1}$)')
    ax.set_xlim(10,30000)
    ax.set_ylim(10,5*10**6)
    ax.legend(loc='lower left',frameon=True)
    fig.subplots_adjust(left=0.16, bottom=0.15, right=0.96, top=0.91, wspace=0.2, hspace=0.09)
    # plt.show()
    plt.savefig(path + date + savedir + 'Fit_%d.png'%i)
    plt.close()

# if date == '07_14_2008/':
#     good_fits = np.asarray([])
#     good_fits1 = np.asarray([])
#     good_fits2 = np.asarray([])
#     good_fits = np.append(good_fits,good_fits1)
#     good_fits = np.append(good_fits,good_fits2)
# elif date == '08_19_2008/':
#     good_fits = np.asarray([])
#     good_fits1 = np.asarray([])
#     good_fits2 = np.asarray([])
#     good_fits = np.append(good_fits,good_fits1)
#     good_fits = np.append(good_fits,good_fits2)
# elif date == '09_08_2008/':
#     good_fits = np.asarray([])
#     good_fits1 = np.asarray([])
#     good_fits2 = np.asarray([])
#     good_fits = np.append(good_fits,good_fits1)
#     good_fits = np.append(good_fits,good_fits2)
# elif date == '09_16_2008/':
#     good_fits = np.asarray([])
#     good_fits1 = np.asarray([])
#     good_fits2 = np.asarray([])
#     good_fits = np.append(good_fits,good_fits1)
#     good_fits = np.append(good_fits,good_fits2)
# else:
#     print('Not one of analyzed THEMIS dates')
#     exit()


# good_fits = np.asarray([8,12,16,18,19,22,30,49,77,95,105,116,117,122,123,124,127,129,183,203])
# good_fits1 = np.asarray([222,260,273,316,334,335,336,337,344,345,348,353,390,393,397,400,407])
# good_fits2 = np.asarray([413,415,416,418,464,471,472,478,519,541,577,638,647,653,655,674])

# good_fits = np.append(good_fits,good_fits1)
# good_fits = np.append(good_fits,good_fits2)
# # print(np.shape(good_fits))

# t1 = dh[41].x[0] - 1215993600
# B_mag = dh[41].y[0]
# t2 = dh[29].x[0] - 1215993600
# B_x = dh[29].y[0][0]
# B_y = dh[29].y[0][1]
# B_z = dh[29].y[0][2]
# t3 = dh[89].x[0] - 1215993600
# n_i = dh[89].y[0]
# t4 = dh[103].x[0] - 1215993600
# vi_x = dh[103].y[0][0]
# vi_y = dh[103].y[0][1]
# vi_z = dh[103].y[0][2]
# t5 = dh[75].x[0] - 1215993600
# T_e = dh[75].y[0]
# t6 = dh[90].x[0] - 1215993600
# T_i = dh[90].y[0]
# # for i in range(5):
# # for i in range(len(good_fits)):
# for i in range(len(time_good)):
#     # x = good_fits[i]
#     xmin = time_good[i] - 15 ## Change back to x
#     xmax = time_good[i] + 15 ## Change back to x
#     fig = plt.figure(figsize=(5,6))
#     ax1 = fig.add_subplot(6,1,1)
#     ax2 = fig.add_subplot(6,1,2)
#     ax3 = fig.add_subplot(6,1,3)
#     ax4 = fig.add_subplot(6,1,4)
#     ax5 = fig.add_subplot(6,1,5)
#     ax6 = fig.add_subplot(6,1,6)
    
#     ax1.plot(t1,B_mag,'k',lw=0.5)
#     ax1.set_ylabel('$|\mathbf{B}_{o}| (nT)$')
#     # plt.setp(ax1.get_xticklabels(), visible=False) #Share x-axis
#     ax1.set_xlim(xmin,xmax)
#     t1_new = np.asarray([])
#     for t in range(len(t1)):
#         if t1[t] >= xmin and t1[t] <= xmax:
#             t1_new = np.append(t1_new,t1[t])   
#     start = list(t1).index(np.nanmin(t1_new))
#     stop = list(t1).index(np.nanmax(t1_new))
#     ax1.set_ylim(np.nanmin(B_mag[start:stop]), np.nanmax(B_mag[start:stop]))
#     ax1.axvline(time_good[x],c='k',ls='--',lw=0.5)
    
#     ax2.plot(t2,B_x,'r',lw=0.5,label='X')
#     ax2.plot(t2,B_y,'g',lw=0.5,label='Y')
#     ax2.plot(t2,B_z,'b',lw=0.5, label ='Z')
#     # ax2.legend(loc='lower left', frameon=True,prop={'size': 6})
#     ax2.set_ylabel('$\mathbf{B}_{o} (nT)$')
#     # plt.setp(ax2.get_xticklabels(), visible=False) #Share x-axis
#     ax2.set_xlim(xmin,xmax)
#     t2_new = np.asarray([])
#     for t in range(len(t2)):
#         if t2[t] >= xmin and t2[t] <= xmax:
#             t2_new = np.append(t2_new,t2[t])   
#     start = list(t2).index(np.nanmin(t2_new))
#     stop = list(t2).index(np.nanmax(t2_new))
#     mins = np.asarray([np.nanmin(B_x[start:stop]),np.nanmin(B_y[start:stop]),np.nanmin(B_z[start:stop])])
#     maxs = np.asarray([np.nanmax(B_x[start:stop]),np.nanmax(B_y[start:stop]),np.nanmax(B_z[start:stop])])
#     min_y = np.nanmin(mins)
#     max_y = np.nanmax(maxs)
#     ax2.set_ylim(min_y, max_y)
#     ax2.axvline(time_good[x],c='k',ls='--',lw=0.5)
    
#     # ax3.semilogy(t3,n_i,'k',lw=0.5)
#     ax3.plot(t3,n_i,'k',lw=0.5)
#     ax3.set_ylabel('$n_{i}\ (cm^{-3})$')
#     # plt.setp(ax3.get_xticklabels(), visible=False) #Share x-axis
#     ax3.set_xlim(xmin,xmax)
#     t3_new = np.asarray([])
#     for t in range(len(t3)):
#         if t3[t] >= xmin and t3[t] <= xmax:
#             t3_new = np.append(t3_new,t3[t])   
#     start = list(t3).index(np.nanmin(t3_new))
#     stop = list(t3).index(np.nanmax(t3_new))
#     ax3.set_ylim(np.nanmin(n_i[start:stop]), np.nanmax(n_i[start:stop]))
#     ax3.axvline(time_good[x],c='k',ls='--',lw=0.5)
    
#     ax4.plot(t4,vi_x,'r',lw=0.5,label='X')
#     ax4.plot(t4,vi_y,'g',lw=0.5,label='Y')
#     ax4.plot(t4,vi_z,'b',lw=0.5,label='Z')
#     ax4.legend(loc='best', frameon=True,prop={'size': 6})
#     ax4.set_ylabel('$\mathbf{V}_{bulk} $(km/s)')
#     # plt.setp(ax4.get_xticklabels(), visible=False) #Share x-axis
#     ax4.set_xlim(xmin,xmax)
#     t4_new = np.asarray([])
#     for t in range(len(t4)):
#         if t4[t] >= xmin and t4[t] <= xmax:
#             t4_new = np.append(t4_new,t4[t])   
#     start = list(t4).index(np.nanmin(t4_new))
#     stop = list(t4).index(np.nanmax(t4_new))
#     mins = np.asarray([np.nanmin(vi_x[start:stop]),np.nanmin(vi_y[start:stop]),np.nanmin(vi_z[start:stop])])
#     maxs = np.asarray([np.nanmax(vi_x[start:stop]),np.nanmax(vi_y[start:stop]),np.nanmax(vi_z[start:stop])])
#     min_y = np.nanmin(mins)
#     max_y = np.nanmax(maxs)
#     ax4.set_ylim(min_y, max_y)
#     ax4.axvline(time_good[x],c='k',ls='--',lw=0.5)
    
#     # ax5.semilogy(t5,T_e,'k',lw=0.5)
#     ax5.plot(t5,T_e,'k',lw=0.5)
#     ax5.set_ylabel('$T_{e}$ (eV)')
#     ax5.set_xlim(xmin,xmax)
#     t5_new = np.asarray([])
#     for t in range(len(t5)):
#         if t5[t] >= xmin and t5[t] <= xmax:
#             t5_new = np.append(t5_new,t5[t])   
#     start = list(t5).index(np.nanmin(t5_new))
#     stop = list(t5).index(np.nanmax(t5_new))
#     ax5.set_ylim(np.nanmin(T_e[start:stop]), np.nanmax(T_e[start:stop]))
#     ax5.axvline(time_good[x],c='k',ls='--',lw=0.5)
    
#     # ax6.semilogy(t6,T_i,'k',lw=0.5)
#     ax6.plot(t6,T_i,'k',lw=0.5)
#     ax6.set_ylabel('$T_{i}$ (eV)')
#     ax6.set_xlabel('Time From Start of Date (s)')
#     ax6.set_xlim(xmin,xmax)
#     t6_new = np.asarray([])
#     for t in range(len(t6)):
#         if t6[t] >= xmin and t6[t] <= xmax:
#             t6_new = np.append(t6_new,t6[t])   
#     start = list(t6).index(np.nanmin(t6_new))
#     stop = list(t6).index(np.nanmax(t6_new))
#     ax6.set_ylim(np.nanmin(T_i[start:stop]), np.nanmax(T_i[start:stop]))
#     ax6.axvline(time_good[x],c='k',ls='--',lw=0.5)
    
#     fig.suptitle('Themis C: 2008-07-14 - Fit With Data Index %d'%x)
#     # fig.suptitle('Themis C: 2008-07-14 Plot 2 - %s Area of Interest'%name)
#     fig.subplots_adjust(left=0.2, bottom=0.1, right=0.96, top=0.93, wspace=0.2, hspace=0.08)
#     # plt.savefig('/Users/Tyler/Documents/NASA/Summer2018/THEMIS/07_14_2008/Plots/Fits_with_data/Select_fits/Index_%d.png'%x)
#     plt.savefig('/Users/Tyler/Documents/NASA/Summer2018/THEMIS/07_14_2008/Plots/Fits_with_data/Remaining_fits/Index_%d.png'%x)

# ## Magnetic Field Magnitude
# t1 = dh[41].x[0] - 1215993600
# y1 = dh[41].y[0]
# ## EFP Plot
# t2 = dh[248].x[0] - 1215993600
# y2_x = dh[248].y[0][0]
# y2_y = dh[248].y[0][1]
# y2_z = dh[248].y[0][2]
# ## EFW Plot
# t3 = dh[242].x[0] - 1215993600
# y3_x = dh[242].y[0][0]
# y3_y = dh[242].y[0][1]
# y3_z = dh[242].y[0][2]
# ## SCP Plot
# t4 = dh[251].x[0] - 1215993600
# y4_x = dh[251].y[0][0]
# y4_y = dh[251].y[0][1]
# y4_z = dh[251].y[0][2]
# ## SCW Plot
# t5 = dh[253].x[0] - 1215993600
# y5_x = dh[253].y[0][0]
# y5_y = dh[253].y[0][1]
# y5_z = dh[253].y[0][2]

# # for i in range(5):
# # for i in range(len(good_fits)):
# #     x = good_fits[i]
# #     xmin = time_good[x] - 2
# #     xmax = time_good[x] + 2
# #     fig = plt.figure(figsize=(7,6.5))
# #     ax1 = fig.add_subplot(5,1,1)
# #     ax2 = fig.add_subplot(5,1,2)
# #     ax3 = fig.add_subplot(5,1,3)
# #     ax4 = fig.add_subplot(5,1,4)
# #     ax5 = fig.add_subplot(5,1,5)
# #     
# #     ax1.plot(t1,y1,'k',lw=0.5)
# #     ax1.set_ylabel('$|\mathbf{B}_{o}| (nT)$',fontsize=8)
# #     # plt.setp(ax1.get_xticklabels(), visible=False) #Share x-axis
# #     ax1.set_xlim(xmin,xmax)
# #     t1_new = np.asarray([])
# #     for t in range(len(t1)):
# #         if t1[t] >= xmin and t1[t] <= xmax:
# #             t1_new = np.append(t1_new,t1[t])   
# #     start = list(t1).index(np.nanmin(t1_new))
# #     stop = list(t1).index(np.nanmax(t1_new))
# #     ax1.set_ylim(np.nanmin(y1[start:stop]), np.nanmax(y1[start:stop]))
# #     ax1.axvline(time_good[x],c='k',ls='--',lw=0.5)
# #     
# #     ax2.plot(t2,y2_x,'r',lw=0.5,label='X')
# #     ax2.plot(t2,y2_y,'g',lw=0.5,label='Y')
# #     ax2.plot(t2,y2_z,'b',lw=0.5, label ='Z')
# #     # ax2.legend(loc='lower left', frameon=True,prop={'size': 6})
# #     ax2.set_ylabel('EFP (mV/m)',fontsize=8)
# #     # plt.setp(ax2.get_xticklabels(), visible=False) #Share x-axis
# #     ax2.set_xlim(xmin,xmax)
# #     t2_new = np.asarray([])
# #     for t in range(len(t2)):
# #         if t2[t] >= xmin and t2[t] <= xmax:
# #             t2_new = np.append(t2_new,t2[t])   
# #     start = list(t2).index(np.nanmin(t2_new))
# #     stop = list(t2).index(np.nanmax(t2_new))
# #     mins = np.asarray([np.nanmin(y2_x[start:stop]),np.nanmin(y2_y[start:stop]),np.nanmin(y2_z[start:stop])])
# #     maxs = np.asarray([np.nanmax(y2_x[start:stop]),np.nanmax(y2_y[start:stop]),np.nanmax(y2_z[start:stop])])
# #     min_y = np.nanmin(mins)
# #     max_y = np.nanmax(maxs)
# #     ax2.set_ylim(min_y, max_y)
# #     ax2.axvline(time_good[x],c='k',ls='--',lw=0.5)
# #     
# #     ax3.plot(t3,y3_x,'r',lw=0.1,label='X')
# #     ax3.plot(t3,y3_y,'g',lw=0.1,label='Y')
# #     ax3.plot(t3,y3_z,'b',lw=0.1,label='Z')
# #     ax3.set_ylabel('EFW (mV/m)',fontsize=8)
# #     # plt.setp(ax3.get_xticklabels(), visible=False) #Share x-axis
# #     ax3.set_xlim(xmin,xmax)
# #     t3_new = np.asarray([])
# #     # for t in range(len(t3)):
# #     #     if t3[t] >= xmin and t3[t] <= xmax:
# #     #         t3_new = np.append(t3_new,t3[t])   
# #     # start = list(t3).index(np.nanmin(t3_new))
# #     # stop = list(t3).index(np.nanmax(t3_new))
# #     # mins = np.asarray([np.nanmin(y3_x[start:stop]),np.nanmin(y3_y[start:stop]),np.nanmin(y3_z[start:stop])])
# #     # maxs = np.asarray([np.nanmax(y3_x[start:stop]),np.nanmax(y3_y[start:stop]),np.nanmax(y3_z[start:stop])])
# #     # min_y = np.nanmin(mins)
# #     # max_y = np.nanmax(maxs)
# #     # ax3.set_ylim(min_y, max_y)
# #     ax3.axvline(time_good[x],c='k',ls='--',lw=0.5)
# #     
# #     ax4.plot(t4,y4_x,'r',lw=0.5,label='X')
# #     ax4.plot(t4,y4_y,'g',lw=0.5,label='Y')
# #     ax4.plot(t4,y4_z,'b',lw=0.5,label='Z')
# #     # ax4.legend(loc='best', frameon=True,prop={'size': 6})
# #     ax4.set_ylabel('SCP (nT)',fontsize=8)
# #     # plt.setp(ax4.get_xticklabels(), visible=False) #Share x-axis
# #     ax4.set_xlim(xmin,xmax)
# #     t4_new = np.asarray([])
# #     for t in range(len(t4)):
# #         if t4[t] >= xmin and t4[t] <= xmax:
# #             t4_new = np.append(t4_new,t4[t])   
# #     start = list(t4).index(np.nanmin(t4_new))
# #     stop = list(t4).index(np.nanmax(t4_new))
# #     mins = np.asarray([np.nanmin(y4_x[start:stop]),np.nanmin(y4_y[start:stop]),np.nanmin(y4_z[start:stop])])
# #     maxs = np.asarray([np.nanmax(y4_x[start:stop]),np.nanmax(y4_y[start:stop]),np.nanmax(y4_z[start:stop])])
# #     min_y = np.nanmin(mins)
# #     max_y = np.nanmax(maxs)
# #     ax4.set_ylim(min_y, max_y)  
# #     ax4.axvline(time_good[x],c='k',ls='--',lw=0.5)  
# #     
# #     ax5.plot(t5,y5_x,'r',lw=0.1,label='X')
# #     ax5.plot(t5,y5_y,'g',lw=0.1,label='Y')
# #     ax5.plot(t5,y5_z,'b',lw=0.1,label='Z')
# #     # ax5.legend(loc='lower left', frameon=True,prop={'size': 6})
# #     ax5.set_ylabel('SCW (nT)',fontsize=8)
# #     # plt.setp(ax5.get_xticklabels(), visible=False) #Share x-axis
# #     ax5.set_xlabel('Time From Start of Date (s)')
# #     ax5.set_xlim(xmin,xmax)
# #     # t5_new = np.asarray([])
# #     # for t in range(len(t5)):
# #     #     if t5[t] >= xmin and t5[t] <= xmax:
# #     #         t5_new = np.append(t5_new,t5[t])   
# #     # start = list(t5).index(np.nanmin(t5_new))
# #     # stop = list(t5).index(np.nanmax(t5_new))
# #     # mins = np.asarray([np.nanmin(y5_x[start:stop]),np.nanmin(y5_y[start:stop]),np.nanmin(y5_z[start:stop])])
# #     # maxs = np.asarray([np.nanmax(y5_x[start:stop]),np.nanmax(y5_y[start:stop]),np.nanmax(y5_z[start:stop])])
# #     # min_y = np.nanmin(mins)
# #     # max_y = np.nanmax(maxs)
# #     # ax5.set_ylim(min_y, max_y)
# #     ax5.axvline(time_good[x],c='k',ls='--',lw=0.5)
# #     
# #     fig.suptitle('Themis C: 2008-07-14 - Fit With Data Index %d'%x)
# #     fig.subplots_adjust(left=0.12, bottom=0.08, right=0.96, top=0.9, wspace=0.2, hspace=0.2)
# #     plt.savefig('/Users/Tyler/Documents/NASA/Summer2018/THEMIS/07_14_2008/Plots/Fits_with_data/Select_fits/Detail/Index_%d_detail.png'%x)

