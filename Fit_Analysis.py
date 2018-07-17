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
date = '07_14_2008/'

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
index1 = get_index('fgh_mag')      ## Magnetic Field Mag  
index2 = get_index('fgh_gse')      ## Magnetic Field Vector   
index3 = get_index('peeb_density') ## Electron Density Burst 
index4 = get_index('peeb_avgtemp') ## Electron Temp Burst    
index5 = get_index('peib_density') ## Ion Density Burst       
index6 = get_index('peib_avgtemp') ## Ion Temp Burst     

'''
For the fits created, I specifically looked for the few that had low reduced chi squared
values as well as those that had a larger than normal (~ few hundred eV). This provided 
a much larger sample than if the only criteria was a low reduced chisq. Also included were
those that had very low reduced chisq values for multiple directions.
'''
if themis == 'THB':
	good_fits = np.asarray([55,110,131,152,230,233,443,523,578,608,642,668,792,889,931])
    good_fits1 = np.asarray([996,1084,1100,1153,1192]) 
    good_fits = np.append(good_fits,good_fits1) # Overall cutoff energy lower than usual
elif themis == 'THC':
	if date == '07_14_2008/':
	    good_fits = np.asarray([0,7,11,15,76,207,208,209,218,219,220,221,226,228,237,273])
	    good_fits1 = np.asarray([379,381,386,562,570,574,593,594,596,598,600,602,603,604])
	    good_fits2 = np.asarray([609,676,690,701,705,713,751,833,865,866,949,1033,1051,1105])
	    good_fits3 = np.asarray([1109,1113,1118,1119,1125])
	    good_fits = np.append(good_fits,good_fits1)
	    good_fits = np.append(good_fits,good_fits2)
	    good_fits = np.append(good_fits,good_fits3)
	elif date == '08_19_2008/': ## Cutoff energies looked much lower overall
	    good_fits = np.asarray([16,46,112,114,148,169,182,199,200,212,242,262,281,287,305])
	    good_fits1 = np.asarray([348,362,375,386,390,392,400,401,403,404,415,419,434,437,440])
	    good_fits2 = np.asarray([452,504,506,521,637,694,704,790,802,828,870,875,881,918,921])
	    good_fits3 = np.asarray([936,947,948,958,960,966,994,1004,1033,1082,1089,1114,1137])
	    good_fits4 = np.asarray([1154,1188,1189])
	    good_fits = np.append(good_fits,good_fits1)
	    good_fits = np.append(good_fits,good_fits2)
	    good_fits = np.append(good_fits,good_fits3)
	    good_fits = np.append(good_fits,good_fits4)
	elif date == '09_08_2008/':
	    good_fits = np.asarray([36,64,69,70,72,80,82,84,85,90,94,95,96,97,98,114,196,254,256]) 
	    good_fits1 = np.asarray([299,328,372,411,431,432,451,453,540,571,598,603,606,625,659])
	    good_fits2 = np.asarray([668,697,698,701,702,717,829,838,843,844,855])
	    good_fits = np.append(good_fits,good_fits1)
	    good_fits = np.append(good_fits,good_fits2)
	elif date == '09_16_2008/':
	    good_fits = np.asarray([15,44,76,78,80,81,82,123,156,188,208,239,276,283,290,311,331])
	    good_fits1 = np.asarray([348,444,585,587,591,605,607,609,611,612,615,622,624,625,626])
	    good_fits2 = np.asarray([628,638,639,642,645,650,656,692,711,722,739,751,754,775,777])
	    good_fits3 = np.asarray([781,785,787,792,800,814,831,837,924,933,945,961,965,968,971])
	    good_fits4 = np.asarray([972,973,974,976,979,980,981,983,986,987,993,994,995,998,1001])
	    good_fits5 = np.asarray([1003,1006,1011,1016,1017,1023,1024,1037,1039,1041,1042,1047])
	    good_fits6 = np.asarray([1051,1053,1058,1061,1069,1071,1072,1076,1080,1084,1087,1090,1109])
	    good_fits = np.append(good_fits,good_fits1)
	    good_fits = np.append(good_fits,good_fits2)
	    good_fits = np.append(good_fits,good_fits3)
	    good_fits = np.append(good_fits,good_fits4)
	    good_fits = np.append(good_fits,good_fits5)
	    good_fits = np.append(good_fits,good_fits6)
	else:
	    print('Not one of analyzed THEMIS dates')
	    exit()
else:
	print('Unknown THEMIS spacecraft')
	exit()

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

