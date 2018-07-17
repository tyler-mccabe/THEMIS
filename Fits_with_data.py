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
plot_fits = False
select_best = False
plot_analysis1 = False
plot_analysis2 = True
slice_by = 'efw'

path = '/Users/tylermccabe/Documents/NASA/Summer2018/THEMIS/'
date = '09_16_2008/'

tplot = themis + '.tplot'
savedir = 'Plots/' + themis + '/Fits/'
move_savedir = 'Plots/' + themis + '/Fits/Best/'
analysis_savedir = 'Plots/' + themis + '/Fit_Analysis/'
plot_title = themis + ': ' + date[:-1]

file = scipy.io.readsav(path+date+tplot,python_dict=True)
dq = file['dq']
dh = dq['dh']
print('\n')
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
# print('par',len(good_indicies_par))
# print('orth',len(good_indicies_orth))
# print('anti',len(good_indicies_anti))
good_indicies = np.union1d(good_indicies_par,good_indicies_orth)
good_indicies = np.union1d(good_indicies,good_indicies_anti)
# print('good_indicies',len(good_indicies),'\n')

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
## Create Plots
##########
if plot_fits == True:
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

############################################################################
############################# Analysis Plots ###############################
############################################################################

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
if select_best == True:
    minE = 20 #eV
    maxE = 5*10**3 #eV
    E = np.linspace(minE,maxE,len(time))
    print('Making',np.size(good_fits),'plots','\n')
    for i in range(len(good_fits)):
        x = good_fits[i]
        print('Plot %d'%x)
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(1,1,1)
        if m.isnan(chi_par_good[x]) == False:
            ax.loglog(y1_good[:,x],v1_good[:,x],'ro',markersize=2,label='Red. $\chi ^{2}$ = %.3f'%chi_par_good[x])
        else:
            ax.loglog(y1_good[:,x],v1_good[:,x],'ro',markersize=2)
        if m.isnan(chi_orth_good[x]) == False:
            ax.loglog(y2_good[:,x],v2_good[:,x],'go',markersize=2,label='Red. $\chi ^{2}$ = %.3f'%chi_orth_good[x])
        else:
            ax.loglog(y2_good[:,x],v2_good[:,x],'go',markersize=2)
        if m.isnan(chi_anti_good[x]) == False:
            ax.loglog(y3_good[:,x],v3_good[:,x],'bo',markersize=2,label='Red. $\chi ^{2}$ = %.3f'%chi_anti_good[x])
        else:
            ax.loglog(y3_good[:,x],v3_good[:,x],'bo',markersize=2)
        ax.axvline(1./np.absolute(C_par_good[x]),c='r',ls='--',lw = 1)
        ax.axvline(1./np.absolute(C_orth_good[x]),c='g',ls='--',lw = 1)
        ax.axvline(1./np.absolute(C_anti_good[x]),c='b',ls='--',lw = 1)
        f_par = A_par_good[x]*E**B_par_good[x]*np.exp(C_par_good[x]*E)
        f_orth = A_orth_good[x]*E**B_orth_good[x]*np.exp(C_orth_good[x]*E)
        f_anti = A_anti_good[x]*E**B_anti_good[x]*np.exp(C_anti_good[x]*E)
        ax.loglog(E,f_par,'r')
        ax.loglog(E,f_orth,'g')
        ax.loglog(E,f_anti,'b')
        ax.set_title(plot_title + ' Time = %s'%(time_utc[x]))
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Flux (#$cm^{-2}s^{-1}sr^{-1}eV^{-1}$)')
        ax.set_xlim(10,30000)
        ax.set_ylim(10,5*10**6)
        ax.legend(loc='lower left',frameon=True)
        fig.subplots_adjust(left=0.16, bottom=0.15, right=0.96, top=0.91, wspace=0.2, hspace=0.09)
        # plt.show()
        plt.savefig(path + date + move_savedir + 'Fit_%d.png'%x)
        plt.close()



##########
## Get Data
##########
index1 = get_index('fgh_mag')      ## Magnetic Field Mag  
index2 = get_index('fgh_gse')      ## Magnetic Field Vector   
index3 = get_index('peeb_density') ## Electron Density Burst 
index4 = get_index('peeb_avgtemp') ## Electron Temp Burst    
index5 = get_index('peib_density') ## Ion Density Burst       
index6 = get_index('peib_avgtemp') ## Ion Temp Burst     

t1 = dh[index1].x[0]
y1 = dh[index1].y[0]
t2 = dh[index2].x[0]
y2_x = dh[index2].y[0][0]
y2_y = dh[index2].y[0][1]
y2_z = dh[index2].y[0][2]
t3 = dh[index3].x[0]
y3 = dh[index3].y[0]
t4 = dh[index4].x[0]
y4 = dh[index4].y[0]
t5 = dh[index5].x[0]
y5 = dh[index5].y[0]
t6 = dh[index6].x[0]
y6= dh[index6].y[0]


##########
## Create Plots
##########

def get_begin_end_indicies(time_array,start_time,stop_time):
    '''Get slice indicies for each of the variables to plot'''
    time = np.asarray([])
    for t in range(len(time_array)):
        if time_array[t] >= start_time and time_array[t] <= stop_time:
            time = np.append(time,time_array[t])
    begin_index = list(time_array).index(np.nanmin(time))
    end_index = list(time_array).index(np.nanmax(time))
    return begin_index,end_index
if plot_analysis1 == True:
    print('Making',np.size(good_fits),'plots for',date[:-1], '\n')
    for i in range(len(good_fits)):
        x = good_fits[i]
        print('Plot %d'%x)
        xmin = time[x] - 30 ## Change back to x
        xmax = time[x] + 30 ## Change back to x
        fig = plt.figure(figsize=(5,7))
        ax1 = fig.add_subplot(6,1,1)
        ax2 = fig.add_subplot(6,1,2,sharex=ax1)
        ax3 = fig.add_subplot(6,1,3,sharex=ax1)
        ax4 = fig.add_subplot(6,1,4,sharex=ax1)
        ax5 = fig.add_subplot(6,1,5,sharex=ax1)
        ax6 = fig.add_subplot(6,1,6,sharex=ax1)
        
        begin, end = get_begin_end_indicies(t1, xmin, xmax)
        ax1.plot(t1,y1,'k',lw=0.5)
        ax1.set_ylabel('$|\mathbf{B}_{o}| (nT)$')
        plt.setp(ax1.get_xticklabels(), visible=False) #Share x-axis
        ax1.set_xlim(xmin,xmax)
        ax1.set_ylim(np.nanmin(y1[begin:end])*0.8, np.nanmax(y1[begin:end])*1.1)
        ax1.axvline(time[x],c='k',ls='--',lw=0.5)
        
        begin, end = get_begin_end_indicies(t2, xmin, xmax)
        ax2.plot(t2,y2_x,'r',lw=0.5,label='X')
        ax2.plot(t2,y2_y,'g',lw=0.5,label='Y')
        ax2.plot(t2,y2_z,'b',lw=0.5, label ='Z')
        # ax2.legend(loc='lower left', frameon=True,prop={'size': 6})
        ax2.set_ylabel('$\mathbf{B}_{o} (nT)$')
        plt.setp(ax2.get_xticklabels(), visible=False) #Share x-axis
        ax2.set_xlim(xmin,xmax)
        mins = np.asarray([np.nanmin(y2_x[begin:end]),np.nanmin(y2_y[begin:end]),np.nanmin(y2_z[begin:end])])
        maxs = np.asarray([np.nanmax(y2_x[begin:end]),np.nanmax(y2_y[begin:end]),np.nanmax(y2_z[begin:end])])
        min_y = np.nanmin(mins)
        max_y = np.nanmax(maxs)
        ax2.set_ylim(min_y, max_y)
        ax2.axvline(time[x],c='k',ls='--',lw=0.5)
        
        begin, end = get_begin_end_indicies(t3, xmin, xmax)
        ax3.plot(t3,y3,'k',lw=0.5)
        ax3.set_ylabel('$n_{e}\ (cm^{-3})$')
        ax3.set_xlim(xmin,xmax)
        ax3.set_ylim(np.nanmin(y3[begin:end]), np.nanmax(y3[begin:end]))
        diff = np.nanmax(y3[begin:end]) - np.nanmin(y3[begin:end])
        if diff >= 800:
            ax3.cla()
            ax3.semilogy(t3,y3,'k',lw=0.5)
            ax3.set_ylabel('$n_{e}\ (cm^{-3})$')
            ax3.set_xlim(xmin,xmax)
            ax3.set_ylim(np.nanmin(y3[begin:end])*0.8, np.nanmax(y3[begin:end])*1.1)
        else:
            pass
        ax3.axvline(time[x],c='k',ls='--',lw=0.5)
        plt.setp(ax3.get_xticklabels(), visible=False) #Share x-axis
        
        begin, end = get_begin_end_indicies(t4, xmin, xmax)
        ax4.plot(t4,y4,'k',lw=0.5)
        ax4.set_ylabel('$T_{e}\ (eV)$')
        ax4.set_xlim(xmin,xmax)
        ax4.set_ylim(np.nanmin(y4[begin:end])*0.8, np.nanmax(y4[begin:end])*1.1)
        diff = np.nanmax(y4[begin:end]) - np.nanmin(y4[begin:end])
        if diff >= 800:
            ax4.cla()
            ax4.semilogy(t4,y4,'k',lw=0.5)
            ax4.set_ylabel('$T_{e}$ (eV)')
            ax4.set_xlim(xmin,xmax)
            ax4.set_ylim(np.nanmin(y4[begin:end])*0.8, np.nanmax(y4[begin:end])*1.1)
        else:
            pass
        ax4.axvline(time[x],c='k',ls='--',lw=0.5)
        plt.setp(ax4.get_xticklabels(), visible=False) #Share x-axis

        begin, end = get_begin_end_indicies(t5, xmin, xmax)
        ax5.plot(t5,y5,'k',lw=0.5)
        ax5.set_ylabel('$n_{i}\ (cm^{-3})$')
        ax5.set_xlim(xmin,xmax)
        ax5.set_ylim(np.nanmin(y5[begin:end]), np.nanmax(y5[begin:end]))
        diff = np.nanmax(y5[begin:end]) - np.nanmin(y5[begin:end])
        if diff >= 800:
            ax5.cla()
            ax5.semilogy(t5,y5,'k',lw=0.5)
            ax5.set_ylabel('$n_{i}\ (cm^{-3})$')
            ax5.set_xlim(xmin,xmax)
            ax5.set_ylim(np.nanmin(y5[begin:end])*0.8, np.nanmax(y5[begin:end])*1.1)
        else:
            pass
        ax5.axvline(time[x],c='k',ls='--',lw=0.5)
        plt.setp(ax5.get_xticklabels(), visible=False) #Share x-axis

        begin, end = get_begin_end_indicies(t6, xmin, xmax)
        ax6.plot(t6,y6,'k',lw=0.5)
        ax6.set_ylabel('$T_{i}$ (eV)')
        ax6.set_xlabel('Time')
        ax6.set_xlim(xmin,xmax)
        ax6.set_ylim(np.nanmin(y6[begin:end])*0.8, np.nanmax(y6[begin:end])*1.1)
        diff = np.nanmax(y6[begin:end]) - np.nanmin(y6[begin:end])
        if diff >= 800:
            ax6.cla()
            ax6.semilogy(t6,y6,'k',lw=0.5)
            ax6.set_ylabel('$T_{i}$ (eV)')
            ax6.set_xlim(xmin,xmax)
            ax6.set_ylim(np.nanmin(y6[begin:end])*0.8, np.nanmax(y6[begin:end])*1.1)
        else:
            pass
        ax6.axvline(time[x],c='k',ls='--',lw=0.5)
        ax6.set_xticklabels(unix_to_utc(ax6.get_xticks()))
        plt.setp(ax6.get_xticklabels(), rotation=30, horizontalalignment='right')
        
        fig.suptitle(plot_title + ' Time = %s'%(time_utc[x]))
        fig.subplots_adjust(left=0.2, bottom=0.1, right=0.96, top=0.93, wspace=0.2, hspace=0.08)
        plt.savefig(path + date + analysis_savedir + 'figure_%d.png'%(x))
        plt.close()

if plot_analysis2 == True:
    index1 = get_index('fgh_mag')      ## Magnetic Field Mag 
    index2 = get_index('efp_l1_cal_tdas_corrected_rmspikes_dsl')  ## DC-coupled electric field vectors  
    index3 = get_index('efw_l1_cal_tdas_corrected_rmspikes_dsl')  ## AC-coupled electric field vectors
    index4 = get_index('scp_l1_cal_NoCutoff_dsl') ## DC-coupled magnetic field vectors from the search coil magnetometer data 
    index5 = get_index('scw_l1_cal_NoCutoff_dsl') ## AC-coupled magnetic field vectors from the burst search coil magnetometer data    
    index6 = get_index('peeb_red_chisq_good') ## Reduced Chi-Squared values         
    # print('\n')
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
    if slice_by == 'fgh_mag':
        slice_array = y1
        slice_time = t1
    elif slice_by == 'efp':
        slice_array = y2_x
        slice_time = t2
    elif slice_by == 'efw':
        slice_array = y3_x
        slice_time = t3
    elif slice_by == 'scp':
        slice_array = y4_x
        slice_time = t4
    elif slice_by == 'scw':
        slice_array = y5_x
        slice_time = t5
    elif slice_by == 'chi':
        slice_array = y6_x
        slice_time = t6
    else:
        print("Slice array not valid")
        exit()
    print('Slicing by',slice_by)
    min_index1 = np.asarray([])
    max_index1 = np.asarray([])

    for i in range(len(slice_array)-1):
        if m.isnan(slice_array[i]) == True and m.isnan(slice_array[i+1]) == False:
            min_index1 = np.append(min_index1,i)
        if m.isnan(slice_array[i]) == False and m.isnan(slice_array[i+1]) == True:
            max_index1 = np.append(max_index1,i+1)
    if len(min_index1) > len(max_index1):
        max_index1 = np.append(max_index1,len(slice_array)-1)
    elif len(min_index1) < 1:
        min_index1 = np.asarray([0])
        max_index1 = np.asarray([])
        for i in range(len(slice_time)-1):
            if (slice_time[i+1]-slice_time[i]) >= 10:
                min_index1 = np.append(min_index1,i+1)
                max_index1 = np.append(max_index1,i)
        max_index1 = np.append(max_index1,len(slice_time)-1)
    else:
        pass
    print('Making',np.size(good_fits),'plots','\n')

    ##############
    # Overall Plot Structure
    ##############
    for i in range(len(good_fits)):
        x = good_fits[i]
        print('Plot ',x)

        fig = plt.figure(figsize=(5.5,6.5))
        ax1 = fig.add_subplot(5,1,1)
        ax2 = fig.add_subplot(5,1,2,sharex=ax1)
        ax3 = fig.add_subplot(5,1,3,sharex=ax1)
        ax4 = fig.add_subplot(5,1,4,sharex=ax1)
        ax5 = fig.add_subplot(5,1,5,sharex=ax1)
        
        xmin = time[x] - 2
        xmax = time[x] + 2
        try:
            begin, end = get_begin_end_indicies(t1, xmin, xmax)
        except:
            ValueError
        ax1.plot(t1,y1,'k',lw=0.5)
        ax1.set_ylabel('$|\mathbf{B}_{o}| (nT)$',fontsize=8)
        try:
            ax1.set_xlim(xmin,xmax)
        except:
            ValueError
        try:
            ax1.set_ylim(np.nanmin(y1[begin:end])*0.8, np.nanmax(y1[begin:end])*1.1)
        except:
            ValueError
        ax1.set_xticklabels(unix_to_utc(ax1.get_xticks()))    
        plt.setp(ax1.get_xticklabels(), visible=False) #Share x-axis
        ax1.axvline(time[x],c='k',ls='--',lw=0.5)

        try:
            begin, end = get_begin_end_indicies(t2, xmin, xmax)
        except:
            ValueError
        ax2.plot(t2,y2_x,'r',lw=0.5,label='X')
        ax2.plot(t2,y2_y,'g',lw=0.5,label='Y')
        ax2.plot(t2,y2_z,'b',lw=0.5, label ='Z')
        ax2.set_ylabel('EFP (mV/m)',fontsize=8)
        try:
            mins = np.asarray([np.nanmin(y2_x[begin:end]),np.nanmin(y2_y[begin:end]),np.nanmin(y2_z[begin:end])])
            maxs = np.asarray([np.nanmax(y2_x[begin:end]),np.nanmax(y2_y[begin:end]),np.nanmax(y2_z[begin:end])])
            min_y = np.nanmin(mins)
            max_y = np.nanmax(maxs) * 1.1
            ax2.set_ylim(min_y, max_y) 
        except:
            ValueError
        try:
            ax2.set_xlim(xmin,xmax)
        except:
            ValueError
        ax2.set_xticklabels(unix_to_utc(ax2.get_xticks()))  
        ax2.axvline(time[x],c='k',ls='--',lw=0.5)  
        plt.setp(ax2.get_xticklabels(), visible=False) #Share x-axis
        
        try:
            begin, end = get_begin_end_indicies(t3, xmin, xmax)    
        except:
            ValueError
        ax3.plot(t3,y3_x,'r',lw=0.5,label='X')
        ax3.plot(t3,y3_y,'g',lw=0.5,label='Y')
        ax3.plot(t3,y3_z,'b',lw=0.5,label='Z')
        ax3.set_ylabel('EFW (mV/m)',fontsize=8)
        try:
            mins = np.asarray([np.nanmin(y3_x[begin:end]),np.nanmin(y3_y[begin:end]),np.nanmin(y3_z[begin:end])])
            maxs = np.asarray([np.nanmax(y3_x[begin:end]),np.nanmax(y3_y[begin:end]),np.nanmax(y3_z[begin:end])])
            min_y = np.nanmin(mins)
            max_y = np.nanmax(maxs) * 1.1
            ax3.set_ylim(min_y, max_y) 
        except:
            ValueError
        try:
            ax3.set_xlim(xmin,xmax)
        except:
            ValueError
        ax3.set_xticklabels(unix_to_utc(ax3.get_xticks()))   
        ax3.axvline(time[x],c='k',ls='--',lw=0.5) 
        plt.setp(ax3.get_xticklabels(), visible=False) #Share x-axis

        try:
            begin, end = get_begin_end_indicies(t4, xmin, xmax)    
        except:
            ValueError
        ax4.plot(t4,y4_x,'r',lw=0.5,label='X')
        ax4.plot(t4,y4_y,'g',lw=0.5,label='Y')
        ax4.plot(t4,y4_z,'b',lw=0.5,label='Z')
        ax4.set_ylabel('SCP (nT)',fontsize=8)
        try: 
            mins = np.asarray([np.nanmin(y4_x[begin:end]),np.nanmin(y4_y[begin:end]),np.nanmin(y4_z[begin:end])])
            maxs = np.asarray([np.nanmax(y4_x[begin:end]),np.nanmax(y4_y[begin:end]),np.nanmax(y4_z[begin:end])])
            min_y = np.nanmin(mins)
            max_y = np.nanmax(maxs) * 1.1
            ax4.set_ylim(min_y, max_y) 
        except:
            ValueError
        try:
            ax4.set_xlim(xmin,xmax)
        except:
            ValueError
        ax4.set_xticklabels(unix_to_utc(ax4.get_xticks()))
        ax4.axvline(time[x],c='k',ls='--',lw=0.5)
        plt.setp(ax4.get_xticklabels(), visible=False) #Share x-axis
        
        try:
            begin, end = get_begin_end_indicies(t5, xmin, xmax)
        except:
            ValueError
        ax5.plot(t5,y5_x,'r',lw=0.5,label='X')
        ax5.plot(t5,y5_y,'g',lw=0.5,label='Y')
        ax5.plot(t5,y5_z,'b',lw=0.5,label='Z')
        ax5.set_ylabel('SCW (nT)',fontsize=8)
        try:
            mins = np.asarray([np.nanmin(y5_x[begin:end]),np.nanmin(y5_y[begin:end]),np.nanmin(y5_z[begin:end])])
            maxs = np.asarray([np.nanmax(y5_x[begin:end]),np.nanmax(y5_y[begin:end]),np.nanmax(y5_z[begin:end])])
            min_y = np.nanmin(mins)
            max_y = np.nanmax(maxs) * 1.1
            ax5.set_ylim(min_y, max_y)
        except:
            ValueError
        try:
            ax5.set_xlim(xmin,xmax)
        except:
            ValueError
        ax5.set_xticklabels(unix_to_utc(ax5.get_xticks()))
        ax5.axvline(time[x],c='k',ls='--',lw=0.5)
        
        fig.suptitle(plot_title)
        plt.setp(ax5.get_xticklabels(), rotation=30, horizontalalignment='right')
        fig.subplots_adjust(left=0.15, bottom=0.1, right=0.97, top=0.94, wspace=0.2, hspace=0.18) #hspace = 0.38 with labels
        plt.savefig(path + date + analysis_savedir + slice_by +'_' + 'figure_%d.png'%(x))
        plt.close()

