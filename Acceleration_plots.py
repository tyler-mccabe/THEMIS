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
date = '08_19_2008/'

tplot = themis + '.tplot'
savedir = 'Plots/' + themis + '/Fits/'

plot_title = themis + ': ' + date[:-1]

file = scipy.io.readsav(path+date+tplot,python_dict=True)
dq = file['dq']
dh = dq['dh']

event = 'SLAM_22_37_45'
before = [1091]
# during = [381,384]
after = [1071]

before_range = [1081,1091]
during_range = [1076,1080]
after_range = [1071,1075]
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
index8 = get_index('fgh_gse')  ## Magnetic Field Vector
index9 = get_index('fgh_mag') 

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
## Magnetic Field Vector
t8 = dh[index8].x[0]
y8_x = dh[index8].y[0][0]
y8_y = dh[index8].y[0][1]
y8_z = dh[index8].y[0][2]
print(np.shape(dh[index8].y[0]))
## Magnetic Field Magnitude
t9 = dh[index9].x[0]
y9 = dh[index9].y[0]

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
good_indicies = np.union1d(good_indicies_par,good_indicies_orth)
good_indicies = np.union1d(good_indicies,good_indicies_anti)

time = np.asarray([])
for num in good_indicies:
    x = int(num)
    time = np.append(time,t0[x])

y8_x = np.interp(t0,t8,y8_x)
y8_y = np.interp(t0,t8,y8_y)
y8_z = np.interp(t0,t8,y8_z)
rows = np.size(y2[:,0])
Bx = []
By = []
Bz = []
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
    x = int(good_indicies[i]) ## x = index of good fit
    Bx = np.append(Bx,y8_x[x])
    By = np.append(By,y8_y[x])
    Bz = np.append(Bz,y8_z[x])
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
B = np.vstack((Bx,By,Bz))
print('B',np.shape(B))
print(len(good_indicies))
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

##########
## Find Plot Time
##########
t_start = time[after[0]]
t_end = time[before[0]]

plot_time = []
plot2_time = []
for num in time:
	if num >= t_start and num <= t_end:
		plot_time = np.append(plot_time,num)
	if num >= time[after_range[0]] and num <= time[before_range[-1]]:
		plot2_time = np.append(plot2_time,num)
# print(plot_time)

##########
## Get Data for Average Energy Change Throughout TIFP
##########
trend = []
trend_par = []
trend_orth = []
trend_anti = []
median_par = []
median_orth = []
median_anti = []
err_par = []
err_orth = []
err_anti = []
err_par_top = []
err_orth_top = []
err_anti_top = []
index = np.arange(after[0],before[0]+1,1)
print(np.shape(index))

for i in index:
	energy = y1_good[:,i]
	flux = v1_good[:,i]
	total_flux = np.nansum(flux)
	par_data = energy*flux/total_flux
	par_data = np.nanmean(par_data)
	trend_par = np.append(trend_par,par_data)

	energy = y2_good[:,i]
	flux = v2_good[:,i]
	total_flux = np.nansum(flux)
	orth_data = energy*flux/total_flux
	orth_data = np.nanmean(orth_data)
	trend_orth = np.append(trend_orth,orth_data)

	energy = y3_good[:,i]
	flux = y3_good[:,i]
	total_flux = np.nansum(flux)
	anti_data = energy*flux/total_flux
	anti_data = np.nanmean(anti_data)
	trend_anti = np.append(trend_anti,anti_data)

	trend = np.append(trend,np.nanmean([par_data,orth_data,anti_data]))
# err_par = np.vstack((err_par_top,err_par))
# err_orth = np.vstack((err_orth_top,err_orth))
# err_anti = np.vstack((err_anti_top,err_anti))
before_array = np.arange(before_range[0],before_range[-1]+1,1)
during_array = np.arange(during_range[0],during_range[-1]+1,1)
after_array = np.arange(after_range[0],after_range[-1]+1,1)
before_energy_par = np.zeros((32,np.size(before_array)))
before_flux_par = np.zeros((32,np.size(before_array)))
during_energy_par = np.zeros((32,np.size(during_array)))
during_flux_par = np.zeros((32,np.size(during_array)))
after_energy_par = np.zeros((32,np.size(after_array)))
after_flux_par = np.zeros((32,np.size(after_array)))

before_energy_orth = np.zeros((32,np.size(before_array)))
before_flux_orth = np.zeros((32,np.size(before_array)))
during_energy_orth = np.zeros((32,np.size(during_array)))
during_flux_orth = np.zeros((32,np.size(during_array)))
after_energy_orth = np.zeros((32,np.size(after_array)))
after_flux_orth = np.zeros((32,np.size(after_array)))

before_energy_anti = np.zeros((32,np.size(before_array)))
before_flux_anti = np.zeros((32,np.size(before_array)))
during_energy_anti = np.zeros((32,np.size(during_array)))
during_flux_anti = np.zeros((32,np.size(during_array)))
after_energy_anti = np.zeros((32,np.size(after_array)))
after_flux_anti = np.zeros((32,np.size(after_array)))
for i in range(len(before_array)):
	x = before_array[i]
	before_energy_par[:,i] = y1_good[:,x]
	before_flux_par[:,i] = v1_good[:,x]
	before_energy_orth[:,i] = y2_good[:,x]
	before_flux_orth[:,i] = v2_good[:,x]
	before_energy_anti[:,i] = y3_good[:,x]
	before_flux_anti[:,i] = v3_good[:,x]
for i in range(len(during_array)):
	x = during_array[i]
	during_energy_par[:,i] = y1_good[:,x]
	during_flux_par[:,i] = v1_good[:,x]
	during_energy_orth[:,i] = y2_good[:,x]
	during_flux_orth[:,i] = v2_good[:,x]
	during_energy_anti[:,i] = y3_good[:,x]
	during_flux_anti[:,i] = v3_good[:,x]
for i in range(len(after_array)):
	x = after_array[i]
	after_energy_par[:,i] = y1_good[:,x]
	after_flux_par[:,i] = v1_good[:,x]
	after_energy_orth[:,i] = y2_good[:,x]
	after_flux_orth[:,i] = v2_good[:,x]
	after_energy_anti[:,i] = y3_good[:,x]
	after_flux_anti[:,i] = v3_good[:,x]
## Points to plot (Medians)
med_before_energy_par = np.nanmedian(before_energy_par,axis=1)
med_before_flux_par = np.nanmedian(before_flux_par,axis=1)
med_before_energy_orth = np.nanmedian(before_energy_orth,axis=1)
med_before_flux_orth = np.nanmedian(before_flux_orth,axis=1)
med_before_energy_anti = np.nanmedian(before_energy_anti,axis=1)
med_before_flux_anti = np.nanmedian(before_flux_anti,axis=1)

med_during_energy_par = np.nanmedian(during_energy_par,axis=1)
med_during_flux_par = np.nanmedian(during_flux_par,axis=1)
med_during_energy_orth = np.nanmedian(during_energy_orth,axis=1)
med_during_flux_orth = np.nanmedian(during_flux_orth,axis=1)
med_during_energy_anti = np.nanmedian(during_energy_anti,axis=1)
med_during_flux_anti = np.nanmedian(during_flux_anti,axis=1)

med_after_energy_par = np.nanmedian(after_energy_par,axis=1)
med_after_flux_par = np.nanmedian(after_flux_par,axis=1)
med_after_energy_orth = np.nanmedian(after_energy_orth,axis=1)
med_after_flux_orth = np.nanmedian(after_flux_orth,axis=1)
med_after_energy_anti = np.nanmedian(after_energy_anti,axis=1)
med_after_flux_anti = np.nanmedian(after_flux_anti,axis=1)

## Error Bars ##
## Min
min_energy_before_par = np.nanmin(before_energy_par,axis=1)
min_flux_before_par = np.nanmin(before_flux_par,axis=1)
min_energy_before_orth = np.nanmin(before_energy_orth,axis=1)
min_flux_before_orth = np.nanmin(before_flux_orth,axis=1)
min_energy_before_anti = np.nanmin(before_energy_anti,axis=1)
min_flux_before_anti = np.nanmin(before_flux_anti,axis=1)

min_energy_during_par = np.nanmin(during_energy_par,axis=1)
min_flux_during_par = np.nanmin(during_flux_par,axis=1)
min_energy_during_orth = np.nanmin(during_energy_orth,axis=1)
min_flux_during_orth = np.nanmin(during_flux_orth,axis=1)
min_energy_during_anti = np.nanmin(during_energy_anti,axis=1)
min_flux_during_anti = np.nanmin(during_flux_anti,axis=1)

min_energy_after_par = np.nanmin(after_energy_par,axis=1)
min_flux_after_par = np.nanmin(after_flux_par,axis=1)
min_energy_after_orth = np.nanmin(after_energy_orth,axis=1)
min_flux_after_orth = np.nanmin(after_flux_orth,axis=1)
min_energy_after_anti = np.nanmin(after_energy_anti,axis=1)
min_flux_after_anti = np.nanmin(after_flux_anti,axis=1)
## Max
max_energy_before_par = np.nanmax(before_energy_par,axis=1)
max_flux_before_par = np.nanmax(before_flux_par,axis=1)
max_energy_before_orth = np.nanmax(before_energy_orth,axis=1)
max_flux_before_orth = np.nanmax(before_flux_orth,axis=1)
max_energy_before_anti = np.nanmax(before_energy_anti,axis=1)
max_flux_before_anti = np.nanmax(before_flux_anti,axis=1)

max_energy_during_par = np.nanmax(during_energy_par,axis=1)
max_flux_during_par = np.nanmax(during_flux_par,axis=1)
max_energy_during_orth = np.nanmax(during_energy_orth,axis=1)
max_flux_during_orth = np.nanmax(during_flux_orth,axis=1)
max_energy_during_anti = np.nanmax(during_energy_anti,axis=1)
max_flux_during_anti = np.nanmax(during_flux_anti,axis=1)

max_energy_after_par = np.nanmax(after_energy_par,axis=1)
max_flux_after_par = np.nanmax(after_flux_par,axis=1)
max_energy_after_orth = np.nanmax(after_energy_orth,axis=1)
max_flux_after_orth = np.nanmax(after_flux_orth,axis=1)
max_energy_after_anti = np.nanmax(after_energy_anti,axis=1)
max_flux_after_anti = np.nanmax(after_flux_anti,axis=1)

## Get Errors to Plot 
err_energy_before_par_left = med_before_energy_par - min_energy_before_par
err_energy_before_par_right = (max_energy_before_par) - (med_before_energy_par)
err_energy_before_par = np.vstack((err_energy_before_par_left,err_energy_before_par_right))
err_energy_during_par_left = (med_during_energy_par) - (min_energy_during_par)
err_energy_during_par_right = (max_energy_during_par) - (med_during_energy_par)
err_energy_during_par = np.vstack((err_energy_during_par_left,err_energy_during_par_right))
err_energy_after_par_left = (med_after_energy_par) - (min_energy_after_par)
err_energy_after_par_right = (max_energy_after_par) - (med_after_energy_par)
err_energy_after_par = np.vstack((err_energy_after_par_left,err_energy_after_par_right))

err_flux_before_par_left = (med_before_flux_par) - (min_flux_before_par)
err_flux_before_par_right = (max_flux_before_par) - (med_before_flux_par)
err_flux_before_par = np.vstack((err_flux_before_par_left,err_flux_before_par_right))
err_flux_during_par_left = (med_during_flux_par) - (min_flux_during_par)
err_flux_during_par_right = (max_flux_during_par) - (med_during_flux_par)
err_flux_during_par = np.vstack((err_flux_during_par_left,err_flux_during_par_right))
err_flux_after_par_left = (med_after_flux_par) - (min_flux_after_par)
err_flux_after_par_right = (max_flux_after_par) - (med_after_flux_par)
err_flux_after_par = np.vstack((err_flux_after_par_left,err_flux_after_par_right))



err_energy_before_orth_left = med_before_energy_orth - min_energy_before_orth
err_energy_before_orth_right = (max_energy_before_orth) - (med_before_energy_orth)
err_energy_before_orth = np.vstack((err_energy_before_orth_left,err_energy_before_orth_right))
err_energy_during_orth_left = (med_during_energy_orth) - (min_energy_during_orth)
err_energy_during_orth_right = (max_energy_during_orth) - (med_during_energy_orth)
err_energy_during_orth = np.vstack((err_energy_during_orth_left,err_energy_during_orth_right))
err_energy_after_orth_left = (med_after_energy_orth) - (min_energy_after_orth)
err_energy_after_orth_right = (max_energy_after_orth) - (med_after_energy_orth)
err_energy_after_orth = np.vstack((err_energy_after_orth_left,err_energy_after_orth_right))

err_flux_before_orth_left = (med_before_flux_orth) - (min_flux_before_orth)
err_flux_before_orth_right = (max_flux_before_orth) - (med_before_flux_orth)
err_flux_before_orth = np.vstack((err_flux_before_orth_left,err_flux_before_orth_right))
err_flux_during_orth_left = (med_during_flux_orth) - (min_flux_during_orth)
err_flux_during_orth_right = (max_flux_during_orth) - (med_during_flux_orth)
err_flux_during_orth = np.vstack((err_flux_during_orth_left,err_flux_during_orth_right))
err_flux_after_orth_left = (med_after_flux_orth) - (min_flux_after_orth)
err_flux_after_orth_right = (max_flux_after_orth) - (med_after_flux_orth)
err_flux_after_orth = np.vstack((err_flux_after_orth_left,err_flux_after_orth_right))


err_energy_before_anti_left = med_before_energy_anti - min_energy_before_anti
err_energy_before_anti_right = (max_energy_before_anti) - (med_before_energy_anti)
err_energy_before_anti = np.vstack((err_energy_before_anti_left,err_energy_before_anti_right))
err_energy_during_anti_left = (med_during_energy_anti) - (min_energy_during_anti)
err_energy_during_anti_right = (max_energy_during_anti) - (med_during_energy_anti)
err_energy_during_anti = np.vstack((err_energy_during_anti_left,err_energy_during_anti_right))
err_energy_after_anti_left = (med_after_energy_anti) - (min_energy_after_anti)
err_energy_after_anti_right = (max_energy_after_anti) - (med_after_energy_anti)
err_energy_after_anti = np.vstack((err_energy_after_anti_left,err_energy_after_anti_right))

err_flux_before_anti_left = (med_before_flux_anti) - (min_flux_before_anti)
err_flux_before_anti_right = (max_flux_before_anti) - (med_before_flux_anti)
err_flux_before_anti = np.vstack((err_flux_before_anti_left,err_flux_before_anti_right))
err_flux_during_anti_left = (med_during_flux_anti) - (min_flux_during_anti)
err_flux_during_anti_right = (max_flux_during_anti) - (med_during_flux_anti)
err_flux_during_anti = np.vstack((err_flux_during_anti_left,err_flux_during_anti_right))
err_flux_after_anti_left = (med_after_flux_anti) - (min_flux_after_anti)
err_flux_after_anti_right = (max_flux_after_anti) - (med_after_flux_anti)
err_flux_after_anti = np.vstack((err_flux_after_anti_left,err_flux_after_anti_right))

##########
## Find Strahl Direction
##########


# ## Put into unit vector
# mag = []
# print('mag shape',np.shape(mag))
# B_slice = np.zeros((3,np.size(index)))
# print('B[0,:]',np.shape(B[0,:]))
# for i in range(len(index)):
# 	x = index[i]
# 	B_slice[:,i] = B[:,x]
# 	magnitude = np.sqrt(B[0,x]**2 + B[1,x]**2 + B[2,x]**2)
# 	mag = np.append(mag,magnitude)
# print('bslice',np.shape(B_slice))
# print('mag',np.shape(mag))
# bu = B_slice/mag
# # B_u = np.vstack((y8_x,y8_y,y8_z))
# print('bu', np.shape(bu))
# bu_s = np.sign(bu)
# test_r13 = (bu_s[0,:]*bu_s[1:,] >= 0)
# print(np.shape(test_r13))
# print('test_r13',test_r13)
# test_r24 = (np.sum(bu_s[:1,:],axis=0) == 0)
# print(np.shape(test_r24))
# print('test_r24',test_r24)

# test_r1 = test_r13 and (np.sum(bu_s[0:1,:],axis=0) > 0)
# print('test_r1',test_r1)

# exit()



print(trend[0])
print(trend)
residual = []
residual_par = []
residual_orth = []
residual_anti = []
if date == '07_14_2008/':
	baseline = trend[-1]
else:
	baseline = trend[0]
for num in trend:
	residual = np.append(residual, (num - baseline))
for num in trend_par:
	residual_par = np.append(residual_par, (num - baseline))
for num in trend_orth:
	residual_orth = np.append(residual_orth, (num - baseline))
for num in trend_anti:
	residual_anti = np.append(residual_anti, (num - baseline))

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(2,2,1)
ax1 = fig.add_subplot(2,2,2)
ax2 = fig.add_subplot(2,2,3)
ax3 = fig.add_subplot(2,2,4)
ax.plot(plot_time,trend,'kx',label='Omni-directional')
ax.plot(plot_time,trend_par,'r.',label='Parallel')
ax.plot(plot_time,trend_orth,'g.',label='Orthogonal')
ax.plot(plot_time,trend_anti,'b.',label='Anti-Parallel')
ax.legend(loc='best',frameon=True,prop={'size': 6})
if date == '07_14_2008/':
	ax.axhline(y=trend[-1],ls = '--',color='black')
else:
	ax.axhline(y=12,ls = '--',color='black')#trend[-1]
ax.set_xticklabels(unix_to_utc(ax.get_xticks()))
ax.set_title('Electron Energy Increase')
ax.set_xlabel('Time of Day')
ax.set_ylabel('Average Energy of Each Electron (eV)')

ax1.errorbar(med_before_energy_par,med_before_flux_par,yerr=err_flux_before_par,xerr=err_energy_before_par,fmt='r^',lw=1,label='Upstream')
ax1.errorbar(med_during_energy_par,med_during_flux_par,yerr=err_flux_during_par,xerr=err_energy_during_par,fmt='gs',lw=1,label='During')
ax1.errorbar(med_after_energy_par,med_after_flux_par,yerr=err_flux_after_par,xerr=err_energy_after_par,fmt='bX',lw=1,label='Downstream')
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylim(10,10**7)
ax1.set_xlim(10,3*10**4)
ax1.legend(loc='best',frameon=True)
ax1.set_title('Parallel Electron Energy Distribution')
ax1.set_ylabel('Flux (#$cm^{-2}s^{-1}sr^{-1}eV^{-1}$)')
ax1.set_xlabel('Energy (eV)')

ax2.errorbar(med_before_energy_orth,med_before_flux_orth,yerr=err_flux_before_orth,xerr=err_energy_before_orth,fmt='r^',lw=1,label='Upstream')
ax2.errorbar(med_during_energy_orth,med_during_flux_orth,yerr=err_flux_during_orth,xerr=err_energy_during_orth,fmt='gs',lw=1,label='During')
ax2.errorbar(med_after_energy_orth,med_after_flux_orth,yerr=err_flux_after_orth,xerr=err_energy_after_orth,fmt='bX',lw=1,label='Downstream')
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_ylim(10,10**7)
ax2.set_xlim(10,3*10**4)
ax2.legend(loc='best',frameon=True)
ax2.set_title('Orthogonal Electron Energy Distribution')
ax2.set_ylabel('Flux (#$cm^{-2}s^{-1}sr^{-1}eV^{-1}$)')
ax2.set_xlabel('Energy (eV)')

ax3.errorbar(med_before_energy_anti,med_before_flux_anti,yerr=err_flux_before_anti,xerr=err_energy_before_anti,fmt='r^',lw=1,label='Upstream')
ax3.errorbar(med_during_energy_anti,med_during_flux_anti,yerr=err_flux_during_anti,xerr=err_energy_during_anti,fmt='gs',lw=1,label='During')
ax3.errorbar(med_after_energy_anti,med_after_flux_anti,yerr=err_flux_after_anti,xerr=err_energy_after_anti,fmt='bX',lw=1,label='Downstream')
ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.set_ylim(10,10**7)
ax3.set_xlim(10,3*10**4)
ax3.legend(loc='best',frameon=True)
ax3.set_title('Anti-Parallel Electron Energy Distribution')
ax3.set_ylabel('Flux (#$cm^{-2}s^{-1}sr^{-1}eV^{-1}$)')
ax3.set_xlabel('Energy (eV)')
# ax1.set_xticklabels(unix_to_utc(ax1.get_xticks()))
fig.subplots_adjust(left=0.06, bottom=0.07, right=0.99, top=0.9, wspace=0.13, hspace=0.25)
plt.suptitle('Electron Acceleration and Energy Distribution Throughout SLAMS',fontsize=18)
# plt.tight_layout()
plt.savefig(path + date + savedir + 'Acceleration_trend_' + event + '.png')
plt.show()
exit()
##########
## Create Plot to Show Acceleration
##########
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(1,1,1)
for i in range(len(before)):
	num = int(before[i])
	ax.loglog(y2_good[:,num],v2_good[:,num],'k^',label='Upstream' if i == 0 else '')
for i in range(len(during)):
	num = int(during[i])
	ax.loglog(y2_good[:,num],v2_good[:,num],'k+',label='During' if i == 0 else '')
for i in range(len(after)):
	num = int(after[i])
	ax.loglog(y2_good[:,num],v2_good[:,num],'k.',label='Downstream' if i == 0 else '')
ax.legend(loc='lower left', frameon=True,prop={'size': 8})
end = int(before[0])
start = int(after[i])
ax.set_title(plot_title + '     Time = %s - %s'%(time_utc[start],time_utc[end]))
ax.set_xlabel('Energy (eV)')
ax.set_ylabel('Flux (#$cm^{-2}s^{-1}sr^{-1}eV^{-1}$)')
plt.savefig(path + date + savedir + 'Acceleration_' + event + '.png')
# plt.show()
