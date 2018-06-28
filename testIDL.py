import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import io

# plt.ion()
# plt.cla()

#file = '/Users/Tyler/Documents/NASA/Summer2018/TestIDL/THEMIS_EESA_Burst_c_SWF_SCF_Power_Law_Exponential_Fit-Results_2008-07-14_1222x11.668-2243x25.049.sav'
file = '/Users/Tyler/Documents/NASA/Summer2018/TestIDL/TPLOT_save_file_THC_FGM-ALL_EESA-IESA-Moments_EFI-Cal-Cor_SCM-Cal-Cor_ESA-SST-Spectra_eVDF_fit_results_2008-07-14_0000x00-2359x59.tplot'
f = scipy.io.readsav(file,python_dict=True,verbose=True) 

dq = f['dq']
dh = dq['dh']
t = dh[100].x[0]  # Index is one less than IDL save file
print(t.shape)
t = t - 1215993600
print(t)

print('y\'s')
y1 = dh[100].y[0][0]
print(y1)
print(np.shape(y1))
y2 = dh[100].y[0][1]
print(y2.shape)
print(y2)
y3 = dh[100].y[0][2]
print(y3.shape)
print(y3)

# v1 = dh[130].v[0][0]
# print(v1.shape)
# print(v1)
# v2 = dh[130].v[0][1]
# print(v2.shape)
# print(v2)
# v3 = dh[130].v[0][2]
# print(v3.shape)
# print(v3)

# 
# for i in range(296):
#     print(dq[i][0],i+1)
#     print(dh[i].y[0].shape)

data = np.asarray([t,y1,y2,y3])
# print(data)
# np.savetxt('/Users/Tyler/Documents/NASA/Summer2018/TestIDL/dh_101_electron_bulk_velocity_gse.txt',data,delimiter='\t',newline = '\n')
# plt.plot(t,y1,'r')
# plt.plot(t,y2,'b')
# plt.plot(t,y3,'g')
# plt.show()