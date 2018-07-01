import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import NullFormatter

plt.ion()
plt.cla()

path = '/Users/Tyler/Documents/NASA/Summer2018/TestIDL/'
file1 = 'dh_273_Fit_param_A.txt'
file2 = 'dh_274_Fit_param_B.txt'
file3 = 'dh_275_Fit_param_C.txt'
file4 = 'dh_284_Red_Chi-Squared_value_each_fit.txt'
file5 = 'dh_131_Average_energy_spectra.txt'
f1 = path + file1
f2 = path + file2
f3 = path + file3
f4 = path + file4
f5 = path + file5
f1 = np.loadtxt(f1) #fit param A
f2 = np.loadtxt(f2) #fit param B
f3 = np.loadtxt(f3) #fit param C
f4 = np.loadtxt(f4) #chi squared values
f5 = np.loadtxt(f5) #energy spectra (21,1362)
# print(np.shape(f4))

minE = 50 #eV
maxE = 3*10**4 #eV
x = np.arange(minE,maxE,10)

bin1 = f5[1:,0] #20 energy values
print(np.shape(bin1))
bin2 = f5[1:,1] #20 energy values
print(np.shape(bin2))
# bin3 = f5[1:,2] #20 energy values
# bin4 = f5[1:,3] #20 energy values
# bin5 = f5[1:,4] #20 energy values
# bin6 = f5[1:,5] #20 energy values
# bin7 = f5[1:,6] #20 energy values
# bin8 = f5[1:,7] #20 energy values
# bin9 = f5[1:,8] #20 energy values
# bin10 = f5[1:,9] #20 energy values
# bin11 = f5[1:,10] #20 energy values
# bin12 = f5[1:,11] #20 energy values
# bin13 = f5[1:,12] #20 energy values
# bin14 = f5[1:,13] #20 energy values
# bin15 = f5[1:,14] #20 energy values
# bin16 = f5[1:,15] #20 energy values
# bin17 = f5[1:,16] #20 energy values
# bin18 = f5[1:,17] #20 energy values
# bin19 = f5[1:,18] #20 energy values
# bin20 = f5[1:,19] #20 energy values

chisq = f4[1:,:]
print('Chi Sq.')
print(np.shape(chisq))
chisq_parallel = str(chisq[0,0])
print(chisq_parallel)
chisq_orth = str(chisq[1,0])
print(chisq_orth)
chisq_antiparallel = str(chisq[2,0])
print(chisq_antiparallel)

# x1 = f1[0,:]
# x2 = f2[0,:]
# x3 = f3[0,:]
# for i in range(1330):
#     if x1[i] != x2[i] != x3[i]:
#         print('Not the same')
#         break
        
# x = x1
A_parallel = f1[1,0]
print(A_parallel)
A_orth = f1[2,0]
A_antiparallel = f1[3,0]        
 
B_parallel = f2[1,0]
print(B_parallel)
B_orth = f2[2,0]
B_antiparallel = f2[3,0]     

C_parallel = f3[1,0]
print(C_parallel)
C_orth = f3[2,0]
C_antiparallel = f3[3,0] 

# y_parallel = A_parallel*x**B_parallel*np.exp(C_parallel*x)
# y_orth = A_orth*x**B_orth*np.exp(C_orth*x)
# y_antiparallel = A_antiparallel*x**B_antiparallel*np.exp(C_antiparallel*x)
y_parallel = A_parallel*bin1**B_parallel*np.exp(C_parallel*bin1)
y_orth = A_orth*bin1**B_orth*np.exp(C_orth*bin1)
y_antiparallel = A_antiparallel*bin1**B_antiparallel*np.exp(C_antiparallel*bin1)
plt.figure()
plt.loglog(bin1,y_parallel,'r',lw=1,label='Parallel')
plt.loglog(bin1,y_orth,'g',lw=1,label='Orthogonal')
plt.loglog(bin1,y_antiparallel,'b',lw=1, label ='Anti-Parallel')
plt.legend(loc='upper right', frameon=True,prop={'size': 9})
plt.xlabel('Energy (eV)')
plt.ylabel('Flux (#$cm^{-2}s^{-1}sr^{-1}eV^{-1}$)')
#plt.title('Power Law Fit with Reduced Chi-Squared = %d')%chisq
plt.title('Power Law Fit')
# plt.text(0.5,0.5,'Reduced $Chi ^{2}$ Parallel = %s'%(chisq_parallel))


# # plt.yscale('log')
# # plt.gca().yaxis.set_minor_formatter(NullFormatter())
# # fig, ax = plt.subplots()
# # ax.axis([1, 10000, 1, 100000])
# # for axis in [ax.xaxis, ax.yaxis]:
# #     axis.set_major_formatter(ScalarFormatter())
# # ax.loglog()
# # line,=ax.plot(x,y_parallel,'r')
plt.tight_layout()
plt.show()