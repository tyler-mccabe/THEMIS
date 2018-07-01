import numpy as np

index = np.asarray([32,42,243,249,252,254])
index = np.asarray([273,274,275,276,277,278,279,280,281,282,283,284])
name = ['_Fit_param_A','_Fit_param_B','_Fit_param_C','_Fit_param_D','_One_sigma_param_A','_One_sigma_param_B','_One_sigma_param_C','_One_sigma_param_D','_Fit_Status_Flag','_Deg_Freedom','_Chi-Squared_value_each_fit','_Red_Chi-Squared_value_each_fit']
path = '/Users/Tyler/Documents/NASA/Summer2018/TestIDL/'
# n=0
# for i in index:
#     filename = str(i)+name[n]
#     file = path + 'dh_%s.txt'%filename
#     print(file)
#     f = np.loadtxt(file)
#     time = f[0]
#     time = time - 1215993600
#     f[0] = time
#     print(time)
#     np.savetxt(file,f,delimiter='\t',newline = '\n')
#     n=n+1
file = path + 'dh_268_dB_Bomag_fgh.txt'
f = np.loadtxt(file)
time = f[0]
time = time - 1215993600
f[0] = time
np.savetxt(file,f,delimiter='\t',newline = '\n')