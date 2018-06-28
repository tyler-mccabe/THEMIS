import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import io

path = '/Volumes/Seagate/NASA/Summer2018/THEMIS/'
date = '07_14_2008/'
tplot = 'TPLOT_save_file_THC_new.tplot'

file = scipy.io.readsav(path+date+tplot,python_dict=True)
dq = file['dq']
dh = dq['dh']

index = 37 # magnetic field magnitude
t = dh[37].x[0] - 1215993600
B = dh[37].y[0]
label = ['magnetic field']
df = pd.DataFrame(np.array(B).byteswap().newbyteorder(),np.array(t).byteswap().newbyteorder(),label)
# print(df)

fig = plt.figure()
df.plot(style='k-',linewidth='0.5')
# df.plot(x='index',y=df.mean(1),style='b')
# plt.legend(loc='best')
plt.show()
# x = np.arange(0,10,1)
# print('x: ',x)
# y = np.arange(0,20,2)
# print('y: ',y)
# label = ['ys']
# df = pd.DataFrame(y,x,label)
# print(df)