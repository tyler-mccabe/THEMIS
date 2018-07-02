import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import io

path = '/Users/tylermccabe/Documents/NASA/Summer2018/THEMIS/'
date = '08_19_2008/'
tplot = 'TPLOT_save_file_THC.tplot'
savedir = 'Plots/THC/'
plot_title = 'Themis C: ' + date[:-1]

file = scipy.io.readsav(path+date+tplot,python_dict=True)
dq = file['dq']
dh = dq['dh']

t = dh[37].x[0]
B = dh[37].y[0]

t1 = dh[70].x[0]
y1 = dh[70].y[0]

mins = np.asarray([0])
maxs = np.asarray([])

for i in range(len(t)-1):
	if (t[i+1]-t[i]) >= 200:
		mins = np.append(mins,i+1)
		maxs = np.append(maxs,i)
maxs = np.append(maxs,len(t)-1)
print(mins)
print(maxs)
for num in mins:
	num = int(num)
	print(t[num]-1.2191e9)
print('\n')
for num in maxs:
	num = int(num)
	print(t[num]-1.2191e9)
for i in range(len(mins)):
	start = int(mins[i])
	end = int(maxs[i])
	plt.plot(t,B,'k')
	plt.xlim(t[start],t[end])
	plt.show()