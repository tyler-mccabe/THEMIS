import numpy as np
import scipy
from scipy import io
f = open('/Users/tylermccabe/Documents/NASA/Summer2018/THEMIS/08_19_2008/THC_tags.txt','w')
file = scipy.io.readsav('/Users/tylermccabe/Documents/NASA/Summer2018/THEMIS/08_19_2008/TPLOT_save_file_THC.tplot',python_dict=True)
dq = file['dq']

tags = []
# index = np.asarray([])
for i in range(len(dq)):
	print(dq[i][0],i+1)
	tag = str(dq[i][0])
	index = str(i+1)
	string_tag = tag + index
	f.write(string_tag +' \n')
f.close()