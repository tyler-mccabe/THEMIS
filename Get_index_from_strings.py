import scipy 
from scipy import io

path = '/Users/tylermccabe/Documents/NASA/Summer2018/THEMIS/'
date = '07_14_2008/'
date = '08_19_2008/'
tplot = 'TPLOT_save_file_THC.tplot'

file = scipy.io.readsav(path+date+tplot,python_dict=True)
dq = file['dq']
dh = dq['dh']

input_string = 'fgh_mag'
input_string1 = 'fgh_gse'
input_string2 = 'peeb_density'
input_string3 = 'peeb_avgtemp'
input_string4 = 'peib_density'
input_string5 = 'peib_avgtemp'

for i in range(len(dq)):
	# print(dq[i][0],'\n',i)
	string_name = dq[i][0].decode()
	# print(string_name)
	if string_name[4:] == input_string:
		index = i
		print(index)

def get_index(string_handle):
	for i in range(len(dq)):
		string = dq[i][0].decode()
		if string[4:] == string_handle:
			index = i
	return index
print(get_index(input_string))
