f = open('/Volumes/Seagate/NASA/Summer2018/THEMIS/08_19_2008/THC_tags.txt','a')
for i in range(len(dq)):
    print(dq[i][0],i+1)
	f.write('%s %d \n'%(dq[i][0],i+1))
f.close()