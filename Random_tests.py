import matplotlib.pyplot as plt
import numpy as np

file = '/Users/Tyler/Documents/NASA/Summer2018/TestIDL/dh_32.txt'
f = np.loadtxt(file)
print(f.shape)
t = f[0,:]
# print(t)
# print(t[0])
# print(t[1])
# print(t[2])
# t1 = t[1]
# t2 = t[2]
# print(t2-t1)
y1 = f[1,:]
print(y1)
y2 = f[2,:]
print(y2)
y3 = f[3,:]
print(y3)

plt.plot(t,y1)
plt.plot(t,y2)
plt.plot(t,y3)
plt.show()