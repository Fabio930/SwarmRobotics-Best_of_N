from main.utility_filter import Filter
from matplotlib import pyplot as plt
import matplotlib as mpl
import random
import numpy as np

num_updates=50
iterations=1000
colors=['#fde725', '#7ad151', '#22a884','#2a788e','#414487','#440154']

record=[0]*num_updates
r1=[0]*num_updates
r2=[0]*num_updates
r3=[0]*num_updates
for i in range(iterations):
    filter=Filter(.75,1)
    for j in range(num_updates):
        filter.update_utility(10+np.random.normal(0,1))
        # print('\n______________________\n',filter.data_1,'\n',filter.data_2)
        record[j]+=filter.distance
for i in range(num_updates):
    record[i] = record[i]*(1/iterations)
    r1[i] = 1-record[i]
    r2[i] = 2*(1-record[i])
    r3[i] = 3*(1-record[i])

plt.plot(record,label=r'Hd',c=colors[1])
plt.plot([1]*num_updates,linestyle=':',color='black')
plt.plot(r1,label=r'r1= 1-Hd',c=colors[2])
plt.plot(r2,label=r'r2= 2(1-Hd)',c=colors[3])
plt.plot(r3,label=r'r3= 3(1-Hd)',c=colors[4])

plt.xlabel('filter updates')
plt.ylim(-.1, 3)
plt.legend(loc='best',borderaxespad=0,ncol=2)
plt.tight_layout()
plt.show()
plt.close()

em=np.arange(0.2,11,1)
Hd2=[None]*len(em)
S=[]
t=0
for j in em:
    flag=[]
    s=0.001
    for i in range(1,10000):
        if t==0:
            S.append(s)
        temp = 1 - np.exp(-.25*(j**2)/(2*(s+.2)*s))
        flag.append(temp)
        s+=.001
    Hd2[t]=flag
    t+=1
fig,ax = plt.subplots(figsize=(10, 5))
for i in range(len(Hd2)):
    ax.plot(S,Hd2[i],color=mpl.cm.viridis((len(Hd2)-i)/len(Hd2)))
plt.xlabel('variance')
N=len(em)
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=0.0,vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm, ticks=np.linspace(1,0.0,N), label='Error', fraction=0.046)
cbar.ax.set_yticklabels([str(i) for i in em])
ax.set_xticks(np.arange(0,10.5,.5))
plt.grid(True,linestyle=':')
plt.tight_layout()
plt.show()
plt.close()
