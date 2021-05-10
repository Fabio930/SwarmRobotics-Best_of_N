# -*- coding: utf-8 -*-

import numpy as np
import os, math
from matplotlib import pyplot as plt

base = os.path.abspath("")+'/results'
if not os.path.exists(base ):
    os.mkdir(base)

resumeT={}
resumeY={}
resumeX={}
for dir in os.listdir(base):
    if not os.path.isfile(os.path.join(base, dir)):
        X={}
        Y={}
        T={}
        lim = ''
        path=os.path.join(base, dir)
        flags = ''
        for i in range(len(dir)):
            if dir[len(dir)-1 - i] != ",":
                flags += dir[len(dir)-1 - i]
            else:
                break
        for i in range(len(flags)):
            lim += flags[len(flags)-1 - i]
        for elem in os.listdir(path):
            if elem[0] == 'P':
                flag=np.array([])
                y=open(os.path.join(path,elem),'r')
                for l in y.readlines():
                    flags=''
                    for c in l:
                        if c != '\n':
                            flags+=c
                    flag = np.append(flag,float(flags))
                y.close()
                Y.update({elem[-9]+elem[-8]+elem[-7]+elem[-6]+elem[-5]:flag})
                if elem[-5] == '1':
                    resumeY.update({elem[-9]+elem[-8]+elem[-7]+elem[-6]+elem[-5]:flag})

            elif elem[0] == 'T':
                flag=np.array([])
                t=open(os.path.join(path,elem),'r')
                for l in t.readlines():
                    flags=''
                    for c in l:
                        if c != '\n':
                            flags+=c
                    flag = np.append(flag,float(flags))
                t.close()
                T.update({elem[-9]+elem[-8]+elem[-7]+elem[-6]+elem[-5]:flag})
                if elem[-5] == '1':
                    resumeT.update({elem[-9]+elem[-8]+elem[-7]+elem[-6]+elem[-5]:flag})
            elif elem[0] == 'r':
                flag=np.array([])
                x=open(os.path.join(path,elem),'r')
                for l in x.readlines():
                    flags=''
                    for c in l:
                        if c != '\n':
                            flags+=c
                    flag = np.append(flag,flags)
                x.close()
                X.update({elem[-9]+elem[-8]+elem[-7]+elem[-6]+elem[-5]:flag})
                if elem[-5] == '1':
                    resumeX.update({elem[-9]+elem[-8]+elem[-7]+elem[-6]+elem[-5]:flag})

        fig, ax = plt.subplots(figsize=(10,5))
        for i in T.keys():
            listX = X.get(i)
            listT = T.get(i)
            listY = Y.get(i)
            index = list(range(len(listX)))
            index.sort(key = listX.__getitem__)
            listX[:] = [listX[i] for i in index]
            listT[:] = [listT[i] for i in index]
            listY[:] = [listY[i] for i in index]
            ax.errorbar(x=listX, y=listT, color=(int(i[-5])*0.1,int(i[-2])*0.1,int(i[-1])*0.1),  label=i )
        ax.set_ylim(bottom=0,top=int(lim)+500)
        plt.title('Average over 100 runs -TIME-')
        plt.ylabel('steps')
        plt.xlabel('r')
        plt.legend()
        plt.savefig(path+'/Results_Time.png')
        # plt.show()
        plt.close()

        fig, ax = plt.subplots(figsize=(10,5))
        for i in Y.keys():
            ax.errorbar(x=X.get(i), y=Y.get(i), color=(int(i[-5])*0.1,int(i[-2])*0.1,int(i[-1])*0.1),  label=i )
        ax.set_ylim(bottom=0,top=1.1)
        plt.title('Average over 100 runs -POP ON THE BEST NODE-')
        plt.ylabel('x1')
        plt.xlabel('r')
        plt.legend()
        plt.savefig(path+'/Results_POP.png')
        # plt.show()
        plt.close()

fig, ax = plt.subplots(figsize=(10,5))
for i in resumeT.keys():
    listX = resumeX.get(i)
    listT = resumeT.get(i)
    index = list(range(len(listX)))
    index.sort(key = listX.__getitem__)
    listX[:] = [listX[i] for i in index]
    listT[:] = [listT[i] for i in index]
    ax.errorbar(x=listX, y=listT, color=(int(i[-2])*0.1,int(i[-1])*0.1,int(i[-5])*0.1), label='N='+i[-2] )
ax.set_ylim(bottom=0,top=int(lim)+500)
plt.title('Resuming plot for different tree size (N) -AVG TIME-')
plt.ylabel('steps')
plt.xlabel('r')
plt.legend()
plt.savefig(base+'/Results_Time.png')
# plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(10,5))
for i in resumeY.keys():
    listX = resumeX.get(i)
    listY = resumeY.get(i)
    index = list(range(len(listX)))
    index.sort(key = listX.__getitem__)
    listX[:] = [listX[i] for i in index]
    listY[:] = [listY[i] for i in index]
    ax.errorbar(x=listX, y=listY, color=(int(i[-2])*0.1,int(i[-1])*0.1,int(i[-5])*0.1),  label='N='+i[-2])
ax.set_ylim(bottom=0,top=1.1)
plt.title('Resuming plot for different tree size (N) -AVG POPULATION ON BEST NODE')
plt.ylabel('x1')
plt.xlabel('r')
plt.legend()
plt.savefig(base+'/Results_POP.png')
# plt.show()
plt.close()
