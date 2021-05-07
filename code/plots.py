# -*- coding: utf-8 -*-

import numpy as np
import os, math
from matplotlib import pyplot as plt

base = os.path.abspath("")+'/results'
if not os.path.exists(base ):
    os.mkdir(base)

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

        fig, ax = plt.subplots(figsize=(10,5))
        for i in T.keys():
            listX = X.get(i)
            listT = T.get(i)
            index = list(range(len(listX)))
            index.sort(key = listX.__getitem__)
            listX[:] = [listX[i] for i in index]
            listT[:] = [listT[i] for i in index]
            ax.errorbar(x=listX, y=listT, color=(np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)), label=i )
        ax.set_ylim(bottom=0,top=int(lim))
        plt.ylabel('x1')
        plt.xlabel('r')
        plt.legend()
        plt.savefig(path+'/Results_Time.png')
        # plt.show()
        plt.close()

        fig, ax = plt.subplots(figsize=(10,5))
        for i in Y.keys():
            listX = X.get(i)
            listY = Y.get(i)
            index = list(range(len(listX)))
            index.sort(key = listX.__getitem__)
            listX[:] = [listX[i] for i in index]
            listY[:] = [listY[i] for i in index]
            ax.errorbar(x=listX, y=listY, color=(np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)), label=i )
        ax.set_ylim(bottom=0,top=1.1)
        plt.ylabel('x1')
        plt.xlabel('r')
        plt.legend()
        plt.savefig(path+'/Results_POP.png')
        # plt.show()
        plt.close()
