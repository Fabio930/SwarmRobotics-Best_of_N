# -*- coding: utf-8 -*-

import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import os, csv
from matplotlib import pyplot as plt

base = os.path.abspath("")+'/results'
if not os.path.exists(base ):
    os.mkdir(base)

for dir in os.listdir(base):
    path=os.path.join(base, dir)
    for sub_dir in os.listdir(path):
        sub_path=os.path.join(path, sub_dir)
        plt.tick_params(bottom='on')
        for elem in os.listdir(sub_path):
            if elem[-4]+elem[-3]+elem[-2]+elem[-1] == '.csv':
                fig,ax = plt.subplots(figsize=(10,5))
                hm = pd.read_csv(os.path.join(sub_path, elem))
                # print(hm,'\n**********************')
                hm['x'] = pd.Categorical(hm['x'])
                hm.head()
                hm = hm.pivot_table('N','x','r',fill_value=0)
                # print(hm,'\n______________________')
                hm = hm.reindex(np.arange(0,1.025,0.025), fill_value=0)
                # print(hm,'\n++++++++++++++++++++++')
                ax=sns.heatmap(hm,cmap='hot_r')
                ax.set_xticks(np.arange(len(hm.columns)))
                ax.set_yticks(np.arange(len(hm.index)))
                ax.set_xticklabels(hm.columns)
                ax.set_yticklabels(hm.index)
                ax.get_yaxis().set_visible(False)
                ax.invert_yaxis()
                X = np.array([])
                Y = np.array([])
                Z = np.array([])
                # Y = np.append(Y,0.5)
                # Z = np.append(Z,0)
                # X = np.append(X,'')
                for i in hm.columns:
                    X = np.append(X,str(i))
                    flag = np.array([])
                    for j in hm.index:
                        if hm.loc[j,i] > 0:
                            flag = np.append(flag,j)
                    Y = np.append(Y,flag.mean())
                    Z = np.append(Z,flag.std())
                u=Y+Z
                d=Y-Z
                for i in range(len(Y)):
                    if Z[i]<0.15:
                        u[i]=Y[i]
                        d[i]=Y[i]
                for i in range(len(Y)):
                    if i > 0:
                        if d[i-1]!=u[i-1]:
                            if d[i]==u[i]:
                                d[i]=d[i-1]
                                Y[i]=Y[i-1]
                for i in range(len(d)):
                    if d[i]<0:
                        d[i]=0
                ax2 = ax.twinx()
                sns.lineplot(x=X,y=Y,color='green',linewidth=1,ls='--',ax=ax2)
                sns.lineplot(x=X,y=u,color='blue',linewidth=1.5,ax=ax2)
                sns.lineplot(x=X,y=d,color='blue',linewidth=1.5,ax=ax2)
                ax2.set_ylim([-0.01,1.01])
                ax2.set_ylabel('x')
                ax.set_xlabel('r')
                plt.tight_layout()
                plt.savefig(sub_path+'/'+sub_dir+'.png')
                # plt.show()
                plt.close()
