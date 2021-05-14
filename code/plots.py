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
        fig,ax = plt.subplots(figsize=(10,5))
        for elem in os.listdir(sub_path):
            if elem[-4]+elem[-3]+elem[-2]+elem[-1] == '.csv':
                if elem[-6]+elem[-5] == 'BI':
                    ax2=ax.twinx()
                    bi = csv.DictReader(open(os.path.join(sub_path, elem)))
                    r = np.array([])
                    m = np.array([])
                    std = np.array([])
                    for row in bi:
                        r = np.append(r,str(row['r']))
                        m = np.append(m,float(row['m']))
                        std = np.append(std,float(row['std']))
                    index = list(range(len(r)))
                    index.sort(key = r.__getitem__)
                    r[:] = [r[i] for i in index]
                    m[:] = [m[i] for i in index]
                    std[:] = [std[i] for i in index]
                    sum = m+std
                    sumO = np.array([])
                    for i in sum:
                        if i > 1:
                            sumO = np.append(sumO,1)
                        else:
                            sumO = np.append(sumO,i)
                    sns.lineplot(x=r,y=m, linewidth=.5,color='green',ls='--',ax=ax2)
                    sns.lineplot(x=r,y=sumO, linewidth=1,color='blue',ax=ax2)
                    ax2.set_ylim([-0.01,1.01])
                    ax2.set_xlim([0,12])
                elif elem[-6]+elem[-5] == 'HM':
                    hm = pd.read_csv(os.path.join(sub_path, elem))
                    # print(hm,'\n**********************')
                    hm['x'] = pd.Categorical(hm['x'])
                    hm.head()
                    hm = hm.pivot_table('N','x','r',fill_value=0)
                    # print(hm,'\n______________________')
                    hm = hm.reindex(np.arange(0,1.02,0.02), fill_value=0)
                    # print(hm,'\n++++++++++++++++++++++')
                    ax=sns.heatmap(hm,cmap=plt.get_cmap("Reds"))
                    ax.set_xticks(np.arange(len(hm.columns)))
                    ax.set_yticks(np.arange(len(hm.index)))
                    ax.set_xticklabels(hm.columns)
                    ax.get_yaxis().set_visible(False)
                    ax.invert_yaxis()
                    ax.set_ylabel('x')
                    ax.set_xlabel('r')
        plt.tight_layout()
        plt.savefig(sub_path+'/'+sub_dir+'.png')
        # plt.show()
        plt.close()
