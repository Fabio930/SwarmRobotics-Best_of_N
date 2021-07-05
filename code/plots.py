# -*- coding: utf-8 -*-

import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import os, csv
from matplotlib import pyplot as plt

bases = [os.path.abspath("")+'/results_known',os.path.abspath("")+'/results_unknown']
for base in bases:
    for dir in os.listdir(base):
        path=os.path.join(base, dir)
        for sub_dir in os.listdir(path):
            sub_path=os.path.join(path, sub_dir)
            for elem in os.listdir(sub_path):
                if elem[-6]+elem[-5]+elem[-4]+elem[-3]+elem[-2]+elem[-1] == 'HM.csv':
                    fig,ax = plt.subplots(figsize=(10,5))
                    plt.tick_params(bottom='on')
                    hm = pd.read_csv(os.path.join(sub_path, elem))
                    # print(hm,'\n**********************')
                    hm['x'] = pd.Categorical(hm['x'])
                    hm.head()
                    hm = hm.pivot_table('N','x','r',fill_value=0)
                    # print(hm,'\n______________________',sub_path,elem)
                    # print(hm,'\n++++++++++++++++++++++')
                    ax=sns.heatmap(hm,cmap='coolwarm')
                    ax.set_xticks(np.arange(len(hm.columns)))
                    ax.set_yticks(np.arange(len(hm.index)))
                    ax.set_xticklabels(hm.columns)
                    TICKS=np.array([])
                    for i in hm.index:
                        if round(i%0.1,2) == 0.0 or round(i%0.1,2) == 0.1:
                            TICKS = np.append(TICKS,round(i,2))
                        else:
                            TICKS = np.append(TICKS,'')
                    ax.set_yticklabels(TICKS)
                    ax.invert_yaxis()
                    ax.set_xlabel('r')
                    plt.tight_layout()
                    plt.savefig(sub_path+'/'+sub_dir+'.png')
                    # plt.show()
                    plt.close()
