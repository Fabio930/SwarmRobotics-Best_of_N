# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os, math, csv
from main.arena import Arena
from main.agent import Agent
from matplotlib import pyplot as plt

class Results:
    def __init__( self, arena ):
        self.heat_map = {}
        count=0
        self.r = arena.agents[0].R()
        self.arena = arena
        self.base = os.path.abspath("results")
        if not os.path.exists(self.base ):
            os.mkdir(self.base )
        self.path = self.base+'/K'+ str(self.arena.tree_branches) +'D'+ str(self.arena.tree_depth)+'_'+str(self.arena.num_agents)+'a,'+str(self.arena.max_steps)+'t'
        if not os.path.exists(self.path ):
            os.mkdir(self.path)
        name=self.path+'/k'+str(self.arena.k)
        if not os.path.exists(name ):
            os.mkdir(name)
        while True:
            self.Npath=name+'/'+str(count)
            if not os.path.exists(self.Npath ):
                os.mkdir(self.Npath)
                break
            else:
                if os.path.exists(self.Npath+'/'+str(self.arena.num_runs)+'runs_HM.csv'):
                    hm = pd.read_csv(self.Npath+'/'+str(self.arena.num_runs)+'runs_HM.csv')
                    if self.r not in list(hm['r']):
                        break
                    count+=1
                else:
                    break
        for i in np.arange(0,1.05,.05):
            self.heat_map.update({round(i,2):0})

    def update(self,X):
        sum = 0
        T=[]
        for i in self.arena.tree.get_leaf_nodes():
            T.append(i.id)
        is_new = True
        for a in X.committed_agents:
            if a is not None:
                sum += 1
        flag1 = sum/self.arena.num_agents
        if flag1==1:
            flag1=0.99
        for j in np.arange(0,1.05,.05):
            if flag1>=j and flag1<j+.05:
                self.heat_map.update({round(j,2):self.heat_map.get(round(j,2))+1})
                break
        for i in os.listdir(self.Npath):
            if os.path.join(self.Npath, i)==self.Npath+'/rt'+str(self.arena.rec_time)+'_'+str(self.arena.num_runs)+'runs_location.csv':
                is_new=False
        with open(self.Npath+'/rt'+str(self.arena.rec_time)+'_'+str(self.arena.num_runs)+'runs_location.csv','a') as f:
            fieldnames1 = ['seed','run_id','r','agent_id','locations','on_chosen_point','best','distances']
            writer = csv.DictWriter(f,fieldnames=fieldnames1)
            if is_new:
                writer.writeheader()
                writer.writerow({'seed':self.arena.rec_time,'run_id':self.arena.max_steps,'r':list(T),'agent_id':'NaN','locations':'NaN','on_chosen_point':'NaN','best':'NaN','distances':'NaN'})
            for i in self.arena.agent_nodes_record.keys():
                writer.writerow({'seed':self.arena.random_seed,'run_id':self.arena.run_id,'r': self.r,'agent_id':i,'locations':list(self.arena.agent_nodes_record.get(i)),'on_chosen_point':list(self.arena.mean_on_chosen_point.get(i)),'best':str(X.id),'distances':list(self.arena.agent_distances_record.get(i))})
        with open(self.Npath+'/rt'+str(self.arena.rec_time)+'_'+str(self.arena.num_runs)+'runs_estimate.csv','a') as f:
            fieldnames1 = ['run_id', 'estimate','node_id','r','best']
            writer = csv.DictWriter(f,fieldnames=fieldnames1)
            if is_new:
                writer.writeheader()
                writer.writerow({'run_id':'NaN','estimate':'NaN','node_id':'NaN','r':self.arena.rec_time,'best':'NaN'})
            for i in self.arena.mean_rec_util.keys():
                writer.writerow({'run_id':self.arena.run_id,'estimate':list(self.arena.mean_rec_util.get(i)),'node_id':i,'r': self.r,'best':str(X.id)})
        print('Register updated')

    def print_mean_on_file(self):
        is_new = True
        for i in os.listdir(self.Npath):
            if os.path.join(self.Npath, i)==os.path.join(self.Npath+'/'+str(self.arena.num_runs)+'runs_HM.csv'):
                is_new=False
        with open(self.Npath+'/'+str(self.arena.num_runs)+'runs_HM.csv','a') as file:
            s = 0
            fieldnames2 = ['r','x','N']
            for i in self.heat_map.keys():
                writer = csv.DictWriter(file, fieldnames=fieldnames2)
                if is_new and s==0:
                    writer.writeheader()
                    s=1
                writer.writerow({'r': self.r,'x':i,'N':self.heat_map.get(i)})
        print('Mean on '+str(self.arena.num_runs)+' runs printed on file')
