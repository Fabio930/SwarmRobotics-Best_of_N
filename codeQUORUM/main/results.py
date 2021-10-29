# -*- coding: utf-8 -*-

import numpy as np
import os, math, csv
from main.arena import Arena
from main.agent import Agent
from matplotlib import pyplot as plt

class Results:
    'A class to store the results '

    def __init__( self, arena ):

        self.heat_map = {}
        self.r = (10*arena.agents[0].h) / (10*arena.agents[0].k)
        self.arena = arena
        self.base = os.path.abspath("results"+'_'+arena.structure)
        if not os.path.exists(self.base ):
            os.mkdir(self.base )
        self.path = self.base+'/K'+ str(self.arena.tree_branches) +'D'+ str(self.arena.tree_depth)+'_'+str(self.arena.num_agents)+'a,'+str(self.arena.max_steps)+'t'
        if not os.path.exists(self.path ):
            os.mkdir(self.path)
        self.Npath=self.path+'/k'+str(self.arena.k)
        if not os.path.exists(self.Npath ):
            os.mkdir(self.Npath)
        for i in np.arange(0,1.05,.05):
            self.heat_map.update({round(i,2):0})

    def update(self):
        sum = 0
        is_new = True
        for a in self.arena.tree.catch_best_lnode().committed_agents:
            if a is not None:
                sum += 1
        flag1 = sum/self.arena.num_agents
        if flag1==1:
            flag1=0.99
        for j in np.arange(0,1.05,.05):
            if flag1>=j and flag1<j+.05:
                self.heat_map.update({round(j,2):self.heat_map.get(round(j,2))+1})
                break
        if self.arena.run_id==1 or self.arena.run_id%10==0:
            for i in os.listdir(self.Npath):
                if os.path.join(self.Npath, i)==os.path.join(self.Npath+'/location.csv'):
                    is_new=False
            with open(self.Npath+'/location.csv','a') as f:
                fieldnames1 = ['id', 'locations','r']
                writer = csv.DictWriter(f,fieldnames=fieldnames1)
                if is_new:
                    writer.writeheader()
                for i in self.arena.agent_nodes_record.keys():
                    writer.writerow({'id':i,'locations':self.arena.agent_nodes_record.get(i),'r': round(self.r,2)})
        print('Register updated')

    def print_mean_on_file(self):
        is_new = True
        for i in os.listdir(self.Npath):
            if os.path.join(self.Npath, i)==os.path.join(self.Npath+'/HM.csv'):
                is_new=False
        with open(self.Npath+'/HM.csv','a') as file:
            s = 0
            fieldnames2 = ['r','x','N']
            for i in self.heat_map.keys():
                writer = csv.DictWriter(file, fieldnames=fieldnames2)
                if is_new and s==0:
                    writer.writeheader()
                    s=1
                writer.writerow({'r': round(self.r,2),'x':i,'N':self.heat_map.get(i)})
        print('Mean on '+str(self.arena.num_runs)+' runs printed on file')
