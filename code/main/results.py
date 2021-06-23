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
        self.times = np.array([])
        self.indx = np.array([])
        self.r = (10*arena.agents[0].h) / (10*arena.agents[0].k)
        self.arena = arena
        self.base = os.path.abspath("results"+'_'+arena.structure)
        if not os.path.exists(self.base ):
            os.mkdir(self.base )

    def update(self):
        sum = 0
        for a in self.arena.tree.catch_best_lnode().committed_agents:
            if a is not None:
                sum += 1
        flag = 0
        for i in self.heat_map.keys():
            if i==sum/self.arena.num_agents:
                flag = 1
                self.heat_map.update({sum/self.arena.num_agents:self.heat_map.get(i)+1})
        if flag == 0:
            self.heat_map.update({sum/self.arena.num_agents:1})
        if self.arena.time is not None:
            self.times = np.append(self.times,self.arena.time[0])
            self.indx = np.append(self.indx,self.arena.time[1])
        else:
            self.times = np.append(self.times,0)
            self.indx = np.append(self.indx,'inf')
        print('Register updated')

    def print_mean_on_file(self):
        path = self.base+'/K'+ str(self.arena.tree_branches) +'D'+ str(self.arena.tree_depth)+'_'+str(self.arena.num_agents)+'a,'+str(self.arena.max_steps)+'t,'+str(Agent.P_a)+'P_a'
        if not os.path.exists(path ):
            os.mkdir(path)
        Npath=path+'/k'+str(self.arena.k)
        is_new = False
        if not os.path.exists(Npath ):
            os.mkdir(Npath)
            is_new = True
        with open(Npath+'/HM.csv','a') as file:
            s = 0
            for i in self.heat_map.keys():
                fieldnames = ['r', 'x','N']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if is_new and s==0:
                    writer.writeheader()
                s+=1
                writer.writerow({'r': round(self.r,2), 'x':i, 'N':self.heat_map.get(i)})
        perc,bad=0,0
        for i in self.indx:
            if i == 'A':
                perc +=1
            if i == 'inf':
                bad +=1
        perc=perc/len(self.indx)
        bad=bad/len(self.indx)
        with open(Npath+'/times.csv','a') as f:
            fieldnames =['t','good%','bad%','r','alpha']
            writer = csv.DictWriter(f,fieldnames=fieldnames)
            if is_new:
                writer.writeheader()
            if len(self.times)>0:
                writer.writerow({'t':self.times.mean(),'good%':perc,'bad%':bad,'r':round(self.r,2),'alpha':self.arena.agents[0].alpha})
            else:
                print('ERROR while printing on file')
        print('Mean on '+str(self.arena.num_runs)+' runs printed on file')
