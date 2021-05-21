# -*- coding: utf-8 -*-

import numpy as np
import os, math, csv
from main.arena import Arena
from matplotlib import pyplot as plt

class Results:
    'A class to store the results '

    def __init__( self, arena ):

        self.heat_map = {}
        self.r = (10*arena.agents[0].h) / (10*arena.agents[0].k)
        self.arena = arena
        self.base = os.path.abspath("results")
        if not os.path.exists(self.base ):
            os.mkdir(self.base )

    def update(self):
        sum = 0
        for a in self.arena.tree.catch_best_lnode()[0].committed_agents:
            if a is not None:
                sum += 1
        flag = 0
        for i in self.heat_map.keys():
            if i==sum/self.arena.num_agents:
                flag = 1
                self.heat_map.update({sum/self.arena.num_agents:self.heat_map.get(i)+1})
        if flag == 0:
            self.heat_map.update({sum/self.arena.num_agents:1})
        print('Register updated')

    def print_mean_on_file(self):
        path = self.base+'/K'+ str(self.arena.tree_branches) +'D'+ str(self.arena.tree_depth)+'_'+str(self.arena.num_agents)+'a,'+str(self.arena.max_steps)
        if not os.path.exists(path ):
            os.mkdir(path)
        path=path+'/v'+str(round(self.tree.catch_best_lnode()[0].utility/((self.num_targets-self.tree.catch_best_lnode()[0].utility)/(len(self.tree.get_leaf_nodes())-1)),2))
        is_new = False
        if not os.path.exists(path ):
            os.mkdir(path)
            is_new = True
        with open(path+'/HM.csv','a') as file:
            s = 0
            for i in self.heat_map.keys():
                fieldnames = ['r', 'x','N']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if is_new and s==0:
                    writer.writeheader()
                s+=1
                writer.writerow({'r': round(self.r,2), 'x':i, 'N':self.heat_map.get(i)})
        print('Mean on '+str(self.arena.num_runs)+' runs printed on file')
