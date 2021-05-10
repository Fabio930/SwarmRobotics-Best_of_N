# -*- coding: utf-8 -*-

import numpy as np
import os, math
from main.arena import Arena
from matplotlib import pyplot as plt

class Results:
    'A class to store the results '

    def __init__( self, arena ):

        self.population_on_best_node = np.array([])
        self.time = np.array([])
        self.r = arena.agents[0].h/arena.agents[0].k
        self.arena = arena
        self.base = os.path.abspath("results")
        if not os.path.exists(self.base ):
            os.mkdir(self.base )

    def update(self):
        sum = 0
        for a in self.arena.tree.catch_best_lnode()[0].committed_agents:
            if a is not None:
                sum += 1
        self.population_on_best_node = np.append(self.population_on_best_node,sum/self.arena.num_agents)
        self.time = np.append(self.time,self.arena.num_steps)
        print('Register updated')

    def print_mean_on_file(self):
        path = self.base+'/K'+ str(self.arena.tree_branches) +'D'+ str(self.arena.tree_depth)+'_'+str(self.arena.num_agents)+'a'+','+str(self.arena.max_steps)
        if not os.path.exists(path ):
            os.mkdir(path)
        x=open(path+'/Pop_on_best_node_'+str(self.arena.tree.catch_best_lnode()[0].utility)+','+str(self.arena.num_targets)+'.txt','a')
        x.write(str(self.population_on_best_node.mean())+'\n')
        x.close()
        x=open(path+'/Times_'+str(self.arena.tree.catch_best_lnode()[0].utility)+','+str(self.arena.num_targets)+'.txt','a')
        x.write(str(self.time.mean())+'\n')
        x.close()
        x=open(path+'/r_'+str(self.arena.tree.catch_best_lnode()[0].utility)+','+str(self.arena.num_targets)+'.txt','a')
        x.write(str(self.r)+'\n')
        x.close()
        print('Mean on '+str(self.arena.num_runs)+' runs printed on file')
