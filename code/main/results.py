# -*- coding: utf-8 -*-

import numpy as np
import os, math
from main.arena import Arena
from matplotlib import pyplot as plt

class Results:
    'A class to store the results '

    def __init__( self, arena ):

        self.population_on_best_node = np.array([])
        self.r = arena.agents[0].h/arena.agents[0].k
        self.arena = arena
        self.base = os.path.abspath("results")
        if not os.path.exists(self.base ):
            os.mkdir(self.base )

    def update(self):
        self.population_on_best_node = np.append(self.population_on_best_node,len(self.arena.tree.catch_node(1).committed_agents)/self.arena.num_agents)
        print('Register updated')

    def print_mean_on_file(self):
        path = self.base+'/K'+ str(self.arena.tree_branches) +'D'+ str(self.arena.tree_depth)
        if not os.path.exists(path ):
            os.mkdir(path)
        r=open(path+'/R'+str(self.arena.tree.catch_node(1).utility)+','+str(self.arena.tree.catch_node(2).utility)+'.txt','a')
        x=open(path+'/agents_on_node'+str(self.arena.tree.catch_node(1).utility)+','+str(self.arena.tree.catch_node(2).utility)+'.txt','a')
        r.write(str(round(self.r,2))+'\n')
        x.write(str(self.population_on_best_node.mean())+'\n')
        r.close()
        x.close()
        print('Mean on '+str(self.arena.num_runs)+' runs printed on file')

    def plots(self):
        X=np.array([])
        Y=np.array([])
        path = self.base+'/K'+ str(self.arena.tree_branches) +'D'+ str(self.arena.tree_depth)
        r=open(path+'/R'+str(self.arena.tree.catch_node(1).utility)+','+str(self.arena.tree.catch_node(2).utility)+'.txt','r')
        x=open(path+'/agents_on_node'+str(self.arena.tree.catch_node(1).utility)+','+str(self.arena.tree.catch_node(2).utility)+'.txt','r')
        line1=r.readlines()
        line2=x.readlines()
        for l in line1:
            flag=''
            for c in l:
                if c != '\n':
                    flag+=c
            X = np.append(X,flag)
        for l in line2:
            flag=''
            for c in l:
                if c != '\n':
                    flag+=c
            Y = np.append(Y,float(flag))
        fig, ax = plt.subplots(figsize=(10,5))
        ax.errorbar(x=X, y=Y, color='blue')
        plt.axis([0, 10, 0, 1.1])
        plt.ylabel('x1')
        plt.xlabel('r')
        plt.savefig(path+'/'+str(self.arena.tree.catch_node(1).utility)+','+str(self.arena.tree.catch_node(2).utility)+'.png')
        # plt.show()
        plt.close()
        r.close()
        x.close()
