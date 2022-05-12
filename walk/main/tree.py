# -*- coding: utf-8 -*-

import math,random
import numpy as np
import scipy.ndimage as st
from main.utility_filter import Filter
from main.vectors import Vec2d

##########################################################################
# main tree class
class Tree:

    num_nodes = 0
    r = 0
    ##########################################################################
    # standart class init
    def __init__(self, num_childs,depth,num_agents,MAX_utility,MAX_std,k,alpha,arena_dim,ref = 'arena'):
        self.tl_corner,self.tr_corner,self.bl_corner,self.br_corner= Vec2d(.01,.01),Vec2d(arena_dim+.01,.01),Vec2d(.01,arena_dim+.01),Vec2d(arena_dim+.01,arena_dim+.01)
        self.distance_from_opt=-1
        if Tree.r==0:
            Tree.childs=num_childs
            Tree.std=MAX_std
            Tree.MAX_utility=MAX_utility
            Tree.arena_dim=arena_dim
            Tree.r+=1
        self.depth=depth
        self.alpha=float(alpha)
        self.gaussian_kernel=None
        if ref == 'arena':
            self.x, self.y1, self.y2 = 0,0,0
            self.id = Tree.num_nodes
            self.utility_mean = 0
            self.utility_std = 0
            self.committed_agents = [None]*num_agents
            self.parent_node = None
            self.child_nodes = [None]*num_childs
            if depth > 0:
                for c in range(num_childs):
                    Tree.num_nodes += 1
                    child = Tree(num_childs,depth-1,num_agents,MAX_utility,MAX_std,k,alpha,arena_dim,ref)
                    child.parent_node = self
                    self.child_nodes[c] = child
            else:
                self.utility_mean = MAX_utility*k
                self.utility_std = MAX_std
                self.distance_from_opt=0
        else:
            self.x, self.y1, self.y2 = 0,0,0
            self.committed_agents = None
            self.parent_node = None
            self.id = Tree.num_nodes
            self.child_nodes = [None]*num_childs
            if depth==0:
                self.filter = Filter(alpha,1)
            else:
                self.filter = Filter(alpha)
            if depth > 0:
                if self.parent_node is not None:
                    self.filter.set_alpha(self.parent_node.filter.alpha)
                for c in range(num_childs):
                    Tree.num_nodes += 1
                    child = Tree(num_childs,depth-1,0,MAX_utility,MAX_std,k,alpha,arena_dim,ref)
                    child.parent_node = self
                    self.child_nodes[c] = child

    ##########################################################################
    def assign_random_MAXutil(self):
        leaf=self.get_random_leaf()
        leaf.utility_mean=Tree.MAX_utility
        leaf.utility_std=Tree.std
        for c in leaf.parent_node.child_nodes:
            if c.id!=leaf.id:
                c.distance_from_opt=1
        self.fix_distance_from_opt(leaf.parent_node,1)
            
    ##########################################################################
    def fix_distance_from_opt(self,leaf,dist_from_opt):
        if leaf.parent_node is not None:
            leafs=[]
            for c in leaf.parent_node.child_nodes:
                if c.id!=leaf.id:
                    for gc in c.get_leaf_nodes():
                        leafs.append(gc)
            for l in leafs:
                l.distance_from_opt=dist_from_opt+1
            self.fix_distance_from_opt(leaf.parent_node,dist_from_opt+1)

    ##########################################################################
    #  returns a random leaf from the relative sub_tree
    def get_random_leaf(self):
        if self.child_nodes[0] is not None:
            return np.random.choice(self.child_nodes).get_random_leaf()
        return self

    ##########################################################################
    #  returns a all leaf nodes from the relative sub_tree
    def get_leaf_nodes(self):
        leafs = np.array([])
        if self.child_nodes[0] is not None:
            for c in self.child_nodes:
                leafs = np.append(leafs,c.get_leaf_nodes())
        else:
            leafs = self
        return leafs

    ##########################################################################
    #  search the node with id=node_id starting from the caller node
    #  if the node is in the sub_tree return it, otherwise return none
    def catch_node(self,node_id):
        if self.id == node_id:
            return self
        elif self.child_nodes[0] is not None:
            for c in self.child_nodes:
                child = c.catch_node(node_id)
                if child is not None:
                    return child
        return None

    ##########################################################################
    #  search the leaf node with the max utility or the nodes in case of equalty
    def catch_best_lnode(self):
        leafs = self.get_leaf_nodes()
        MAX = 0
        pos = 0
        for l in range(len(leafs)):
            if leafs[l].utility_mean > MAX:
                MAX = leafs[l].utility_mean
                pos = l
        return leafs[pos]

    ##########################################################################
    def get_parent(self,node_id):
        if self.parent_node is not None:
            if self.parent_node.id == node_id:
                return self.parent_node
            return self.parent_node.get_parent(node_id)
        return None

    def get_sub_node(self,node_id):
        for c in self.child_nodes:
            if c is not None and c.catch_node(node_id) is not None:
                return c
        return None

    def get_sibling_node(self,node_id):
        if self.parent_node is not None:
            for c in self.parent_node.child_nodes:
                if c.id != self.id:
                    if c is not None and c.catch_node(node_id) is not None:
                        return c
        return None

    ##########################################################################
    # updates the node utility looking at the utilities of the sub-tree
    def update_tree_utility(self):
        if self.child_nodes[0] is not None:
            self.utility_mean=0
            for c in self.child_nodes:
                c.update_tree_utility()
                if self.utility_mean < c.utility_mean:
                    self.utility_mean = c.utility_mean

    ##########################################################################
    def copy_corners(self,node):
        self.tl_corner,self.tr_corner,self.bl_corner,self.br_corner= node.tl_corner,node.tr_corner,node.bl_corner,node.br_corner

    ##########################################################################
    def adjust_arena(self,branches,ref=None):
        w1=self.tl_corner.__getitem__(0)
        w2=self.tr_corner.__getitem__(0)
        h1=self.tr_corner.__getitem__(1)
        h2=self.br_corner.__getitem__(1)
        indx=0
        for i in range(1,9):
            if branches==4**i:
                indx=i
                break
        if indx!=0:
            # print(self.id,w1,w2,h1,h2)
            dif=(w2-w1)/(2**indx)
            h2=dif + h1
            w2=dif + w1
            if self.child_nodes[0] is not None:
                count=0
                for c in range(len(self.child_nodes)):
                    self.child_nodes[c].tl_corner=Vec2d(w1,h1)
                    self.child_nodes[c].tr_corner=Vec2d(w2,h1)
                    self.child_nodes[c].bl_corner=Vec2d(w1,h2)
                    self.child_nodes[c].br_corner=Vec2d(w2,h2)
                    self.child_nodes[c].adjust_arena(branches)
                    w1=w2
                    w2=w2+dif
                    count+=1
                    if count==2**indx:
                        count=0
                        w1=self.tl_corner.__getitem__(0)
                        w2=dif + w1
                        h1=h2
                        h2=dif + h1
        else:
            if ref==None:
                # print(self.id,w1,w2,h1,h2)
                dif=(w2-w1)/len(self.child_nodes)
                w2=self.tr_corner.__getitem__(0)/len(self.child_nodes) + w1
                for c in self.child_nodes:
                    c.tl_corner=Vec2d(w1,h1)
                    c.tr_corner=Vec2d(w2,h1)
                    c.bl_corner=Vec2d(w1,h2)
                    c.br_corner=Vec2d(w2,h2)
                    c.adjust_arena(1,1)
                    w1=w1+dif
                    w2=w2+dif
            else:
                if ref==1:
                    # print(self.id,w1,w2,h1,h2)
                    dif=(h2-h1)/len(self.child_nodes)
                    h2=h1+dif
                    if self.child_nodes[0] is not None:
                        for c in self.child_nodes:
                            c.tl_corner=Vec2d(w1,h1)
                            c.tr_corner=Vec2d(w2,h1)
                            c.bl_corner=Vec2d(w1,h2)
                            c.br_corner=Vec2d(w2,h2)
                            c.adjust_arena(1,0)
                            h1=h1+dif
                            h2=h2+dif
                else:
                    # print(self.id,w1,w2,h1,h2)
                    dif=(w2-w1)/len(self.child_nodes)
                    w2=w1+dif
                    if self.child_nodes[0] is not None:
                        for c in self.child_nodes:
                            c.tl_corner=Vec2d(w1,h1)
                            c.tr_corner=Vec2d(w2,h1)
                            c.bl_corner=Vec2d(w1,h2)
                            c.br_corner=Vec2d(w2,h2)
                            c.adjust_arena(1,1)
                            w1=w1+dif
                            w2=w2+dif

        if self.child_nodes[0] is None:
            X = np.arange(self.tl_corner.__getitem__(0), self.tr_corner.__getitem__(0), 0.01)
            Y = np.arange(self.tl_corner.__getitem__(1), self.bl_corner.__getitem__(1), 0.01)
            self.gaussian_kernel = np.zeros((len(Y),len(X)))
            for i in range(len(self.gaussian_kernel)):
                for j in range(len(self.gaussian_kernel[i])):
                    self.gaussian_kernel[i][j] = self.utility_mean + np.random.normal(0,self.utility_std)

    def arrange_root_kernel(self):
        leafs=self.get_leaf_nodes()
        Xl=self.tl_corner.__getitem__(0)
        Xr=self.tr_corner.__getitem__(0)
        Yt=self.tr_corner.__getitem__(1)
        Yb=self.br_corner.__getitem__(1)
        matrix=[]
        list=[]
        sem=0
        while True:
            for l in leafs:
                if l.tl_corner.__getitem__(0)>=Xl-.01 and l.tl_corner.__getitem__(0)<=Xl+.01 and l.tl_corner.__getitem__(1)>=Yt-.01 and l.tl_corner.__getitem__(1)<=Yt+.01:
                    Xl=l.tr_corner.__getitem__(0)
                    if Xl>=Xr:
                        Xl=self.tl_corner.__getitem__(0)
                        Yt=l.br_corner.__getitem__(1)
                        if Yt>=Yb:
                            sem=1
                        list=np.append(list,l.gaussian_kernel,axis=1)
                        if len(matrix)==0:
                            matrix=np.array(list)
                        else:
                            matrix=np.append(matrix,list,axis=0)
                        list=[]
                    else:
                        if len(list)==0:
                            list=np.array(l.gaussian_kernel)
                        else:
                            list=np.append(list,l.gaussian_kernel,axis=1)
            if sem==1:
                break
        self.gaussian_kernel=np.block(matrix)
