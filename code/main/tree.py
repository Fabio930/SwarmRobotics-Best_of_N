# -*- coding: utf-8 -*-

import random
import numpy as np
from main.utility_filter import Filter

##########################################################################
# main tree class
class Tree:

    num_nodes = 0
    s = 0
    ##########################################################################
    # standart class init
    def __init__(self, num_childs,depth,num_agents,MAX_utility,MAX_std,k,alpha,reg,ref = 'arena'):
        if ref == 'arena':
            self.x, self.y1, self.y2 = 0,0,0
            self.alpha=float(alpha)
            self.id = Tree.num_nodes
            self.utility_mean = 0
            self.utility_std = 0
            self.committed_agents = [None]*num_agents
            self.parent_node = None
            self.child_nodes = [None]*num_childs

            if depth > 0:
                for c in range(num_childs):
                    Tree.num_nodes += 1
                    child = Tree(num_childs,depth-1,num_agents,MAX_utility,MAX_std,k,alpha,reg,ref)
                    child.parent_node = self
                    self.child_nodes[c] = child
            else:
                if Tree.s == 0:
                    self.utility_mean = MAX_utility
                    self.utility_std = MAX_std
                    Tree.s = 1
                else:
                    self.utility_mean = MAX_utility*k
                    self.utility_std = MAX_std

        elif ref == 'known':
            self.x, self.y1, self.y2 = 0,0,0
            self.alpha=float(alpha)
            self.id = Tree.num_nodes
            self.committed_agents = [None]*num_agents
            self.parent_node = None
            self.child_nodes = [None]*num_childs
            self.filter = Filter(alpha)
            if depth > 0:
                if self.parent_node is not None:
                    flag = self.parent_node.filter.alpha - const
                    if flag >=0:
                        self.alpha = flag
                    else:
                        self.alpha = 0
                    self.filter.set_alpha(self.alpha)
                for c in range(num_childs):
                    Tree.num_nodes += 1
                    child = Tree(num_childs,depth-1,num_agents,MAX_utility,MAX_std,k,alpha,reg,ref)
                    child.parent_node = self
                    self.child_nodes[c] = child
        else:
            self.x, self.y1, self.y2 = 0,0,0
            self.alpha=float(alpha)
            self.id = 0
            self.committed_agents = [None]*num_agents
            self.parent_node = None
            self.child_nodes = [None]
            self.filter = Filter(alpha)

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
            for c in self.child_nodes:
                c.update_tree_utility()
                self.utility_mean += c.utility_mean
            self.utility_mean = self.utility_mean/len(self.child_nodes)

    ##########################################################################

    def add_child(self,selected_node,const):
        if self.child_nodes[0] is None:
            self.child_nodes[0]=Tree(0,0,len(selected_node.committed_agents),0,0,0,selected_node.alpha,const,'unknown')
        else:
            self.child_nodes.append(Tree(0,0,len(selected_node.committed_agents),0,0,0,selected_node.alpha,const,'unknown'))
        self.child_nodes[-1].parent_node = self
        self.child_nodes[-1].id = selected_node.id
        flag = self.filter.alpha - const
        if flag >=0:
            self.child_nodes[-1].alpha = flag
        else:
            self.child_nodes[-1].alpha = 0
        self.child_nodes[-1].filter.set_alpha(self.child_nodes[-1].alpha)
