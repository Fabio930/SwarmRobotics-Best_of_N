# -*- coding: utf-8 -*-

import random
import numpy as np

##########################################################################
# main tree class
class Tree:

    num_nodes = 0

    ##########################################################################
    # standart class init
    def __init__(self, num_childs,depth,num_agents):

        self.x, self.y = 0,0

        self.id = Tree.num_nodes
        self.utility = 0

        self.committed_agents = [None]*num_agents
        self.targets =  np.array([])

        self.parent_node = None
        self.child_nodes = [None]*num_childs
        if depth > 0:
            for c in range(num_childs):
                Tree.num_nodes += 1
                child = Tree(num_childs,depth-1,num_agents)
                child.parent_node = self
                self.child_nodes[c] = child

    ##########################################################################
    #  returns a random leaf from the relative sub_tree
    def get_random_leaf(self):
        if self.child_nodes[0] is not None:
            return np.random.choice(self.child_nodes).get_random_leaf()
        return self

    ##########################################################################
    #  search the node with id=node_id starting from the caller node
    #  if the node is in the sub_tree return it, otherwise return none
    def catch_node(self,node_id):
        if self.id == node_id:
            return self
        if self.child_nodes[0] is not None:
            for c in self.child_nodes:
                child = c.catch_node(node_id)
                if child is not None:
                    return child
        return None

    ##########################################################################
    def get_sub_node(self,node_id):
        for c in self.child_nodes:
            if c.catch_node(node_id) is not None:
                return c
        return None

    def get_sibling_node(self,node_id):
        for c in self.parent_node.child_nodes:
            if c.id != self.id:
                if c.catch_node(node_id) is not None:
                    return c
        return None

    ##########################################################################
    # updates the node utility looking at the utilities of the sub-tree
    def update_tree_utility(self):
        self.utility = 0
        if self.child_nodes[0] is not None:
            for c in self.child_nodes:
                c.update_tree_utility()
                self.utility += c.utility
            self.utility = self.utility/len(self.child_nodes)
        else:
            for t in self.targets:
                self.utility += 1

    ##########################################################################
    # check if all agents are in the same node
    def check_finish_condt(self):
        if self.id > 0:
            sum = 0
            for a in self.committed_agents:
                if a is not None:
                    sum += 1
            if sum == len(self.committed_agents):
                return True
        if self.child_nodes[0] is not None:
            for c in self.child_nodes:
                if c.check_finish_condt():
                    return True
        return False
