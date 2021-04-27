# -*- coding: utf-8 -*-

import random
import numpy as np

##########################################################################
# main tree class
class Tree:

    num_nodes = 0

    ##########################################################################
    # standart class init
    def __init__(self, childs,depth):

        self.id = Tree.num_nodes
        self.utility = 0

        self.committed_agents = np.array([])
        self.targets =  np.array([])

        #  probability transitions
        self.committment = 0
        self.recruitment = 0
        self.abandonment = 0
        self.cross_inhibition, self.self_inhibition = 0, 0

        self.parent_node = None
        self.child_nodes = np.array([])
        if depth > 0:
            for c in range(childs):
                Tree.num_nodes += 1
                child = Tree(childs,depth-1)
                child.parent_node = self
                self.child_nodes = np.append(self.child_nodes,child)

    ##########################################################################
    #  returns a random leaf from the relative sub_tree
    def get_random_leaf(self):
        if len(self.child_nodes) > 0:
            return np.random.choice(self.child_nodes).get_random_leaf()
        return self

    ##########################################################################
    # resets the transition probabilities of the node
    def reset_prob_transitions(self):
        self.committment = 0
        self.recruitment = 0
        self.abandonment = 0
        self.cross_inhibition, self.self_inhibition = 0, 0

    def prob(self):
        return self.committment+self.recruitment+self.abandonment+self.self_inhibition+self.cross_inhibition

    ##########################################################################
    #  search the node with id=node_id starting from the caller node
    #  if the node is in the sub_tree return it, otherwise return none
    def catch_node(self,node_id):
        if self.id == node_id:
            return self
        if len(self.child_nodes) > 0:
            for c in self.child_nodes:
                if c.id==node_id:
                    return c
            for c in self.child_nodes:
                child = c.catch_node(node_id)
                if child is not None:
                    return child
        return None

    ##########################################################################
    #  search the node with id=node_id in siblings nodes of the caller node
    #  if the node is there return it, otherwise return none
    def catch_sibling(self,node_id):
        if self.parent_node is not None:
            for c in self.parent_node.child_nodes:
                if c.id != self.id and c.id == node_id:
                    return c
        return None

    ##########################################################################
    #  search the node with id=node_id in the sub-trees of siblings nodes
    #  if the node is there return it, otherwise return none
    def catch_sibling_node(self,node_id):
        if self.parent_node is not None:
            for c in self.parent_node.child_nodes:
                if c.id != self.id:
                    if len(c.child_nodes) > 0:
                        for n in c.child_nodes:
                            nephew = n.catch_node(node_id)
                            if nephew is not None:
                                return nephew
        return None

    ##########################################################################
    # updates the node utility looking at the utilities of the sub-tree
    def update_tree_utility(self):
        self.utility = 0
        if len(self.child_nodes) > 0:
            for c in self.child_nodes:
                c.update_tree_utility()
                self.utility += c.utility
            self.utility = self.utility/len(self.child_nodes)
        else:
            for t in self.targets:
                self.utility += 1

    def get_max_utility(self):
        max = 0
        if len(self.child_nodes) > 0:
            for c in self.child_nodes:
                r = c.get_max_utility()
                if r > max:
                    max = r
        else:
            for t in self.targets:
                max += 1
        return max
    ##########################################################################
    # erase the agent from the relative committed list
    def erase_agent(self,agent):
        flag = None
        for i in range(len(self.committed_agents)):
            if self.committed_agents[i].id == agent.id:
                flag = i
        if flag is not None:
            self.committed_agents = np.delete(self.committed_agents,flag)
        else:
            if len(self.child_nodes) > 0:
                for c in self.child_nodes:
                    c.erase_agent(agent)
