# -*- coding: utf-8 -*-
import sys,copy,math
import numpy as np
import logging as lg
from main.tree import Tree
########################################################################################
## Agent
########################################################################################

##########################################################################
# factory to dynamically create agents
class AgentFactory:
    factories = {}
    def add_factory(id, agent_factory):
        AgentFactory.factories[id] = agent_factory
    add_factory = staticmethod(add_factory)

    def create_agent(config_element, arena):
        agent_pkg = config_element.attrib.get("pkg")
        if agent_pkg is None:
            return Agent.Factory().create(config_element, arena)
        id = agent_pkg + ".agent"
        agent_type = config_element.attrib.get("type")
        if agent_type is not None:
            id = agent_pkg + "." + agent_type + ".agent"
        return AgentFactory.factories[id].create(config_element, arena)
    create_agent = staticmethod(create_agent)


##########################################################################
# the main agent class
class Agent:

    mode = 'normal'
    num_agents = 0
    arena      = None
    k = .3
    h = .7
    size = 0.33
    P_a = 0.5
    P_d = 0.5

    class Factory:
        def create(self, config_element, arena): return Agent(config_element, arena)

    ##########################################################################
    # Initialisation of the Agent class
    def __init__(self, config_element, arena):

        # identification
        self.id = Agent.num_agents

        if self.id == 0:
            # select the mode ('normal' or 'log')
            if config_element.attrib.get("mode") is not None:
                if config_element.attrib.get("mode") == 'log':
                    Agent.mode = 'log'
            # reference to the arena
            Agent.arena = arena
            # agent class' attributes
            if config_element.attrib.get("prob_ascend") is not None:
                a = float(config_element.attrib["prob_ascend"])
                if a < 0 or a >1:
                    print ("[ERROR] for tag <agent> in configuration file the parameter <prob_ascend> should be in [0,1]")
                    sys.exit(2)
                Agent.P_a = a
            if config_element.attrib.get("prob_descend") is not None:
                d = float(config_element.attrib["prob_descend"])
                if d < 0 or d >1:
                    print ("[ERROR] for tag <agent> in configuration file the parameter <prob_descend> should be in [0,1]")
                    sys.exit(2)
                Agent.P_d = d
            if Agent.P_a + Agent.P_d > 1:
                print ("[ERROR] for tag <agent> in configuration file the sum <prob_ascend+prob_descend> should be in [0,1]")
                sys.exit(2)
            if config_element.attrib.get("k") is not None:
                k = float(config_element.attrib["k"])
                if k<=0 or k>1:
                    print ("[ERROR] for tag <agent> in configuration file the parameter <k> should be in (0,1]")
                    sys.exit(2)
                Agent.k = k
            if config_element.attrib.get("h") is not None:
                h = float(config_element.attrib["h"])
                if h<0 or h>1:
                    print ("[ERROR] for tag <agent> in configuration file the parameter <h> should be in [0,1]")
                    sys.exit(2)
                Agent.h = h
            if Agent.h + Agent.k > 1:
                print ("[ERROR] for tag <agent> in configuration file the sum <h+k> should be in (0,1]")
                sys.exit(2)

        # state (0=descending,1=ascending)
        self.state = 0

        # position is the id of the current node
        self.position = 0
        self.prev_position = 0

        # world representation
        self.tree = arena.get_tree_copy()
        self.init_tree = copy.deepcopy(self.tree)

        Agent.num_agents += 1

        if Agent.mode == 'log':
            lg.basicConfig(filename='history.log',
                        filemode='a',
                        format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%H:%M:%S',
                        level= lg.INFO)

    ##########################################################################
    # generic init function brings back to initial positions
    def init_experiment( self ):
        self.tree = copy.deepcopy(self.init_tree)
        self.state = 0
        self.position = 0
        self.prev_position = 0

    ##########################################################################
    # update the utilities of the world model and choose the next state of the agent
    def control(self):
        # lg.info(f'Agent #{self.id} is in node {self.position} with state {self.state}.')
        node = self.tree.catch_node(self.position)
        node.utility = Agent.arena.get_node_utility(node.id)
        self.update_world_utilities(node)
        if self.state == 0 and np.random.uniform(0,1) < Agent.P_a:
            self.state = 1
        elif self.state ==1 and np.random.uniform(0,1) < Agent.P_d:
            self.state = 0

        if Agent.mode == 'log':
            lg.info(f'Agent #{self.id} is in new state {self.state}.')

    ##########################################################################
    # propagates the info gathered in the current node to the parent nodes
    def update_world_utilities(self,node):
        if node.parent_node is not None:
            utility = 0
            for c in node.parent_node.child_nodes:
                utility += c.utility
            node.parent_node.utility = utility/len(node.parent_node.child_nodes)
            self.update_world_utilities(node.parent_node)

    ##########################################################################
    # save position and choose a new one
    def update(self):
        self.prev_position = self.position
        node = self.tree.catch_node(self.position)
        neighbours = Agent.arena.get_neighbour_agents(self,1)
        self.update_neighbours_position(neighbours)
        if self.state == 0:
            self.descending(neighbours,node)
        else:
            self.ascending(neighbours,node)

    ##########################################################################
    # updates the agent position in the tree structure
    def update_neighbours_position(self,agents):
        for a in agents:
            prev_node = self.tree.catch_node(a.prev_position)
            prev_node.committed_agents[a.id] = None
            node = self.tree.catch_node(a.position)
            node.committed_agents[a.id] = a

        if Agent.mode == 'log':
            lg.info(f'Agent #{self.id} from node {self.position} sees {len(agents)} agents.')

    ##########################################################################
    # descending transition
    def descending(self,agents,node):
        if node.child_nodes[0] is not None:
            selected_node = np.random.choice(node.child_nodes)
            committment = Agent.k * Agent.arena.get_node_utility(selected_node.id)/Agent.arena.max_targets_per_node
            agent_node = None
            if len(agents) > 0:
                agent = np.random.choice(agents)
                agent_node = node.get_sub_node(agent.position)
            recruitment = 0
            if agent_node is not None:
                recruitment = Agent.h * agent.tree.catch_node(self.position).utility/Agent.arena.max_targets_per_node
            p = np.random.uniform(0,1)
            if p < committment:
                self.position = selected_node.id

                if Agent.mode == 'log':
                    lg.info(f'Agent #{self.id} in node {self.prev_position} is committed to node {selected_node.id}.')
            elif p < committment + recruitment:
                self.position = agent_node.id

                if Agent.mode == 'log':
                    lg.info(f'Agent #{self.id} in node {self.prev_position} is recruited to node {agent_node.id}.')
            else:
                if Agent.mode == 'log':
                    lg.info(f'Agent #{self.id} stays in node {self.position}.')

    ##########################################################################
    # ascending transition
    def ascending(self,agents,node):
        if node.parent_node is not None:
            abandonment = 0 #Agent.k *(1-node.utility/Agent.arena.max_targets_per_node)
            # agent_node_self = None
            agent_node_cross = None
            if len(agents) > 0:
                agent = np.random.choice(agents)
                # agent_node_self = node.catch_node(agent.position)
                agent_node_cross = node.get_sibling_node(agent.position)
            self_inhibition,cross_inhibition = 0,0
            # if agent_node_self is not None:
            #     self_inhibition = 0 #Agent.h * agent.tree.catch_node(self.position).utility/Agent.arena.max_targets_per_node
            if agent_node_cross is not None:
                cross_inhibition = Agent.h * agent.tree.catch_node(agent_node_cross.id).utility/Agent.arena.max_targets_per_node
            p = np.random.uniform(0,1)
            if p < abandonment + self_inhibition + cross_inhibition:
                self.position = node.parent_node.id

                if Agent.mode == 'log':
                    lg.info(f'Agent #{self.id} leaves node {self.prev_position} to parent_node {self.position}.')
            else:
                if Agent.mode == 'log':
                    lg.info(f'Agent #{self.id} stays in node {self.position}.')

    ##########################################################################
    # check if an agent is a neighbour
    def is_neighbour(self,agent):
        node = self.tree.catch_node(self.position)
        if node.catch_node(agent.position) is not None:
            return True
        elif node.parent_node is not None:
            if node.parent_node.id == agent.position:
                return True
            elif node.get_sibling_node(agent.position) is not None:
                return True
        return False
