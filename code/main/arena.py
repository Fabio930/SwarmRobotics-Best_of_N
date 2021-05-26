# -*- coding: utf-8 -*-

import random, copy,time
import numpy as np
from main.agent import AgentFactory
from main.tree import Tree

########################################################################################
## Arena
########################################################################################
# factory to dynamically create the arena
class ArenaFactory:
    factories = {}
    def add_factory(id, arena_factory):
        ArenaFactory.factories[id] = arena_factory
    add_factory = staticmethod(add_factory)

    def create_arena(config_element):
        arena_pkg = config_element.attrib.get("pkg")
        if arena_pkg is None:
            return Arena.Factory().create(config_element)
        id = arena_pkg + ".arena"
        return ArenaFactory.factories[id].create(config_element)
    create_arena = staticmethod(create_arena)

##########################################################################
# main arena class
class Arena:
    'this class manages the enviroment of the multi-agent simualtion'

    class Factory:
        def create(self, config_element): return Arena(config_element)

    ##########################################################################
    # standart class init
    def __init__( self, config_element ):

        # random seed
        self.random_seed = None if config_element.attrib.get("random_seed") is None else int(config_element.attrib["random_seed"])

        # number of agents to initialize
        if config_element.attrib.get("num_agents") is None:
            print ("[ERROR] missing attribute 'num_agents' in tag <arena>")
            sys.exit(2)
        self.num_agents = int(config_element.attrib["num_agents"])

        # maximum utility for leaf nodes
        self.MAX_utility = 10
        if config_element.attrib.get("MAX_utility") is not None:
            self.MAX_utility = float(config_element.attrib["MAX_utility"])
            if self.MAX_utility < 0:
                print ("[ERROR] attribute 'MAX_utility' in tag <arena> must be in [0,)")
                sys.exit(2)

        self.MAX_std = 0
        if config_element.attrib.get("MAX_std") is not None:
            self.MAX_std = float(config_element.attrib["MAX_std"])
            if self.MAX_std < 0:
                print ("[ERROR] attribute 'MAX_std' in tag <arena> must be in [0,)")
                sys.exit(2)

        self.v = 1
        if config_element.attrib.get("v") is not None:
            self.v = float(config_element.attrib["v"])
            if self.v < 1:
                print ("[ERROR] attribute 'v' in tag <arena> must be in [1,)")
                sys.exit(2)

        # number of runs to execute
        self.num_runs = 1 if config_element.attrib.get("num_runs") is None else int(config_element.attrib["num_runs"])
        self.run_id = 0

        # current simulation step and max number of simulation steps - 0 means no limits
        self.num_steps = 0
        self.max_steps = 0 if config_element.attrib.get("max_steps") is None else int(config_element.attrib["max_steps"])

        self.agents = np.array([])

        self.tree_branches = 2
        if config_element.attrib.get("tree_branches") is not None:
            tree_branches = int(config_element.attrib["tree_branches"])
            if tree_branches > 2:
                self.tree_branches = tree_branches

        self.tree_depth = 1
        if config_element.attrib.get("tree_depth") is not None:
            tree_depth = int(config_element.attrib["tree_depth"])
            if tree_depth > 1:
                self.tree_depth = tree_depth

        self.num_nodes = 0
        for i in range(self.tree_depth+1):
            self.num_nodes += self.tree_branches**i

        self.tree = Tree(self.tree_branches,self.tree_depth,self.num_agents,self.MAX_utility,self.MAX_std,self.v)

        self.create_agents(config_element)
        self.initialize_agents()
        self.tree.update_tree_utility()
        self.tree_copy = copy.deepcopy(self.tree)

    ##########################################################################
    # create the agents
    def create_agents( self, config_element ):
        # Get the tree correspnding to agent parameters
        agent_config= config_element.find('agent')
        if agent_config is None:
            print ("[ERROR] required tag <agent> in configuration file is missing")
            sys.exit(2)

        # dynamically load the desired module
        lib_pkg    = agent_config.attrib.get("pkg")
        if lib_pkg is not None:
            importlib.import_module(lib_pkg + ".agent", lib_pkg)

        for i in range(0,self.num_agents):
            self.agents = np.append(self.agents,AgentFactory.create_agent(agent_config, self))

    ##########################################################################
    # assign agents to root tree of the tree and update their world representation
    def initialize_agents(self):
        for a in self.agents:
            self.tree.committed_agents[a.id] = a
        print(str(self.num_agents)+' Agents initialized in the root node')

    ##########################################################################
    # set the random seed
    def set_random_seed( self, seed = None ):
        if seed is not None:
            random.seed(seed)
        elif self.random_seed != 0:
            random.seed(self.random_seed)
        else:
            random.seed()

    ##########################################################################
    # initialisation/reset of the experiment variables
    def init_experiment( self ):
        print('Experiment started')
        self.num_steps = 0
        self.tree = copy.deepcopy(self.tree_copy)
        for agent in self.agents:
            agent.init_experiment()

    ##########################################################################
    # run experiment until finished
    def run_experiment( self ):
        print('Running')
        while not self.experiment_finished():
            self.update()

    ##########################################################################
    # updates the simulation state
    def update( self ):
        # first, call the control() function for each agent,
        # which computes the desired motion and the next agent state
        for a in self.agents:
            a.control()

        # then, apply the desired motion and update the agent state
        for a in self.agents:
            a.update()
            prev_node = self.tree.catch_node(a.prev_position)
            prev_node.committed_agents[a.id] = None
            node = self.tree.catch_node(a.position)
            node.committed_agents[a.id] = a

        self.num_steps += 1

    ##########################################################################
    # determines if an exeperiment is finished
    def experiment_finished( self):
        if (self.max_steps > 0) and (self.max_steps <= self.num_steps):
            print("Run finished")
            return 1

    ##########################################################################
    # return the list of agents
    def get_neighbour_agents( self, agent, ref='infinite'):
        neighbour_list = []
        if ref == 'infinite':
            for a in self.agents:
                if a is not agent :
                    neighbour_list.append(a)
        else:
            for a in self.agents:
                if a is not agent :
                    if agent.is_neighbour(a):
                        neighbour_list.append(a)
        return neighbour_list

    ##########################################################################
    # return a the utility of a random leaf node with his id
    def get_node_utility(self,node_id):
        node = self.tree.catch_node(node_id)
        if node.child_nodes[0] is not None:
            selected_node = np.random.choice(node.child_nodes)
            return self.get_node_utility(selected_node.id)
        else:
            value = np.random.normal(self.tree.catch_node(node_id).utility_mean,self.tree.catch_node(node_id).utility_std)
            if value < 0:
                value = 0
            elif value > self.MAX_utility:
                value = self.MAX_utility
            return node_id , value
