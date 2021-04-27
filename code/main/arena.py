# -*- coding: utf-8 -*-

import random, copy,time
import numpy as np
from main.agent import AgentFactory
from main.target import TargetFactory
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

        # number of targets to initialize
        self.num_targets = 1
        if config_element.attrib.get("num_targets") is not None:
            self.num_targets = int(config_element.attrib["num_targets"])

        self.max_agent_per_node = 2
        if config_element.attrib.get("Max_agents_node") is not None:
            self.max_agent_per_node = int(config_element.attrib["Max_agents_node"])
            if self.max_agent_per_node < 1:
                print ("[ERROR] invalid value for 'Max_agents_node' [1,) in tag <arena>")
                sys.exit(2)

        # number of runs to execute
        self.num_runs = 1 if config_element.attrib.get("num_runs") is None else int(config_element.attrib["num_runs"])
        self.run_id = 0

        # current simulation step and max number of simulation steps - 0 means no limits
        self.num_steps = 0
        self.max_steps = 0 if config_element.attrib.get("max_steps") is None else int(config_element.attrib["max_steps"])

        # length of a simulation step (in seconds)
        # self.timestep_length = 0.1 if config_element.attrib.get("timestep_length") is None else float(config_element.attrib["timestep_length"])

        self.agents = np.array([])
        self.targets = np.array([])

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

        self.tree = Tree(self.tree_branches,self.tree_depth)

        self.create_targets(config_element)
        self.rand_assign_targets()
        self.update_utility()

        time.sleep(2)

        self.create_agents(config_element)
        self.initialize_agents()

    ##########################################################################
    # create the targets
    def create_targets(self,config_element):
        target_config= config_element.find('target')
        if target_config is None:
            print ("[ERROR] required tag <target> in configuration file is missing")
            sys.exit(2)

        # dynamically load the desired module
        lib_pkg = target_config.attrib.get("pkg")
        if lib_pkg is not None:
            importlib.import_module(lib_pkg + ".target", lib_pkg)

        for i in range(0,self.num_targets):
            self.targets = np.append(self.targets,TargetFactory.create_target(target_config, self))

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
    # assign targets to random leaf nodes of the acrual tree
    def rand_assign_targets(self): #quando finisce ad assegnare i target?
        for t in self.targets:
            leaf = self.tree.get_random_leaf()
            leaf.targets = np.append(leaf.targets,t)
            t.assign = leaf.id
            print('target '+str(t.id)+' in node',leaf.id)

    ##########################################################################
    # assign agents to root tree of the tree and update their world representation
    def initialize_agents(self):
        for a in self.agents:
            self.tree.committed_agents = np.append(self.tree.committed_agents,a)
        for a in self.agents:
            a.tree = self.get_tree_copy()
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
        self.num_steps = 0
        for agent in self.agents:
            agent.init_experiment()

    ##########################################################################
    # run experiment until finished
    def run_experiment( self ):
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
            self.tree.erase_agent(a)
            node = self.tree.catch_node(a.position)
            node.committed_agents = np.append(node.committed_agents,a)

        self.num_steps += 1
        time.sleep(.3)

    ##########################################################################
    # determines if an exeperiment is finished
    def experiment_finished( self):

        return (self.max_steps > 0) and (self.max_steps <= self.num_steps)


    ##########################################################################
    # save results to file, if any
    def save_results( self ):
        return None


    ##########################################################################
    # return the list of agents
    def get_neighbour_agents( self, agent ):
        neighbour_list = []
        for a in self.agents:
            if a is not agent :
                neighbour_list.append(a)
        return neighbour_list

    ##########################################################################
    # return a copy of the Tree
    def get_tree_copy(self):
        return copy.deepcopy(self.tree)

    def get_node_utility(self,node_id):
        return self.tree.catch_node(node_id).utility

    def update_utility(self):
        self.tree.update_tree_utility()
        self.max_utility = self.tree.get_max_utility()
