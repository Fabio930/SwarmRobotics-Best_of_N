# -*- coding: utf-8 -*-

import random
import numpy as np
from main.agent import AgentFactory
from main.target import TargetFactory
########################################################################################
## KDTree arena
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
            return Arena.Factory().create(config_element,None)
        id = arena_pkg + ".arena"
        return ArenaFactory.factories[id].create(config_element,None)
    create_arena = staticmethod(create_arena)


##########################################################################
# main arena class
class Arena:
    'this class manages the enviroment of the multi-agent simulation'
    config_element = None
    random_seed = None
    num_runs = 1
    run_id = 0

    num_steps = 0
    max_steps = 0
    timestep_length = 0.1

    depth = 1
    num_branches = 2
    agents = np.array([])
    num_agents = 0

    targets = np.array([])
    num_targets = 0

    class Factory:
        def create(self, config_element, parent): return Arena(config_element , parent)

    ##########################################################################
    # standart class init
    def __init__(self, config_element ,parent):
        self.branches = np.array([])
        self.id = [0,0] # ID for root node, in general id=[layer, node #]
        self.parent = None
        self.targets = np.array([])
        self.agents = np.array([])

        if parent is None:
            Arena.config_element = config_element
            # random seed
            if config_element.attrib.get("random_seed") is not None:
                Arena.random_seed = int(config_element.attrib["random_seed"])

            # number of agents to initialize
            if config_element.attrib.get("num_agents") is None:
                print ("[ERROR] missing attribute 'num_agents' in tag <arena>")
                sys.exit(2)
            Arena.num_agents = int(config_element.attrib["num_agents"])

            # number of targets to initialize
            if config_element.attrib.get("num_targets") is not None:
                Arena.num_targets = int(config_element.attrib["num_targets"])

            # number of runs to execute
            if config_element.attrib.get("num_runs") is not None:
                Arena.num_runs = int(config_element.attrib["num_runs"])

            # max number of simulation steps - 0 means no limits
            if config_element.attrib.get("max_steps") is not None:
                Arena.max_steps = int(config_element.attrib["max_steps"])

            # length of a simulation step (in seconds)
            if config_element.attrib.get("timestep_length") is not None:
                Arena.timestep_length = float(config_element.attrib["timestep_length"])

            # depth and branches of the tree structure (the minimum is a binary tree of 2 layers)
            conf = config_element.attrib.get("depth")
            if config_element.attrib.get("depth") is not None:
                if int(conf) > 1:
                    Arena.depth = int(conf)
            self.depth = Arena.depth

            conf = config_element.attrib.get("num_branches")
            if conf is not None:
                if int(conf) > 2:
                    Arena.num_branches = int(conf)
        else:
            self.parent = parent
            if len(parent.branches) == 0:
                self.id = [parent.id[0]+1,parent.id[1]*Arena.num_branches]
            else:
                flag = parent.branches[len(parent.branches)-1].id[1]
                self.id = [parent.id[0]+1,flag+1]
            self.depth = parent.depth - 1

        print("Node #"+str(self.id)+" initialized. Depth:"+str(self.depth))
        if self.depth > 0:
            self.create_branches()

    ##########################################################################
    # create the agents
    def create_agents(self, config_element):
        # Get the node correspnding to agent parameters
        agent_config= config_element.find('agent')
        if agent_config is None:
            print ("[ERROR] required tag <agent> in configuration file is missing")
            sys.exit(2)

        # dynamically load the desired module
        lib_pkg = agent_config.attrib.get("pkg")
        if lib_pkg is not None:
            importlib.import_module(lib_pkg + ".agent", lib_pkg)

        for i in range(0,Arena.num_agents):
            Arena.agents = np.append(Arena.agents,AgentFactory.create_agent(agent_config, self))

    ##########################################################################
    # create the branches
    def create_branches(self):
        for i in range(Arena.num_branches):
            self.branches = np.append(self.branches,Arena(Arena.config_element,self))


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

        for i in range(0,Arena.num_targets):
            Arena.targets = np.append(Arena.targets,TargetFactory.create_target(target_config, self))

    # assign targets to leaf nodes
    def assign_targets(self):
        for t in Arena.targets:
            if t.assigned[0] == False:
                if len(self.branches) > 0:
                    np.random.choice(self.branches).assign_targets()
                else:
                    sem = np.random.normal(.5,.3)
                    if sem > 1:
                        self.targets = np.append(self.targets,t)
                        t.assigned[0] = True
                        t.assigned[1] = self.id
                        print("Target #"+str(t.id)+" assigned to leaf #"+str(self.id))
                    else:
                        parent = self.parent
                        while parent.parent is not None:
                            parent = parent.parent
                        parent.assign_targets()



    ##########################################################################
    # set the random seed
    def set_random_seed( self, seed = None ):
        if seed is not None:
            random.seed(seed)
        elif Arena.random_seed != 0:
            random.seed(Arena.random_seed)
        else:
            random.seed()

    def get_seed(self):
        return Arena.random_seed


    ##########################################################################
    # initialisation/reset of the experiment variables
    def init_experiment( self ):
        Arena.num_steps = 0
        Arena.agents = np.array([])
        Arena.targets = np.array([])
        self.create_targets(Arena.config_element)
        self.assign_targets()
        self.create_agents(Arena.config_element)
        for a in Arena.agents:
            self.agents = np.append(self.agents,a)
        for agent in Arena.agents:
            agent.init_experiment()
        print("********************************************")

    ##########################################################################
    # run experiment until finished
    def run_experiment( self ):
        while not self.experiment_finished():
            self.update()


    ##########################################################################
    # updates the simulation state
    def update( self ):
        print('UPDATE ENVIRONMENT____________________')
        # first, call the control() function for each agent,
        # which computes the desired motion and the next agent state
        for a in Arena.agents:
            a.control()

        # then, apply the desired motion and update the agent state
        for a in Arena.agents:
            a.update()
        Arena.num_steps += 1

    def update_run_id(self,num_runs):
        Arena.run_id = num_runs

    def get_run_id(self):
        return Arena.run_id

    def get_num_runs(self):
        return Arena.num_runs

    ##########################################################################
    # determines if an exeperiment is finished
    def experiment_finished( self):
        return (Arena.max_steps > 0) and (Arena.max_steps <= Arena.num_steps)

    ##########################################################################
    # return the list of agents in the whole tree
    def get_neighbour_agents( self, id):
        neighbour_list = np.array([])
        for a in Arena.agents:
            if a.id != id:
                neighbour_list = np.append(neighbour_list,a)
        return neighbour_list

    ##########################################################################
    # returns the list of targets in the sub tree (they can be only in leaf nodes)
    def get_targets(self):
        targets = np.array([])
        if len(self.branches) > 0:
            for b in self.branches:
                flag=b.get_targets()
                for t in flag:
                    targets = np.append(targets,t)
        else:
            for t in self.targets:
                targets = np.append(targets,t)
        return targets


    ##########################################################################
    # returns the number of steps from the actual node to another
    def get_distance(self,node):
        distance = 0
        if self.id != node.id:
            flag = self.utility_distanceD(node)
            if flag[0]==True:
                distance += flag[1]
                return distance
            flag = self.utility_distanceU(node)
            if flag[0]==True:
                distance += flag[1]
                return distance
            if flag[0]==False:
                print("ERROR")
        return distance

    def utility_distanceD(self,node):
        if len(self.branches) > 0:
            for b in self.branches:
                if b.id == node.id:
                    return True ,1
            for b in self.branches:
                flag = b.utility_distanceD(node)
                if flag[0]==True:
                    return True , flag[1]+1
        return False , 0

    def utility_distanceU(self,node):
        distance = 1
        if self.parent.id == node.id:
            return True , distance
        flag = self.parent.utility_distanceD(node)
        if flag[0]==True:
            return True , flag[1]+distance
        flag = self.parent.utility_distanceU(node)
        if flag[0]==True:
            return True , flag[1]+distance
        return False, distance

    ##########################################################################
    # utility functions for agent's control subroutine
    def has_in_subtree(self,node):
        if len(self.branches) > 0:
            for b in self.branches:
                if b.id != self.id:
                    if b.id == node.id:
                        return True
            for b in self.branches:
                if b.has_in_subtree(node) == True:
                    return True
        return False

    def has_in_siblings(self,node):
        for b in self.parent.branches:
            if b.id != self.id:
                if b.id==node.id:
                    return True
        return False

    def has_in_subtree_of_siblings(self,node):
        for b in self.parent.branches:
            if b.id != self.id:
                if b.has_in_subtree(node) == True:
                    return True
        return False


    ##########################################################################
    # returns an univocous string representation of the node id for agent usage
    def get_id( self ):
        return str(self.id[0])+""+str(self.id[1])

    ##########################################################################
    # save results to file, if any
    def save_results( self ):
        return None
