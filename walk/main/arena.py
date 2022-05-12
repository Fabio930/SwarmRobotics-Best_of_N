# -*- coding: utf-8 -*-
import sys, random, copy,math,time,curses
import numpy as np
from main.agent import AgentFactory
from main.tree import Tree
from main.vectors import Vec2d
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
        self.random_seed = 0
        self.timestep_length = 1 if config_element.attrib.get("timestep_length") is None else float(config_element.attrib.get("timestep_length"))
        self.arena_dimension = 1
        ssize = config_element.attrib.get("size");
        if ssize is not None:
            if float(ssize)>0:
                self.arena_dimension = float(ssize)
            else:
                print ("[WARNING] attribute 'size' in tag <arena> is not valid, initialization to 1")
        self.debug=False
        if config_element.attrib.get("debug") is not None:
            d = config_element.attrib["debug"]
            if d=='y':
                self.debug=True
        if config_element.attrib.get("num_agents") is None:
            print ("[ERROR] missing attribute 'num_agents' in tag <arena>")
            sys.exit(1)
        self.num_agents = int(config_element.attrib["num_agents"])
        if config_element.attrib.get("MAX_utility") is not None:
            self.MAX_utility = float(config_element.attrib["MAX_utility"])
            if self.MAX_utility < 0:
                print ("[ERROR] attribute 'MAX_utility' in tag <arena> must be in [0,)")
                sys.exit(1)
        self.MAX_std = 0
        if config_element.attrib.get("MAX_std") is not None:
            self.MAX_std = float(config_element.attrib["MAX_std"])
            if self.MAX_std < 0:
                print ("[ERROR] attribute 'MAX_std' in tag <arena> must be in [0,)")
                sys.exit(1)
        self.k = 1
        if config_element.attrib.get("k") is not None:
            self.k = float(config_element.attrib["k"])
            if self.k<0 or self.k>1:
                print ("[ERROR] attribute 'k' in tag <arena> must be in [0,1]")
                sys.exit(1)
        # number of runs to execute
        self.num_runs = 1 if config_element.attrib.get("num_runs") is None else int(config_element.attrib["num_runs"])
        self.run_id = 0
        # current simulation step and max number of simulation steps - 0 means no limits
        self.num_steps = 0
        self.max_steps = 0 if config_element.attrib.get("max_steps") is None else int(config_element.attrib["max_steps"])
        self.rec_time=1 if config_element.attrib.get("rec_time") is None else int(config_element.attrib["rec_time"])
        self.agents = []
        self.tree_branches = 2
        if config_element.attrib.get("tree_branches") is not None:
            tree_branches = int(config_element.attrib["tree_branches"])
            if tree_branches < 2:
                print ("[ERROR] attribute 'tree_branches' in tag <arena> must be greater than 1")
                sys.exit(2)
            self.tree_branches = tree_branches
        self.tree_depth = 1
        if config_element.attrib.get("tree_depth") is not None:
            tree_depth = int(config_element.attrib["tree_depth"])
            if tree_depth < 1:
                print ("[ERROR] attribute 'depth' in tag <arena> must be greater than 1")
                sys.exit(2)
            self.tree_depth = tree_depth
        self.num_nodes = 0
        for i in range(self.tree_depth+1):
            self.num_nodes += self.tree_branches**i
        self.tree = Tree(self.tree_branches,self.tree_depth,self.num_agents,self.MAX_utility,self.MAX_std,self.k,0,self.arena_dimension)
        self.tree.assign_random_MAXutil()
        self.tree.update_tree_utility()
        self.tree.adjust_arena(self.tree_branches)
        self.tree.arrange_root_kernel()
        self.tree_copy = copy.deepcopy(self.tree)
        self.create_agents(config_element)
    ##########################################################################
    # create the agents
    def create_agents( self, config_element ):
        # Get the tree correspnding to agent parameters
        agent_config= config_element.find('agent')
        if agent_config is None:
            print ("[ERROR] required tag <agent> in configuration file is missing")
            sys.exit(1)
        # dynamically load the desired module
        lib_pkg    = agent_config.attrib.get("pkg")
        if lib_pkg is not None:
            importlib.import_module(lib_pkg + ".agent", lib_pkg)
        for i in range(0,self.num_agents):
            self.agents.append(AgentFactory.create_agent(agent_config, self))

    ##########################################################################
    # assign agents to root tree of the tree and update their world representation
    def initialize_agents(self,leafs):
        for a in self.agents:
            temp_util={}
            self.tree.committed_agents[a.id] = a
            sum=0
            for l in leafs:
                node = a.tree.catch_node(l.id)
                if node.filter.utility is not None:
                    sum += l.utility_mean-node.filter.utility
                    temp_util.update({l.id:round(l.utility_mean-node.filter.utility,3)})
                else:
                    sum += l.utility_mean
                    temp_util.update({l.id:round(l.utility_mean,3)})
            sum=sum/len(leafs)
            temp_util.update({0:round(sum,3)})
            self.rec_util.update({a.id:temp_util})
        for l in self.rec_util.get(0).keys():
            sum=0
            for a in self.rec_util.keys():
                sum+=self.rec_util.get(a).get(l)
            sum=sum/len(self.agents)
            self.swarm_rec_util.update({l:[sum]})
            self.mean_rec_util.update({l:[sum]})
        print(str(self.num_agents)+' Agents initialized in the root node')

    ##########################################################################
    def update_rec_util(self,leafs):
        for agent in self.agents:
            self.mean_on_chosen_point.update({agent.id:np.append(self.mean_on_chosen_point.get(agent.id),agent.on_chosen_point)})
            self.agent_nodes_record.update({agent.id:np.append(self.agent_nodes_record.get(agent.id),agent.current_node)})
            self.agent_distances_record.update({agent.id:np.append(self.agent_distances_record.get(agent.id),self.tree.catch_node(agent.current_node).distance_from_opt)})
            sum=0
            for l in leafs:
                node=agent.tree.catch_node(l.id)
                if node.filter.utility is not None:
                    sum += l.utility_mean-node.filter.utility
                    self.rec_util.get(agent.id).update({l.id:round(l.utility_mean-node.filter.utility,3)})
                else:
                    sum += l.utility_mean-0
                    self.rec_util.get(agent.id).update({l.id:round(l.utility_mean-0,3)})
            sum=sum/len(leafs)
            self.rec_util.get(agent.id).update({0:round(sum,3)})
        for l in self.rec_util.get(0).keys():
            sum=0
            for a in self.rec_util.keys():
                sum+=self.rec_util.get(a).get(l)
            sum=sum/len(self.agents)
            self.swarm_rec_util.update({l:np.append(self.swarm_rec_util.get(l),round(sum,3))})
            self.mean_rec_util.update({l:np.append(self.mean_rec_util.get(l),round(self.swarm_rec_util.get(l).mean(),3))})

    ##########################################################################
    # set the random seed
    def set_random_seed( self , seed = None):
        if seed is not None:
            random.seed(seed)
        elif self.random_seed != 0:
            random.seed(self.random_seed)
        else:
            random.seed()

    ##########################################################################
    # initialisation/reset of the experiment variables
    def init_experiment( self ):
        try:
            r_val_stamp=str(self.agents[0].R())
        except:
            r_val_stamp='None'
        print('Experiment started. r: '+r_val_stamp+', k: '+str(self.k)+', tree depth:'+str(self.tree_depth)+', tree branches:'+str(self.tree_branches))
        self.num_steps = 0
        self.tree = copy.deepcopy(self.tree_copy)
        self.leafs = self.tree.get_leaf_nodes()
        self.agent_nodes_record = {}
        self.agent_distances_record = {}
        self.rec_util={}
        self.rcd_permission = True if self.rec_time>0 and self.max_steps>0 else False
        self.mean_rec_util={}
        self.swarm_rec_util={}
        self.mean_on_chosen_point={}
        self.random_seed = self.run_id*np.random.choice(np.arange(1,100))
        self.set_random_seed()
        for a in self.agents:
            self.agent_nodes_record.update({a.id:[0]})
            self.agent_distances_record.update({a.id:[0]})
            self.mean_on_chosen_point.update({a.id:[0]})
            a.calc_init_pos(self.agents)
            a.init_experiment()
        self.initialize_agents(self.leafs)

    ##########################################################################
    # run experiment until finished
    def run_experiment( self ):
        while not self.experiment_finished():
            self.update()
            print('Running...    '+str(self.num_steps), end=" steps\r", flush=True)

    ##########################################################################
    # updates the simulation state
    def update( self ):
        for a in self.agents:
            neighbors=self.get_neighbor_agents(a)
            a.update(neighbors)
            prev_node = self.tree.catch_node(a.prev_node)
            node = self.tree.catch_node(a.current_node)
            prev_node.committed_agents[a.id] = None
            node.committed_agents[a.id] = a
            self.update_agent_position(a,neighbors) if a.collision else self.update_agent_position(a)

        if self.rcd_permission and self.num_steps%self.rec_time==0:
            self.update_rec_util(self.leafs)
        if self.debug:
            for a in self.agents:
                print('A_id:'+str(a.id)+', Commit_node:'+str(a.committed)+', Prev_node:'+str(a.prev_node)+', Curr_node:'+str(a.current_node)+', # messages:'+str(len(a.messages.keys()))+', action:'+str(a.action)+', p:'+str(a.p)+', val:'+str(a.value)+', r:'+str(a.r))
            print('---------------------------------------------------------------')
        self.num_steps += 1

    ##########################################################################
    # determines if an exeperiment is finished
    def experiment_finished( self):
        if (self.max_steps > 0) and (self.max_steps <= self.num_steps):
            print("Run finished")
            return 1

    ##########################################################################
    def update_agent_position(self,a,neighbors=[]):
        positions=[]
        for n in neighbors:
            if a.id!=n.id:
                positions.append(n.position)
        a.move(positions)
        a.calc_error_pos()

    ##########################################################################
    # return the list of agents
    def get_neighbor_agents( self, agent):
        neighbor_list = []
        for a in self.agents:
            if a.id != agent.id and a.position.get_distance(agent.position)<=agent.comm_distance:
                neighbor_list.append(a)
        return neighbor_list

    ##########################################################################
    # return a the utility of a random leaf node with his id
    def get_node_utility(self,node_id,position):
        node = self.tree.catch_node(node_id)
        if node.child_nodes[0] is not None:
            for child in node.child_nodes:
                if position.isin(child.tl_corner,child.br_corner):
                    return self.get_node_utility(child.id,position)
        else:
            return node.id , self.tree.gaussian_kernel[int(position.__getitem__(1)/.01)-1][int(position.__getitem__(0)/.01)-1]
