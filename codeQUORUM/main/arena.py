# -*- coding: utf-8 -*-
import sys, random, copy,math,time
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

        self.agent_nodes_record = {}
        # random seed
        self.random_seed = None if config_element.attrib.get("random_seed") is None else int(config_element.attrib["random_seed"])

        # arena_size
        ssize = config_element.attrib.get("size");
        if ssize is None:
            print ("[ERROR] missing attribute 'size' in tag <arena>")
            sys.exit(2)
        self.arena_dimension = float(ssize)

        self.debug=False
        if config_element.attrib.get("debug") is not None:
            d = config_element.attrib["debug"]
            if d=='y':
                self.debug=True

        self.structure = 'known'
        if config_element.attrib.get("structure") is not None:
            a = config_element.attrib["structure"]
            if a =='unknown':
                self.structure='unknown'

        # number of agents to initialize
        if config_element.attrib.get("num_agents") is None:
            print ("[ERROR] missing attribute 'num_agents' in tag <arena>")
            sys.exit(2)
        self.num_agents = int(config_element.attrib["num_agents"])

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

        self.k = 1
        if config_element.attrib.get("k") is not None:
            self.k = float(config_element.attrib["k"])
            if self.k<0 or self.k>1:
                print ("[ERROR] attribute 'k' in tag <arena> must be in [0,1]")
                sys.exit(2)

        # number of runs to execute
        self.num_runs = 1 if config_element.attrib.get("num_runs") is None else int(config_element.attrib["num_runs"])
        self.run_id = 0

        # current simulation step and max number of simulation steps - 0 means no limits
        self.num_steps = 0
        self.max_steps = 0 if config_element.attrib.get("max_steps") is None else int(config_element.attrib["max_steps"])

        # length of a simulation step (in seconds)
        self.timestep_length = 0.1 if config_element.attrib.get("timestep_length") is None else float(config_element.attrib["timestep_length"])

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

        self.tree = Tree(self.tree_branches,self.tree_depth,self.num_agents,self.MAX_utility,self.MAX_std,self.k,0,0,self.arena_dimension)

        self.tree.update_tree_utility()
        self.tree.adjust_arena(self.tree_branches)
        self.tree.arrange_root_kernel()
        self.tree_copy = copy.deepcopy(self.tree)

        self.create_agents(config_element)
        self.initialize_agents()
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
        print('Experiment started',', r: '+str(round(self.agents[0].h/self.agents[0].k,2)))
        self.num_steps = 0
        self.tree = copy.deepcopy(self.tree_copy)

        for a in self.agents:
            self.agent_nodes_record.update({a.id:np.array([])})
            a.init_pos=Vec2d(np.random.uniform(self.tree.tl_corner.__getitem__(0),self.tree.tr_corner.__getitem__(0)),np.random.uniform(self.tree.tl_corner.__getitem__(1),self.tree.br_corner.__getitem__(1)))
            a.init_experiment()

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
            if self.num_steps%10==0 or self.num_steps==0:
                self.agent_nodes_record.update({a.id:np.append(self.agent_nodes_record.get(a.id),a.current_node)})
            a.control()
        # then, apply the desired motion and update the agent state
        for a in self.agents:
            a.update()
            prev_node = self.tree.catch_node(a.prev_node)
            node = self.tree.catch_node(a.current_node)
            prev_node.committed_agents[a.id] = None
            node.committed_agents[a.id] = a

        if self.debug:
            for a in self.agents:
                print('A_id:'+str(a.id)+', Prev_node:'+str(a.prev_node)+', Curr_node:'+str(a.current_node)+', # neighbors:'+str(len(a.neighbors))+', state:'+str(a.state)+', action:'+str(a.action)+', p:'+str(a.p)+', val:'+str(a.value))
        self.num_steps += 1
    ##########################################################################
    # determines if an exeperiment is finished
    def experiment_finished( self):
        if (self.max_steps > 0) and (self.max_steps <= self.num_steps):
            print("Run finished")
            return 1

    ##########################################################################
    # return the list of agents
    def get_neighbor_agents( self, agent):
        neighbor_list = []
        node = self.tree.catch_node(agent.current_node)
        for a in self.agents:
            if a.id != agent.id :
                if a.current_node==agent.current_node:
                    neighbor_list.append([a,a.current_node])
                else:
                    flag = node.get_sub_node(a.current_node)
                    if flag is not None:
                        neighbor_list.append([a,flag.id])
                    else:
                        if node.parent_node is not None:
                            if node.parent_node.id == a.current_node:
                                neighbor_list.append([a,a.current_node])
                            elif node.get_sibling_node(a.current_node) is not None:
                                neighbor_list.append([a,node.get_sibling_node(a.current_node).id])
        return neighbor_list

    def get_all_agents(self,agent):
        neighbor_list=[]
        for a in self.agents:
            if a.id != agent.id :
                neighbor_list.append([a,a.current_node])
        return neighbor_list

    ##########################################################################
    def agents_commited_to_node(self,node):
        tot=0
        for a in node.committed_agents:
            if a is not None:
                tot+=1
        if node.child_nodes[0] is not None:
            for c in node.child_nodes:
                tot+=self.agents_commited_to_node(c)
        return tot

    ##########################################################################
    # return a the utility of a random leaf node with his id
    def get_node_utility(self,node_id):
        node = self.tree.catch_node(node_id)
        if node.child_nodes[0] is not None:
            return self.get_node_utility(np.random.choice(node.child_nodes).id)
        else:
            node = self.tree.catch_node(node_id)
            return node.id , node.utility_mean
