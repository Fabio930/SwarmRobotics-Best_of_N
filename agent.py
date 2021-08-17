# -*- coding: utf-8 -*-
import sys,copy,math,random
import numpy as np
import logging as lg
from main.vectors import Vec2d
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

    num_agents = 0
    arena = None
    alpha = .8
    beta = .025
    k = .3
    h = .7
    comm_range = 'limited'

    class Factory:
        def create(self, config_element, arena): return Agent(config_element, arena)

    ##########################################################################
    # Initialisation of the Agent class
    def __init__(self, config_element, arena):

        # identification
        self.id = Agent.num_agents

        self.P_a=0.5
        if config_element.attrib.get("P_a") is not None:
            a = float(config_element.attrib["P_a"])
            if a < 0 or a >1:
                print ("[ERROR] for tag <agent> in configuration file the parameter <P_a> should be in [0,1]")
                sys.exit(2)
            self.P_a = a

        self.P_d=0.5
        if config_element.attrib.get("P_d") is not None:
            d = float(config_element.attrib["P_d"])
            if d < 0 or d >1:
                print ("[ERROR] for tag <agent> in configuration file the parameter <P_d> should be in [0,1]")
                sys.exit(2)
            self.P_d = d
        # agent class' attributes
        if self.id == 0:
            Agent.arena = arena

            # parse custon parameters from configuration file
            Agent.size = 0.033 if config_element.attrib.get("agent_size") is None else float(config_element.attrib["agent_size"])

            if config_element.attrib.get("comm_range") is not None:
                a = config_element.attrib["comm_range"]
                if a=='limited' or a=='unlimited':
                    Agent.comm_range = a
                else:
                    print ("[ERROR] for tag <agent> in configuration file the parameter <comm_range> is not valid")
                    sys.exit(2)

            Agent.init_Pa=self.P_a
            Agent.init_Pd=self.P_d

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

            if config_element.attrib.get("alpha") is not None:
                a = float(config_element.attrib["alpha"])
                if a<0 or a>1:
                    print ("[ERROR] for tag <agent> in configuration file the parameter <alpha> should be in (0,1]")
                    sys.exit(2)
                Agent.alpha = a

            if config_element.attrib.get("beta") is not None:
                a = float(config_element.attrib["beta"])
                if a <0 or a>1:
                    print ("[ERROR] for tag <agent> in configuration file the parameter <beta> should be in (0,1]")
                    sys.exit(2)
                Agent.beta = a

        # state (0=descending,1=ascending)
        self.state = np.random.choice([0,1])

        # position is the id of the current node
        self.current_node = 0
        self.prev_node = 0
        self.sensing = None

        self.position=Vec2d(0,0)
        self.prev_pos=Vec2d(0,0)

        # store initial positions for reset
        self.init_pos = Vec2d(0,0)

        # world representation
        Tree.num_nodes = 0
        self.tree = Tree(Agent.arena.tree_branches,Agent.arena.tree_depth,Agent.arena.num_agents,0,0,1,Agent.alpha,Agent.beta,Agent.arena.arena_dimension,Agent.arena.structure)
        for n in range(self.tree.num_nodes+1):
            node=self.tree.catch_node(n)
            node.copy_corners(Agent.arena.tree.catch_node(n))
        self.init_tree = copy.deepcopy(self.tree)

        Agent.num_agents += 1

    ##########################################################################

    def get_a_b(self):
        return Agent.alpha,Agent.beta

    ##########################################################################
    # generic init function brings back to initial positions
    def init_experiment( self ):
        self.tree = copy.deepcopy(self.init_tree)
        self.state = np.random.choice([0,1])
        self.current_node = 0
        self.prev_node = 0
        self.sensing = None
        self.P_a=Agent.init_Pa
        self.P_d=Agent.init_Pd
        self.position=self.init_pos

    ##########################################################################
    #  update the world model and choose the next state of the agent and the desired motion
    def control(self):
        node = Agent.arena.tree.catch_node(self.current_node)
        if node.child_nodes[0] is not None:
            if self.tree.catch_node(self.current_node).child_nodes[0] is None:
                for n in node.child_nodes:
                    self.tree.catch_node(self.current_node).add_child(n,Agent.beta)

        self.P_a=(len(Agent.arena.get_neighbor_agents(self,2))+1)/self.arena.num_agents
        self.P_d=self.P_a

        p = np.random.uniform(0,1)
        if self.state==0 and p<self.P_a:
            self.state = 1
        elif self.state==1 and p<self.P_d:
            self.state = 0

        if Agent.comm_range == 'limited':
            self.neighbors = Agent.arena.get_neighbor_agents(self,self.state)
            # self.update_neighbors_position()
        else:
            self.neighbors = Agent.arena.get_all_agents(self)
            # self.update_neighbors_position()

    ##########################################################################
    # propagates the info gathered in the current node to the parent nodes

    def update_world_utility(self,sensed_utility,leaf_id,ref):
        leaf = self.tree.catch_node(leaf_id)
        if ref == None:
            if leaf is None:
                leaf = self.tree.catch_node(self.sensing)
            leaf.filter.update_utility(sensed_utility)
            if leaf.parent_node is not None:
                flag=0
                for c in leaf.parent_node.child_nodes:
                    if c.filter.utility is not None:
                        flag+=c.filter.utility
                leaf.parent_node.filter.update_utility(flag/len(leaf.parent_node.child_nodes))
                self.update_world_utility(sensed_utility,leaf_id,leaf.parent_node)
        else:
            if ref.parent_node is not None:
                flag=0
                for c in ref.parent_node.child_nodes:
                    if c.filter.utility is not None:
                        flag+=c.filter.utility
                ref.parent_node.filter.update_utility(flag/len(ref.parent_node.child_nodes))
                self.update_world_utility(sensed_utility,leaf_id,ref.parent_node)

    ##########################################################################
    # save position, choose a new one and update velocity
    def update(self):
        self.prev_node = self.current_node
        self.prev_pos = self.position
        self.update_position()
        self.move()

    ##########################################################################
    # updates the agent position in the tree structure
    def update_neighbors_position(self):
        for a in self.neighbors:
            actual = self.tree.catch_node(a[1])
            self.reset_committedAgents_list(self.tree.catch_node(0),a[0].id)
            actual.committed_agents[a[0].id] = a

    ##########################################################################
    def get_id_utility(self,pos):
        node = self.tree.catch_node(pos)
        return node.id, node.filter.utility

    def reset_committedAgents_list(self,node,agent_id):
        node.committed_agents[agent_id] = None
        if node.child_nodes[0] is not None:
            for i in node.child_nodes:
                self.reset_committedAgents_list(i,agent_id)

    ##########################################################################
    def move(self):
        node = self.tree.catch_node(self.current_node)
        # descending transition
        if self.state==0:
            commitment = 0
            if node.child_nodes[0] is not None:
                selected_node = np.random.choice(node.child_nodes)
                leaf_id, leaf_utility = Agent.arena.get_node_utility(selected_node.id)
                self.sensingC=selected_node.id
                self.sensing=self.sensingC
                self.update_world_utility(leaf_utility,leaf_id,None)
                if selected_node.filter.utility == None or selected_node.filter.utility < 0:
                    percent = 0
                elif selected_node.filter.utility < Agent.arena.MAX_utility:
                    percent = selected_node.filter.utility/Agent.arena.MAX_utility
                else:
                    percent = 1
                commitment = Agent.k * percent
            else:
                leaf_id, leaf_utility = Agent.arena.get_node_utility(node.id)
                self.sensingC=node.id
                self.sensing=self.sensingC
                self.update_world_utility(leaf_utility,leaf_id,None)
            agent_node = None

            if len(self.neighbors) > 0:
                id = np.random.choice(np.arange(len(self.neighbors)))
                agent = self.neighbors[id]
                agent_node = node.get_sub_node(agent[1])
            recruitment = 0

            if agent_node is not None:
                leaf_id,leaf_utility=agent[0].get_id_utility(agent[1])
                self.sensing=agent_node.id
                self.update_world_utility(leaf_utility,leaf_id,None)
                utility = self.tree.catch_node(self.sensing).filter.utility
                if utility <= 0:
                    percent = 0
                elif utility < Agent.arena.MAX_utility:
                    percent = utility/Agent.arena.MAX_utility
                else:
                    percent = 1
                recruitment = Agent.h * percent

            p = np.random.uniform(0,1)
            if p < commitment:
                self.current_node = self.sensingC

            elif p < commitment + recruitment:
                self.current_node = agent_node.id

        # ascending transition
        else:
            leaf_id, leaf_utility = Agent.arena.get_node_utility(node.id)
            self.sensing=node.id
            self.update_world_utility(leaf_utility,leaf_id,None)
            if node.parent_node is not None:
                utility = node.filter.utility
                if utility <= 0:
                    percent = 1
                else:
                    percent = 1/(1 + utility)
                abandonment = Agent.k * percent
                agent_nodeC = None
                if len(self.neighbors) > 0:
                    id = np.random.choice(np.arange(len(self.neighbors)))
                    agent = self.neighbors[id]
                    agent_nodeC = node.get_sibling_node(agent[1])
                cross_inhibition = 0
                if agent_nodeC is not None:
                    leaf_id,leaf_utility=agent[0].get_id_utility(agent[1])
                    self.sensing=agent_nodeC.id
                    self.update_world_utility(leaf_utility,leaf_id,None)
                    utility = self.tree.catch_node(self.sensing).filter.utility
                    if utility<= 0:
                        percent = 0
                    elif utility< Agent.arena.MAX_utility:
                        percent = utility/Agent.arena.MAX_utility
                    else:
                        percent = 1
                    cross_inhibition = Agent.h * percent
                p = np.random.uniform(0,1)
                if p < abandonment + cross_inhibition:
                    self.current_node = node.parent_node.id

    ##########################################################################
    # Update the position and velocity according to a desired velocity
    def update_position( self):
        node=self.tree.catch_node(self.current_node)
        self.position = Vec2d(node.tl_corner.__getitem__(0) + ((node.tr_corner.__getitem__(0)-node.tl_corner.__getitem__(0))*.5),node.tl_corner.__getitem__(1) + ((node.bl_corner.__getitem__(1)-node.tl_corner.__getitem__(1))*.5))
