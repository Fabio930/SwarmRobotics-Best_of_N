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
    arena = None
    alpha = .8
    beta = .025
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

        # agent class' attributes
        if self.id == 0:
            Agent.arena = arena
            if config_element.attrib.get("P_a") is not None:
                a = float(config_element.attrib["P_a"])
                if a < 0 or a >1:
                    print ("[ERROR] for tag <agent> in configuration file the parameter <P_a> should be in [0,1]")
                    sys.exit(2)
                Agent.P_a = a

            if config_element.attrib.get("P_d") is not None:
                d = float(config_element.attrib["P_d"])
                if d < 0 or d >1:
                    print ("[ERROR] for tag <agent> in configuration file the parameter <P_d> should be in [0,1]")
                    sys.exit(2)
                Agent.P_d = d

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

            if Agent.mode == 'log':
                lg.basicConfig(filename='history.log',
                            filemode='a',
                            format='%(asctime)s - %(levelname)s: %(message)s',
                            datefmt='%H:%M:%S',
                            level= lg.INFO)
        # state (0=descending,1=ascending)
        self.state = np.random.choice([0,1])

        # position is the id of the current node
        self.position = 0
        self.prev_position = 0
        self.sensing = None

        # world representation
        Tree.num_nodes = 0
        self.tree = Tree(Agent.arena.tree_branches,Agent.arena.tree_depth,Agent.arena.num_agents,0,0,1,Agent.alpha,Agent.beta,Agent.arena.structure)
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
        self.position = 0
        self.prev_position = 0
        self.sensing = None

    ##########################################################################
    # update the utilities of the world model and choose the next state of the agent
    def control(self):
        node = Agent.arena.tree.catch_node(self.position)
        if node.child_nodes[0] is not None:
            if self.tree.catch_node(self.position).child_nodes[0] is None:
                for n in node.child_nodes:
                    self.tree.catch_node(self.position).add_child(n,Agent.beta)
        p = np.random.uniform(0,1)
        if self.state==0 and p<Agent.P_a:
            self.state = 1
            # print(self.state,self.id,'++++++++++++')
        elif self.state==1 and p<Agent.P_d:
            self.state = 0

        if self.position==0:
            self.state=0
        else:
            leafs=self.tree.get_leaf_nodes()
            for l in leafs:
                if self.position==l.id:
                    self.state=1
                    break
            # print(self.state,self.id,'------------')
        # lg.info(f'Agent #{self.id} is in new state {self.state}.')

    ##########################################################################
    # propagates the info gathered in the current node to the parent nodes

    def update_world_utility(self,sensed_utility,leaf_id,ref):
        leaf = self.tree.catch_node(leaf_id)
        if ref == None:
            if leaf is None:
                leaf = self.tree.catch_node(self.sensing)
            leaf.filter.update_utility(sensed_utility)
            if leaf.parent_node is not None:
                for c in leaf.parent_node.child_nodes:
                    if c is not None:
                        leaf.parent_node.filter.update_utility(c.filter.utility)
                self.update_world_utility(sensed_utility,leaf_id,leaf.parent_node)
        else:
            if ref.parent_node is not None:
                for c in ref.parent_node.child_nodes:
                    if c is not None:
                        ref.parent_node.filter.update_utility(c.filter.utility)
                self.update_world_utility(sensed_utility,leaf_id,ref.parent_node)


    ##########################################################################
    # save position and choose a new one
    def update(self):
        self.prev_position = self.position
        if self.state == 0:
            neighborsD = Agent.arena.get_neighbor_agentsD(self)
            self.update_neighbors_position(neighborsD)
            self.descending(neighborsD)
        else:
            neighborsA = Agent.arena.get_neighbor_agentsA(self)
            self.update_neighbors_position(neighborsA)
            self.ascending(neighborsA)

    ##########################################################################
    # updates the agent position in the tree structure
    def update_neighbors_position(self,agents):
        for a in agents:
            actual = self.tree.catch_node(a[0].position)
            self.reset_committedAgents_list(self.tree.catch_node(0),a[0].id)
            if actual is not None:
                actual.committed_agents[a[0].id] = a
            else:
                actual = self.tree.catch_node(a[1])
                actual.committed_agents[a[0].id] = a


        # lg.info(f'Agent #{self.id} from node {self.position} sees {len(agents)} agents.')

    ##########################################################################
    def get_id_utility(self):
        node = self.tree.catch_node(self.position)
        return node.id, node.filter.utility

    def reset_committedAgents_list(self,node,agent_id):
        node.committed_agents[agent_id] = None
        if node.child_nodes[0] is not None:
            for i in node.child_nodes:
                self.reset_committedAgents_list(i,agent_id)

    ##########################################################################
    # descending transition
    def descending(self,agents):
        commitment = 0
        node = self.tree.catch_node(self.position)
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

        if len(agents) > 0:
            id = np.random.choice(np.arange(len(agents)))
            agent = agents[id]
            agent_node = node.get_sub_node(agent[1])
        recruitment = 0

        if agent_node is not None:
            leaf_id,leaf_utility=agent[0].get_id_utility()
            self.sensing=agent_node.id
            self.update_world_utility(leaf_utility,leaf_id,None)
            utility = self.tree.catch_node(self.sensing).filter.utility
            if utility<= 0:
                percent = 0
            elif utility < Agent.arena.MAX_utility:
                percent = utility/Agent.arena.MAX_utility
            else:
                percent = 1
            recruitment = Agent.h * percent
        p = np.random.uniform(0,1)
        if p < commitment:
            self.position = self.sensingC
            # print('committed',self.id,self.prev_position,'to',self.position,'c',commitment)
            # lg.info(f'Agent #{self.id} in node {self.prev_position} is committed to node {selected_node.id}.')
        elif p < commitment + recruitment:
            self.position = agent_node.id
        #     print('recruited',self.id,self.prev_position,'to',self.position,'r',recruitment)
        # else:
        #     print('fermo',self.id,self.position,'c',commitment,'r',recruitment)
#     lg.info(f'Agent #{self.id} in node {self.prev_position} is recruited to node {agent_node.id}.')
        # else:
        #     lg.info(f'Agent #{self.id} stays in node {self.position}.')


    ##########################################################################
    # ascending transition
    def ascending(self,agents):
        node = self.tree.catch_node(self.position)
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
            agent_node = None
            if len(agents) > 0:
                id = np.random.choice(np.arange(len(agents)))
                agent = agents[id]
                agent_node = node.get_sibling_node(agent[1])
            cross_inhibition = 0
            if agent_node is not None:
                leaf_id,leaf_utility=agent[0].get_id_utility()
                self.sensing=agent_node.id
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
                self.position = node.parent_node.id
                # print('abandon',self.id,self.prev_position,'to',self.position,'a',abandonment,'ci',cross_inhibition)
            # else:
                # print('fermo',self.id,self.position,'a',abandonment,'ci',cross_inhibition)
            #     lg.info(f'Agent #{self.id} leaves node {self.prev_position} to parent_node {self.position}.')
            # else:
            #     lg.info(f'Agent #{self.id} stays in node {self.position}.')
