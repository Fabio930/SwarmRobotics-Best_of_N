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
        self.alpha = 0

        if self.id == 0:
            # select the mode ('normal' or 'log')
            if config_element.attrib.get("mode") is not None:
                if config_element.attrib.get("mode") == 'log':
                    Agent.mode = 'log'

            # reference to the arena
            Agent.arena = arena

            # agent class' attributes
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

            if Agent.P_a + Agent.P_d > 1:
                print ("[ERROR] for tag <agent> in configuration file the sum <P_a+P_d> should be in [0,1]")
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

            if config_element.attrib.get("alpha") is not None:
                self.alpha = float(config_element.attrib["alpha"])
                if self.alpha<0 or self.alpha>1:
                    print ("[ERROR] for tag <agent> in configuration file the parameter <alpha> should be in (0,1]")
                    sys.exit(2)
            else:
                print ("[ERROR] for tag <agent> in configuration file the parameter <alpha> is missing")
                sys.exit(2)

        # state (0=descending,1=ascending)
        self.state = 0

        # position is the id of the current node
        self.position = 0
        self.prev_position = 0
        self.next_pos = 0

        # world representation
        Tree.num_nodes = 0
        self.tree = Tree(self.arena.tree_branches,self.arena.tree_depth,self.arena.num_agents,0,0,1,self.alpha,self.arena.structure)
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
        self.next_pos = 0
        self.prev_position = 0

    ##########################################################################
    # update the utilities of the world model and choose the next state of the agent
    def control(self,ref):
        node = self.arena.tree.catch_node(self.position)
        leaf_id, leaf_utility = Agent.arena.get_node_utility(node.id)
        temp = Agent.arena.tree.catch_node(self.position).get_sub_node(leaf_id)
        if temp is not None:
            self.next_pos = temp.id

        if ref =='known':
            self.update_world_utilities(self.tree.catch_node(leaf_id),leaf_utility)
        else:
            self.update_node_utility(leaf_utility,leaf_id,None)
        p = np.random.uniform(0,1)
        if self.state == 0 and p < Agent.P_a:
            self.state = 1
        elif self.state ==1 and p < Agent.P_d:
            self.state = 0

        if Agent.mode == 'log':
            lg.info(f'Agent #{self.id} is in new state {self.state}.')

    ##########################################################################
    # propagates the info gathered in the current node to the parent nodes
    def update_world_utilities(self,node,sensed_utility):
        if node.parent_node is not None:
            utility = 0
            for c in node.parent_node.child_nodes:
                # print(self.id,',',c.id,c.filter.utility)

                if c.id==node.id:
                    c.filter.update_utility(sensed_utility)
                    utility += c.filter.utility
                else:
                    utility += c.filter.utility
            node.parent_node.filter.overwrite_utility(utility/len(node.parent_node.child_nodes))
            self.update_world_utilities(node.parent_node,sensed_utility)

    def update_node_utility(self,sensed_utility,leaf_id,ref):
        next = self.tree.catch_node(self.next_pos)
        leaf = self.tree.catch_node(leaf_id)
        if leaf is not None:
            self.update_world_utilities(leaf,sensed_utility)
        else:
            if next is None:
                self.tree.catch_node(self.position).add_child(Agent.arena.tree.catch_node(self.next_pos))
            node = self.tree.catch_node(self.next_pos)
            if ref == None:
                node.filter.update_utility(sensed_utility)
                # print(node.id,node.filter.utility)
                if node.parent_node is not None:
                    utility = 0
                    for c in node.parent_node.child_nodes:
                        if c is not None:
                            utility += c.filter.utility
                    if utility > 0:
                        node.parent_node.filter.overwrite_utility(utility/len(node.parent_node.child_nodes))
                        self.update_node_utility(sensed_utility,leaf_id,node.parent_node)
            else:
                if ref.parent_node is not None:
                    utility = 0
                    for c in ref.parent_node.child_nodes:
                        if c is not None:
                            utility += c.filter.utility
                    if utility > 0:
                        ref.parent_node.filter.overwrite_utility(utility/len(ref.parent_node.child_nodes))
                        self.update_node_utility(sensed_utility,leaf_id,ref.parent_node)


    ##########################################################################
    # save position and choose a new one
    def update(self,ref):
        self.prev_position = self.position
        neighbours = Agent.arena.get_neighbour_agents(self)
        self.update_neighbours_position(neighbours,ref)
        if self.state == 0:
            self.descending(neighbours)
        else:
            self.ascending(neighbours)

    ##########################################################################
    # updates the agent position in the tree structure
    def update_neighbours_position(self,agents,ref):
        if ref =='known':
            for a in agents[0]:
                prev_node = self.tree.catch_node(a.prev_position)
                prev_node.committed_agents[a.id] = None
                node = self.tree.catch_node(a.position)
                node.committed_agents[a.id] = a
        else:
            for a in agents[0]:
                prev_node = self.tree.catch_node(a.prev_position)
                if prev_node is not None:
                    prev_node.committed_agents[a.id] = None
                node = self.tree.catch_node(a.position)
                if node is not None:
                    node.committed_agents[a.id] = a

        if Agent.mode == 'log':
            lg.info(f'Agent #{self.id} from node {self.position} sees {len(agents[0])} agents.')

    ##########################################################################
    # descending transition
    def descending(self,agents):
        selected_node=self.tree.catch_node(self.next_pos)
        if selected_node.filter.utility < 0:
            percent = 0
        elif selected_node.filter.utility < Agent.arena.MAX_utility:
            percent = selected_node.filter.utility/Agent.arena.MAX_utility
        else:
            percent = 1
        # print(percent,'dk',self.id,self.position,self.next_pos)
        committment = Agent.k * percent
        agent_node = None

        if len(agents[0]) > 0:
            agent = np.random.choice(agents[0])
            arena_node = Agent.arena.tree.catch_node(self.position)
            agent_node = arena_node.get_sub_node(agent.position)
        recruitment = 0

        if agent_node is not None:
            utility=agent.tree.catch_node(agent.position).filter.utility
            if utility<= 0:
                percent = 0
            elif utility < Agent.arena.MAX_utility:
                percent = utility/Agent.arena.MAX_utility
            else:
                percent = 1
            # print(percent,'dh',self.id,self.position,agent.position)
            recruitment = Agent.h * percent
        p = np.random.uniform(0,1)
        if p < committment:
            self.position = selected_node.id
            # print('committed',self.position)
            if Agent.mode == 'log':
                lg.info(f'Agent #{self.id} in node {self.prev_position} is committed to node {selected_node.id}.')
        elif p < committment + recruitment:
            if self.tree.catch_node(agent_node.id) is None:
                self.tree.catch_node(self.position).add_child(agent.tree.catch_node(agent_node.id))
            self.position = agent_node.id
            # print('recruited',self.position)

            if Agent.mode == 'log':
                lg.info(f'Agent #{self.id} in node {self.prev_position} is recruited to node {agent_node.id}.')
        else:
            if Agent.mode == 'log':
                lg.info(f'Agent #{self.id} stays in node {self.position}.')


    ##########################################################################
    # ascending transition
    def ascending(self,agents):
        node = self.tree.catch_node(self.position)
        if node.parent_node is not None:
            utility = node.filter.utility
            if utility <= 0:
                percent = 1
            else:
                percent = 1/(1 + utility)
            # print(percent,'ak',self.id,self.position)
            abandonment = Agent.k * percent
            agent_node_cross = None
            if len(agents[0]) > 0:
                agent = np.random.choice(agents[0])
                agent_node_cross = Agent.arena.tree.catch_node(self.position).get_sibling_node(agent.position)
            cross_inhibition = 0
            if agent_node_cross is not None:
                utility = agent.tree.catch_node(agent.position).filter.utility
                if utility<= 0:
                    percent = 0
                elif utility< Agent.arena.MAX_utility:
                    percent = utility/Agent.arena.MAX_utility
                else:
                    percent = 1
                # print(percent,'ah',self.id,self.position,agent.position)
                cross_inhibition = Agent.h * percent
            p = np.random.uniform(0,1)
            if p < abandonment + cross_inhibition:
                self.position = node.parent_node.id
                # print('abandon',self.position)

                if Agent.mode == 'log':
                    lg.info(f'Agent #{self.id} leaves node {self.prev_position} to parent_node {self.position}.')
            else:
                if Agent.mode == 'log':
                    lg.info(f'Agent #{self.id} stays in node {self.position}.')
