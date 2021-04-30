# -*- coding: utf-8 -*-
import sys,copy,math
import numpy as np
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
    arena      = None
    k = .8
    h = .2
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

            # reference to the arena
            Agent.arena = arena
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
        print("Agent "+str(self.id)+" is in node "+str(self.position))
        node = self.tree.catch_node(self.position)
        node.utility = Agent.arena.get_node_utility(node.id)
        self.update_world_utilities(node)
        if self.state == 0 and np.random.uniform(0,1) < Agent.P_a:
            self.state = 1
        elif self.state ==1 and np.random.uniform(0,1) < Agent.P_d:
            self.state = 0

    ##########################################################################
    def update_world_utilities(self,node):
        if node.parent_node is not None:
            utility = 0
            for c in node.parent_node.child_nodes:
                utility += c.utility
            node.parent_node.utility = utility/len(node.parent_node.child_nodes)
            self.update_world_utilities(node.parent_node)

    ##########################################################################
    # udpate the position
    def update(self):
        node = self.tree.catch_node(self.position) # nodo attuale
        neighbours = Agent.arena.get_neighbour_agents(self)
        self.update_neighbours_position(neighbours)
        if self.state == 0:
            self.descending(neighbours,node)
        else:
            self.ascending(neighbours,node)
        print("Agent "+str(self.id)+" is moving from "+str(self.prev_position)+" to "+str(self.position))

    ##########################################################################
    def update_neighbours_position(self,agents):
        for a in agents:
            self.tree.erase_agent(a)
            node = self.tree.catch_node(a.position)
            node.committed_agents = np.append(node.committed_agents,a)

    ##########################################################################
    def descending(self,agents,node):
        self.prev_position = self.position
        print("descending")

        if len(node.child_nodes) > 0:
            selected_node = np.random.choice(node.child_nodes)
            committment = Agent.k *  node.utility/Agent.arena.max_targets_per_node
            agent_node = None
            if len(agents) > 0:
                agent = np.random.choice(agents)
                agent_node = node.get_sub_node(agent.position)
            recruitment = 0
            # se l'agente si trova nel sotto-albero del mio nodo uso le sue informazioni
            # sulla zona da cui raggiungerlo per il recruitment
            if agent_node is not None:
                recruitment = Agent.h * agent.tree.catch_node(self.position).utility/Agent.arena.max_targets_per_node
            p = np.random.uniform(0,1)
            if p < committment:
                self.position = selected_node.id
                print("committed")
                print(p,committment,recruitment,'+++')
            # se sono reclutato mi muovo nella direzione dell'agente
            elif p < committment + recruitment:
                self.position = agent_node.id
                print("recruited")
                print(p,committment,recruitment,'+++')

    def ascending(self,agents,node):
        self.prev_position = self.position
        print("ascending")

        if node.parent_node is not None:
            abandonment = 0 #Agent.k *(1-node.utility/Agent.arena.max_targets_per_node)
            agent_node_self = None
            agent_node_cross = None
            if len(agents) > 0:
                agent = np.random.choice(agents)
                agent_node_self = node.catch_node(agent.position)
                agent_node_cross = node.get_sibling_node(agent.position)
            self_inhibition,cross_inhibition = 0,0
            # se l'agente si trova nel mio stesso nodo o in un nodo inferiore
            # uso le sue informazioni sulla mia posizione per la self_inhibition
            if agent_node_self is not None:
                self_inhibition = 0 #Agent.h * agent.tree.catch_node(self.position).utility/Agent.arena.max_targets_per_node
            # se l'agente si trova in un nodo con il nodo parente in comune al mio o nei relativi sotto-alberi
            # uso le sue informazioni sulla posizione con nodo parente in comune per la cross_inhibition
            elif agent_node_cross is not None:
                cross_inhibition = Agent.h * agent.tree.catch_node(agent_node_cross.id).utility/Agent.arena.max_targets_per_node
            p = np.random.uniform(0,1)
            if p < abandonment + self_inhibition + cross_inhibition:
                self.position = node.parent_node.id
                print("inhibited")
                print(p,cross_inhibition,'---')

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
