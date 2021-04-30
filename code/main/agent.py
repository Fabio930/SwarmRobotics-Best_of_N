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

    max_targets_per_node = 0
    max_agents_per_node = 0

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
            Agent.max_targets_per_node = arena.max_targets_per_node
            Agent.max_agents_per_node = arena.max_agents_per_node

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
                print ("[ERROR] for tag <agent> in configuration file the parameter <h+k> should be in (0,1]")
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
    # compute the desired motion and the next state of the agent
    def control(self):
        print("Agent "+str(self.id)+" is in node "+str(self.position))
        node = self.tree.catch_node(self.position) # nodo attuale
        neighbours = Agent.arena.get_neighbour_agents(self)
        self.update_neighbours_position(neighbours)
        self.state = np.random.randint(2)
        if self.state == 0:
            self.descending(neighbours,node)
        else:
            self.ascending(neighbours,node)

    ##########################################################################
    # generic init function brings back to initial positions
    def init_experiment( self ):
        self.tree = copy.deepcopy(self.init_tree)
        self.state = 0
        self.position = 0
        self.prev_position = 0

    ##########################################################################
    # functions for control routine
    def descending(self,agents,node):
        self.prev_position = self.position

        if len(node.child_nodes) > 0:
            selected_node = np.random.choice(node.child_nodes)
            p = abs(np.random.uniform(0,1))

            committment = Agent.k *  selected_node.utility/Agent.max_targets_per_node
            recruitment = 0
            agent = np.random.choice(agents)
            # se l'agente si trova nel sotto-albero del mio nodo uso le sue informazioni
            # sulla zona da cui raggiungerlo per il recruitment
            agent_node = node.get_sub_node(agent.position)

            if agent_node is not None:
                recruitment = Agent.h * agent.tree.catch_node(agent_node.id).utility/Agent.max_targets_per_node

            if p < committment:
                self.position = selected_node.id
            # se sono reclutato mi muovo nella direzione dell'agente
            elif agent_node is not None and p < committment + recruitment:
                self.position = agent_node.id

    def ascending(self,agents,node):
        self.prev_position = self.position

        if node.parent_node is not None:
            agent = np.random.choice(agents)
            p = abs(np.random.uniform(0,1))

            # abandonment basato sull'utilità del nodo corrente
            abandonment = 0 #Agent.k *(1-node.utility/Agent.max_targets_per_node)
            self_inhibition,cross_inhibition = 0,0
            agent_node = node.get_sibling_node(agent.position)

            # se l'agente si trova nel mio stesso nodo o in un nodo inferiore
            # uso le sue informazioni sulla mia posizione per la self_inhibition
            if node.catch_node(agent.position) is not None:
                if len(node.committed_agents) > 0.75*Agent.max_agents_per_node:
                    self_inhibition = 0 #Agent.h * agent.tree.catch_node(self.position).utility/Agent.max_targets_per_node

            # se l'agente si trova in un nodo con il nodo parente in comune al mio o nei relativi sotto-alberi
            # uso le sue informazioni sulla posizione con nodo parente in comune per la cross_inhibition
            elif agent_node is not None:
                if 0.25*Agent.max_agents_per_node > len(agent_node.committed_agents):
                    cross_inhibition = Agent.h * agent.tree.catch_node(agent_node.id).utility/Agent.max_targets_per_node

            if p < abandonment + self_inhibition + cross_inhibition:
                self.position = node.parent_node.id

    def update_neighbours_position(self,agents):
        for a in agents:
            self.tree.erase_agent(a)
            node = self.tree.catch_node(a.position)
            node.committed_agents = np.append(node.committed_agents,a)

    ##########################################################################
    # udpate of the world model after control routine
    def update(self):
        node = self.tree.catch_node(self.position)
        node.utility = Agent.arena.get_node_utility(node.id)
        self.update_utilities(node)

    def update_utilities(self,node):
        if node.parent_node is not None:
            utility = 0
            for c in node.parent_node.child_nodes:
                utility += c.utility
            node.parent_node.utility = utility/len(node.parent_node.child_nodes)
            self.update_utilities(node.parent_node)
