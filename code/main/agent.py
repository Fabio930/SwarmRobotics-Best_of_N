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

        # state (0=descending,1=ascending)
        self.state = 0
        # move (0=stay,1=change area)
        self.move = 0

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
        node = self.tree.catch_node(self.position)
        agents = Agent.arena.get_neighbour_agents(self)
        self.update_neighbours_position(agents)
        node.reset_prob_transitions()
        random_agent = np.random.choice(agents)
        self.state = np.random.randint(2)
        if self.state == 0:
            self.update_descend_prob(random_agent,node)
        else:
            self.update_ascend_prob(random_agent,node)
        self.rand_wheel(node)
        self.new_position(node)
        print(node.utility,'______________________________')

    ##########################################################################
    # generic init function brings back to initial positions
    def init_experiment( self ):
        self.tree = copy.deepcopy(self.init_tree)
        self.state = 0
        self.position = 0
        self.prev_position = 0
        self.move = 0

    ##########################################################################
    # update functions for control routine
    def update_descend_prob(self,agent,node):
        print('Descending')
        sup = Agent.arena.max_utility
        node.committment = Agent.k *  node.utility/sup
        print('commit',node.committment)
        if agent.position!=self.position and node.catch_node(agent.position) is not None:
            agent_node = agent.tree.catch_node(self.position)
            node.recruitment = Agent.h * agent_node.utility/sup

    def update_ascend_prob(self,agent,node):
        max_agent_per_node = Agent.arena.max_agent_per_node
        sup = Agent.arena.max_utility
        print('Ascending')
        node.abandonment = Agent.k * (1 - node.utility/sup)
        print('abandon',node.abandonment)
        if self.position!=agent.position and node.catch_node(agent.position) is not None:
            agent_node = agent.tree.catch_node(self.position)
            node.self_inhibition = Agent.h * agent_node.utility/sup * np.heaviside(len(node.committed_agents),0.75*max_agent_per_node)
        elif node.catch_sibling(agent.position) is not None:
            agent_node = agent.tree.catch_node(self.position)
            node.self_inhibition = Agent.h * agent_node.utility/sup * np.heaviside(len(node.committed_agents),0.75*max_agent_per_node)
        elif node.catch_sibling_node(agent.position) is not None:
            agent_node = agent.tree.catch_node(agent.position)
            node.cross_inhibition = Agent.h * agent_node.utility/sup * np.heaviside(0.25*max_agent_per_node,len(node.committed_agents))

    def rand_wheel(self,node):
        self.move = 0
        p = abs(np.random.uniform(0,1))
        if p < node.prob():
            self.move = 1
        if self.move == 1:
            print('moving')
        else:
            print('I stay here')


    def new_position(self,node):
        self.prev_position = self.position
        if self.move == 1:
            if self.state == 0: # chose a random position in one of the childs_nodes if there are
                if len(node.child_nodes) > 0:
                    self.position = np.random.choice(node.child_nodes).id
            else: # go to parent_node
                if node.parent_node is not None:
                    self.position = node.parent_node.id

    def update_neighbours_position(self,agents):#cerca la posizione dei vicini salvata nel mio mondo, elimina i relativi committed_agents. POI aggiungi i committed_agents in base alle posizionni attuali
        for a in agents:
            self.tree.erase_agent(a)
            node = self.tree.catch_node(a.position)
            node.committed_agents = np.append(node.committed_agents,a)

    ##########################################################################
    # udpate of the world model after control routine
    def update(self):
        node = self.tree.catch_node(self.position)
        node.utility = Agent.arena.get_node_utility(node.id)
        self.propagate_info(node)

    def propagate_info(self,node):
        if node.parent_node is not None:
            utility = 0
            for c in node.parent_node.child_nodes:
                utility += c.utility
            node.parent_node.utility = utility/len(node.parent_node.child_nodes)
            self.propagate_info(node.parent_node)
