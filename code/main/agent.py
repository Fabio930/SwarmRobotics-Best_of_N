# -*- coding: utf-8 -*-
import sys
import numpy as np
########################################################################################
## Pysage Agent
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
    root = None
    P_up = 0.5
    k = 1
    h = 1

    class Factory:
        def create(self, config_element, arena): return Agent(config_element, arena)


    ##########################################################################
    # Initialisation of the Agent class
    def __init__(self, config_element, arena):

        if Agent.num_agents == 0:
            # reference to the arena
            Agent.root = arena

            if config_element.attrib.get("k") is not None:
                Agent.k = int(config_element.attrib["k"])
            if config_element.attrib.get("h") is not None:
                Agent.h = int(config_element.attrib["h"])
            if config_element.attrib.get("P_up") is not None:
                Agent.P_up = int(config_element.attrib["P_up"])
                if Agent.P_up < 0 or Agent.P_up > 1:
                    print ("[ERROR] for tag <agent> in configuration file the parameter <P_up> should be in [0,1]")
                    sys.exit(2)

        # identification
        self.id = Agent.num_agents

        # current position
        self.node = arena

        # stored utilities and probability transitions
        # the dictionary keys are the nodes ids
        self.nodes_utility = {} # the value is a value for the utility[
        self.nodes_transition = {} #the value is a vector [committment,recruitment,abandonment,self-inhibition,cross-inhibition]
        self.nodes_reference = {}

        # committment to a target
        self.committed = False

        # state: 0=descending, 1=ascending
        self.state = 0

        # choice is a node
        self.choice = None

        Agent.num_agents += 1
        print("Agent #"+str(self.id)+" initialized in node:"+str(self.node.id))

    ##########################################################################
    # generic init function brings back to initial configuration
    def init_experiment( self ):
        self.node = Agent.root
        self.state = 0
        self.choice = None
        self.committed = False
        self.nodes_utility = {}
        self.nodes_transition  = {}
        self.nodes_reference = {}
        self.initialize_nodes()

    ##########################################################################
    # compute a random motion on the tree and the next state of the agent
    def control(self):
        print("Agent #"+str(self.id)+" in node:"+str(self.node.id)+" is controlling")
        agents = self.node.get_neighbour_agents(self.id)

        flag = self.node.get_id()
        self.nodes_transition.update({flag:[self.nodes_transition.get(flag)[0],self.nodes_transition.get(flag)[1],0,0,0]})
        if len(self.node.branches) > 0:
            for b in self.node.branches:
                flag = b.get_id()
                self.nodes_transition.update({flag:[0,0,self.nodes_transition.get(flag)[2],self.nodes_transition.get(flag)[3],self.nodes_transition.get(flag)[4]]})
        random_agent = np.random.choice(agents)

        if self.choice != None:
            self.state = self.pick_a_random_state()

        if self.state == 0:
            if len(self.node.branches) > 0:
                for b in self.node.branches:
                    self.update_node_transition(b,random_agent,self.state)
                self.choice=np.random.choice(self.node.branches)
            else:
                self.choice = None
        else:
            if self.node.parent is not None:
                self.update_node_transition(self.node,random_agent,self.state)
                self.choice=self.node.parent
            else:
                self.choice = None

    ##########################################################################
    # generic update function to be overloaded by subclasses
    def update( self ):
        print("Agent #"+str(self.id)+" in node:"+str(self.node.id)+" is updating")
        agents = self.node.get_neighbour_agents(self.id)
        if self.choice is not None:
            flag = None
            for a in range(len(self.node.agents)):
                if self.node.agents[a].id == self.id:
                    flag = a
            self.node.agents = np.delete(self.node.agents,flag)
            self.choice.agents = np.append(self.choice.agents,self)
            self.node = self.choice
            self.update_node_utility(self.node,agents)
        else:
            if self.state == 0:
                self.state = 1
            else:
                self.state = 0

    ##########################################################################
    # Choose a random state
    def pick_a_random_state(self):
        # if np.random.normal(Agent.P_up ,0.5) > .5:
        #     return 1
        # return 0
        return np.random.choice([0,1])

    ##########################################################################
    # function to initialize the tree structure
    def initialize_nodes(self):
        self.nodes_utility.update({self.node.get_id():0})
        self.nodes_transition.update({self.node.get_id():[0,0,0,0,0]})
        self.nodes_reference.update({self.node.get_id():self.node})
        for branch in self.node.branches:
            self.init_branch(branch)
        print("Agent #"+str(self.id)+" has initialized the structure")

    def init_branch(self,branch):
        self.nodes_reference.update({branch.get_id():branch})
        self.nodes_utility.update({branch.get_id():0})
        self.nodes_transition.update({branch.get_id():[0,0,0,0,0]})
        if len(branch.branches) > 0:
            for b in branch.branches:
                self.init_branch(b)

    ##########################################################################
    # updates agent's utility register
    def update_node_utility(self,node,agents):
        targets = node.get_targets()
        node_utility = 0
        for t in targets:
            node_utility += t.quality*(1-self.normalized_cost_distance(t))/self.normalized_lower_bound_cost(agents,t)
        self.nodes_utility.update({node.get_id():node_utility})
        print(node_utility)

    def normalized_cost_distance(self,t):
        MAX_distance = 2 * Agent.root.depth
        node = self.nodes_reference.get(str(t.assigned[1][0])+""+str(t.assigned[1][1]))
        distance = self.node.get_distance(node)
        # print(distance,MAX_distance)
        return distance/MAX_distance

    def normalized_lower_bound_cost(self, agents,t):
        sum = 0
        for a in agents:
            sum += t.quality*(1-a.normalized_cost_distance(t))
        return sum

    ##########################################################################
    # updates agent's transition register of a chosen node
    def update_node_transition(self,node,agent,s):
        if s == 0:
            # [committment,recruitment,0,0,0]
            recruitment = 0
            committment = Agent.k * self.nodes_utility.get(node.get_id())
            if node.has_in_subtree(agent.node):
                recruitment = Agent.h * agent.nodes_utility.get(node.get_id())
            self.nodes_transition.update({node.get_id():[committment,recruitment,0,0,0]})
        else:
            # [0,0,abandonment,self_inhibition,cross_inhibition]
            self_inhibition ,cross_inhibition = 0, 0
            abandonment = Agent.k * (1 - self.nodes_utility.get(node.get_id()))
            if node.has_in_subtree(agent.node) or node.has_in_siblings(agent.node):
                self_inhibition = Agent.h * agent.nodes_utility.get(node.get_id())*np.heaviside(len(node.agents),0.75*5) # capacita del nodo provvisoria
            if node.has_in_subtree_of_siblings(agent.node):
                cross_inhibition = Agent.h * agent.nodes_utility.get(node.get_id())*np.heaviside(0.25*5,len(node.agents))
            self.nodes_transition.update({node.get_id():[0,0,abandonment,self_inhibition,cross_inhibition]})
