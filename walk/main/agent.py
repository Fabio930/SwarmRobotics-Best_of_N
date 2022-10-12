# -*- coding: utf-8 -*-
# @author Fabio Oddi <fabioddi24@gmail.com>

import sys,copy,math,random
import numpy as np
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
    size = .033
    alpha = .75
    comm_distance = .1
    r_gain = 1
    expiring_time_qrm = 11
    list_min_dim =10
    quorum_perc = 1

    collision=False
    min_pass=0
    max_pass=6
    dif_pass = max_pass-min_pass
    pass_check=np.arange(1,min_pass + int(dif_pass/2) + 1,1)
    pass_vec=np.arange(min_pass+1,max_pass+1,1)
    angle_vec=np.arange(60,100,1)

    class Factory:
        def create(self, config_element, arena): return Agent(config_element, arena)

    ##########################################################################
    # Initialisation of the Agent class
    def __init__(self, config_element, arena):

        # identification
        self.id = Agent.num_agents
        if self.id == 0:
            Agent.arena = arena
            # parse custon parameters from configuration file
            if config_element.attrib.get("collision") is not None:
                col = int(config_element.attrib["collision"])
                if col==1:
                    Agent.collision=True
            if config_element.attrib.get("agent_size") is not None:
                siz = int(config_element.attrib["agent_size"])
                if siz<=0:
                    print ("[WARNING] for tag <agent> in configuration file the parameter <agent_size> should be > 0. Initialized to 0.033.\n")
                else:
                    Agent.size = siz
            mr=math.sqrt(2 * self.arena.arena_dimension**2)*100
            if config_element.attrib.get("time_reset_msg") is not None:
                mr = int(config_element.attrib["time_reset_msg"])
                if mr<=0:
                    print ("[WARNING] for tag <agent> in configuration file the parameter <time_reset_msg> should be > 0 and integer. Initialized equal to the maximum distance in the arena.\n")
            Agent.expiring_time_msg = mr*(1/self.arena.timestep_length)
            if config_element.attrib.get("time_reset_quorum") is not None:
                tr = int(config_element.attrib["time_reset_quorum"])
                if tr<=0:
                    print ("[WARNING] for tag <agent> in configuration file the parameter <time_reset_quorum> should be > 0 and integer. Initialized to 11.\n")
                else:
                    Agent.expiring_time_qrm = tr
            if config_element.attrib.get("quorum_list_min") is not None:
                bm = float(config_element.attrib["quorum_list_min"])
                if bm<0 or bm>Agent.arena.num_agents:
                    print ("[WARNING] for tag <agent> in configuration file the parameter <quorum_list_min> should be in [0,num_agents]. Initialized to 10.\n")
                else:
                    Agent.list_min_dim = bm
            if config_element.attrib.get("quorum_min") is not None:
                qm = float(config_element.attrib["quorum_min"])
                if qm<0 or qm>1:
                    print ("[WARNING] for tag <agent> in configuration file the parameter <quorum_min> should be in [0,1]. Initialized to 1.\n")
                else:
                    Agent.quorum_perc = qm
            if config_element.attrib.get("comm_distance") is not None:
                dst = float(config_element.attrib["comm_distance"])
                if dst<0:
                    print ("[WARNING] for tag <agent> in configuration file the parameter <comm_distance> should be in [0,inf). Initialized to the double of agents' size. Initialized to 0.1 .\n")
                else:
                    Agent.comm_distance = dst
            Agent.comm_distance += Agent.size*.5 if Agent.collision else 0
            if config_element.attrib.get("alpha") is not None:
                a = float(config_element.attrib["alpha"])
                if a<0 or a>1:
                    print ("[WARNING] for tag <agent> in configuration file the parameter <alpha> should be in (0,1]. Initialized to 0.75 .\n")
                else:
                    Agent.alpha = a
            if config_element.attrib.get("r_gain") is not None:
                tp = float(config_element.attrib["r_gain"])
                if tp<0:
                    print ("[WARNING] for tag <agent> in configuration file the parameter <r_gain> should be greather or equal to 0. Initialized to 1.\n")
                else:
                    Agent.r_gain=tp
        # world representation
        Tree.num_nodes = 0
        self.tree = Tree(Agent.arena.tree_branches,Agent.arena.tree_depth,Agent.arena.num_agents,Agent.arena.MAX_utility,0,1,Agent.alpha,Agent.arena.arena_dimension,'agent')
        for n in range(self.tree.num_nodes+1):
            node = self.tree.catch_node(n)
            node.copy_corners(Agent.arena.tree.catch_node(n))
        self.init_tree = copy.deepcopy(self.tree)
        # store initial positions for reset
        self.init_pos = Vec2d(0,0)
        Agent.num_agents += 1

    ##########################################################################
    # generic init function brings back to initial positions
    def init_experiment( self ):
        random.seed((self.id+1)*np.random.choice(np.arange(1,10,.5)))
        self.tree = copy.deepcopy(self.init_tree)
        self.leafs = self.tree.get_leaf_nodes()
        self.current_node = 0
        self.prev_node = 0
        self.prev_pos = self.init_pos
        self.position = self.init_pos
        self.point = Vec2d(np.random.choice(np.arange(self.tree.tl_corner.__getitem__(0),self.tree.tr_corner.__getitem__(0),.01)),np.random.choice(np.arange(self.tree.tl_corner.__getitem__(1),self.tree.bl_corner.__getitem__(1),.01)))
        self.error_pos = round(self.position.get_distance(self.point),2)
        self.committed = 0
        self.passing = 0
        self.broadcast_buffer = []
        self.messages = {}
        self.angle = 0
        self.on_chosen_point = 0
        self.zero_count = 0
        self.quorum_list = {}
        self.chosen_message = None
        self.action = None
        self.value = None
        self.p = None
        self.r = 0
        self.h = 0
        self.k = 1

    ##########################################################################
    # propagates the info gathered in the current node to the parent nodes
    def update_world_utility(self,sensed_utility,leaf_id,ref):
        leaf = self.tree.catch_node(leaf_id)
        if ref == None:
            leaf.filter.update_utility(sensed_utility)
            if leaf.parent_node is not None:
                tmp=0
                distance=0
                for c in leaf.parent_node.child_nodes:
                    if c.filter.utility is not None:
                        if tmp < c.filter.utility:
                            tmp = c.filter.utility
                    if distance<c.filter.distance:
                        distance = c.filter.distance
                leaf.parent_node.filter.update_utility(tmp,distance)
                self.update_world_utility(sensed_utility,leaf_id,leaf.parent_node)
        else:
            if ref.parent_node is not None:
                tmp=0
                distance=0
                for c in ref.parent_node.child_nodes:
                    if c.filter.utility is not None:
                        if tmp < c.filter.utility:
                            tmp = c.filter.utility
                    if distance<c.filter.distance:
                        distance = c.filter.distance
                ref.parent_node.filter.update_utility(tmp,distance)
                self.update_world_utility(sensed_utility,leaf_id,ref.parent_node)

    ##########################################################################
    def update(self,neighbors):
        self.prev_node = self.current_node
        self.prev_pos = self.position
        self.check_4_old_messages()
        self.read_messages_from_buffer(neighbors)
        self.decision(neighbors)

    ##########################################################################
    def R(self):
        return str(Agent.r_gain)+'(1 - D)'

    ##########################################################################
    def decision(self,neighbors):
        node = self.tree.catch_node(self.current_node)
        info_array = []
        self.action = None
        self.value = None
        self.p = None
        self.chosen_message = None
        if self.error_pos==0 and self.position.isin(node.tl_corner,node.br_corner):
            self.on_chosen_point+=1
            to_eraseQ=[]
            for i in self.quorum_list.keys():
                self.quorum_list.update({i:[self.quorum_list.get(i)[0],self.quorum_list.get(i)[1]+1]})
                if self.quorum_list.get(i)[1]>=Agent.expiring_time_qrm:
                    to_eraseQ.append(i)
            for idq in to_eraseQ:
                self.quorum_list.pop(idq)
            if len(list(self.messages.keys()))>0:
                self.chosen_message = np.random.choice(list(self.messages.keys()))
                info_array = self.messages.get(self.chosen_message)
                self.messages = {}
                if self.committed==node.id:
                    if node.get_sibling_node(info_array[0]) is None:
                        self.quorum_list.update({self.chosen_message:[info_array[0],0]})
                elif self.committed==node.parent_node.id:
                    self.quorum_list.update({self.chosen_message:[info_array[0],0]})
            if len(self.quorum_list.keys())>=Agent.list_min_dim:
                quorum = 0
                for i in self.quorum_list.keys():
                    if self.quorum_list.get(i)[0]==node.id or node.get_sub_node(self.quorum_list.get(i)[0]) is not None:
                        quorum += 1
                if quorum >= len(self.quorum_list.keys())*Agent.quorum_perc:
                    if self.committed != node.id:
                        self.committed = node.id
            elif len(self.quorum_list.keys()) < Agent.list_min_dim*.8 :
                if node.parent_node is not None:
                    if self.committed != node.parent_node.id:
                        self.committed = node.parent_node.id
            commitment = 0
            recruitment = 0
            abandonment = 0
            cross_inhibition = 0
            selected_node = None
            if node.child_nodes[0] is not None:
                for c in node.child_nodes:
                    if self.point.isin(c.tl_corner,c.br_corner):
                        selected_node = c
                        break
                if self.position.isin(selected_node.tl_corner,selected_node.br_corner):
                    leaf_id, leaf_utility = Agent.arena.get_node_utility(selected_node.id,self.position)
                    self.update_world_utility(leaf_utility,leaf_id,None)
                    if node.id == self.committed:
                        if selected_node.filter.utility < 0:
                            commitment = 0
                        elif selected_node.filter.utility < Agent.arena.MAX_utility:
                            commitment = selected_node.filter.utility/Agent.arena.MAX_utility
                        else:
                            commitment = 1
            else:
                leaf_id, leaf_utility = Agent.arena.get_node_utility(node.id,self.position)
                self.update_world_utility(leaf_utility,leaf_id,None)
            if node.parent_node is not None:
                if self.committed == node.parent_node.id:
                    abandonment = 1 if node.filter.utility <= 0 else 1/(1 + node.filter.utility)
            agent_node = None
            agent_nodeC = None
            if self.chosen_message is not None:
                agent_node = node.get_sub_node(info_array[0])
                agent_nodeC = node.get_sibling_node(info_array[0])
                if agent_node is not None:
                    if node.id == self.committed:
                        for i in info_array[1].keys():
                            self.update_world_utility(info_array[1].get(i),i,None)
                        utility = self.tree.catch_node(agent_node.id).filter.utility
                        if utility <= 0:
                            recruitment = 0
                        elif utility < Agent.arena.MAX_utility:
                            recruitment =  utility / Agent.arena.MAX_utility
                        else:
                            recruitment = 1
                elif agent_nodeC is not None:
                    if node.parent_node.id == self.committed:
                        for i in info_array[1].keys():
                            self.update_world_utility(info_array[1].get(i),i,None)
                        utility = self.tree.catch_node(agent_nodeC.id).filter.utility
                        if utility <= 0:
                            cross_inhibition = 0
                        elif utility< Agent.arena.MAX_utility:
                            cross_inhibition = utility / Agent.arena.MAX_utility
                        else:
                            cross_inhibition = 1
            self.r = Agent.r_gain*(1 - self.tree.catch_node(self.committed).filter.distance)
            r = self.r
            if agent_node is not None or agent_nodeC is not None:
                r = info_array[2]
            self.h = r/(1+r)
            self.k = 1/(1+r)
            commitment = self.k * commitment
            abandonment = self.k * abandonment
            recruitment = self.h * recruitment
            cross_inhibition = self.h * cross_inhibition
            p = np.random.uniform(0,1)
            self.p = p
            if p < commitment:
                self.current_node = selected_node.id
                self.action='commitment'
                self.value=commitment
            elif p < (commitment + recruitment):
                self.current_node = agent_node.id
                self.action='recruitment'
                self.value=[recruitment,agent_node.id]
            elif p < (commitment + cross_inhibition):
                self.current_node = node.parent_node.id
                self.action='cross_inhibition'
                self.value=[cross_inhibition,agent_nodeC.id]
            elif p < (commitment + recruitment + cross_inhibition + abandonment)*.667:
                self.current_node = node.parent_node.id
                self.action = 'abandonment'
                self.value = abandonment
            self.broadcast(neighbors)

    #########################################################################
    def broadcast(self,neighbors):
        if len(neighbors)>0:
            info={}
            for l in self.leafs:
                if l.filter.utility is not None:
                    info.update({l.id:l.filter.utility})
            if len(info.keys())>0:
                for n in neighbors:
                    try:
                        n.broadcast_buffer.append([self.id,self.current_node,info,self.r,0,0])
                    except Exception:
                        print('WARNING message not deliverd from agentID:',self.id,'to neighborID:',n.id)

    #########################################################################
    def update_and_re_broadcast(self,neighbors,indx):
        self.messages.update({indx[0]:[indx[1],indx[2],indx[3],indx[4],indx[5]]})
        for n in neighbors:
            try:
                n.broadcast_buffer.append([indx[0],indx[1],indx[2],indx[3],indx[4],indx[5]+1])
            except Exception:
                print('WARNING message not deliverd duringe re-broadcast from agentID:',self.id,'to neighborID:',n.id)

    #########################################################################
    def read_messages_from_buffer(self,neighbors):
        for i in self.broadcast_buffer:
            if i[0]!=self.id:
                if i[0] not in self.messages.keys() or i[5]<self.messages.get(i[0])[4]:
                    self.update_and_re_broadcast(neighbors,i)
                elif i[5]==self.messages.get(i[0])[4]:
                    if i[4]<self.messages.get(i[0])[3]:
                        self.update_and_re_broadcast(neighbors,i)
        self.broadcast_buffer = []

    #########################################################################
    def check_4_old_messages(self):
        to_eraseM=[]
        for i in self.messages.keys():
            self.messages.update({i:[self.messages.get(i)[0],self.messages.get(i)[1],self.messages.get(i)[2],self.messages.get(i)[3]+1,self.messages.get(i)[4]]})
            if self.messages.get(i)[3]>=Agent.expiring_time_msg:
                to_eraseM.append(i)
        for idm in to_eraseM:
            self.messages.pop(idm)

    #########################################################################
    def move(self,positions):
        node = self.tree.catch_node(self.current_node)
        self.step = .01*self.arena.timestep_length
        if self.error_pos==0:
            self.point = Vec2d(np.random.choice(np.arange(node.tl_corner.__getitem__(0),node.tr_corner.__getitem__(0),.01)),np.random.choice(np.arange(node.tl_corner.__getitem__(1),node.bl_corner.__getitem__(1),.01)))
        angle = math.pi + math.atan2(self.prev_pos.__getitem__(1)-self.point.__getitem__(1),self.prev_pos.__getitem__(0)-self.point.__getitem__(0))
        self.position = self.calc_new_pos(angle,positions,node)

    #########################################################################
    def calc_error_pos(self):
        self.error_pos = round(self.position.get_distance(self.point),2)

    #########################################################################
    def calc_init_pos(self,agents):
        self.init_pos=Vec2d(np.random.uniform(self.tree.tl_corner.__getitem__(0),self.tree.tr_corner.__getitem__(0)),np.random.uniform(self.tree.tl_corner.__getitem__(1),self.tree.br_corner.__getitem__(1)))
        flag = True
        while flag:
            flag=False
            for a in agents:
                if a.id!=self.id:
                    if self.init_pos.get_distance(a.init_pos)<=self.size:
                        self.init_pos=Vec2d(np.random.uniform(self.tree.tl_corner.__getitem__(0),self.tree.tr_corner.__getitem__(0)),np.random.uniform(self.tree.tl_corner.__getitem__(1),self.tree.br_corner.__getitem__(1)))
                        flag = True
                        break

    #########################################################################
    def calc_new_pos(self,angle,positions,node):
        dx = self.step*math.cos(angle)
        dy = self.step*math.sin(angle)
        new_pos = Vec2d(self.prev_pos.__getitem__(0)+dx,self.prev_pos.__getitem__(1)+dy)
        for p in positions:
            if round(new_pos.get_distance(p),2)<=self.size:
                s=0
                if self.passing<=0:
                    angle+=(math.pi/180)*np.random.choice(self.angle_vec)
                    self.passing=np.random.choice(self.pass_vec)*(1/self.arena.timestep_length)
                    self.angle=angle
                else:
                    angle=self.angle*(1+np.random.uniform(.03,.33))
                    self.passing-=1
                dx = self.step*math.cos(angle)
                dy = self.step*math.sin(angle)
                new_pos = Vec2d(self.prev_pos.__getitem__(0)+dx,self.prev_pos.__getitem__(1)+dy)
                if not new_pos.isin(self.tree.tl_corner,self.tree.br_corner):
                    new_pos = self.prev_pos
                    s=1
                else:
                    for q in positions:
                        if new_pos.get_distance(q)<=self.size:
                            new_pos = self.prev_pos
                            break
                if s==0 and new_pos == self.prev_pos and self.passing > 0 and self.passing <= np.random.choice(self.pass_check):
                    self.point = Vec2d(np.random.choice(np.arange(node.tl_corner.__getitem__(0),node.tr_corner.__getitem__(0),.01)),np.random.choice(np.arange(node.tl_corner.__getitem__(1),node.bl_corner.__getitem__(1),.01)))
                    self.passing = 0
        return new_pos
