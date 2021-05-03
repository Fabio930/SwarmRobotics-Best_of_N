# -*- coding: utf-8 -*-
import math
import tkinter as tk
import numpy as np
from main.agent import Agent
########################################################################################
## Pysage GUI
########################################################################################

##########################################################################
# factory to dynamically create the gui
class GUIFactory:
    factories = {}
    def add_factory(id, gui_factory):
        GUIFactory.factories[id] = gui_factory
    add_factory = staticmethod(add_factory)

    def create_gui(master, arena, config_element):
        gui_pkg = config_element.attrib.get("pkg")
        if gui_pkg is None:
            return PysageGUI.Factory().create(master, arena, config_element)
        id = gui_pkg + ".gui"
        gui_type = config_element.attrib.get("type")
        if gui_type is not None:
            id = gui_pkg + "." + gui_type + ".gui"
        return GUIFactory.factories[id].create(master, arena, config_element)
    create_gui = staticmethod(create_gui)


##########################################################################
# GUI main class
class PysageGUI(object):

    class Factory:
        def create(self, master, arena, config_element): return PysageGUI(master, arena, config_element)

    ##########################################################################
    # standart class init
    def __init__(self, master, arena, config_element):

        self.master = master

        self.delay  = 1.0 if config_element.attrib.get("delay")  is None else float(config_element.attrib["delay"])
        self.pixels_per_meter = 250 if config_element.attrib.get("pixels_per_meter")  is None else int(config_element.attrib["pixels_per_meter"])

        # Initialize the arena and the agents
        self.arena = arena
        self.tree = arena.get_tree_copy()
        self.agents_id = [0]*self.arena.num_agents
        self.nodes_id =  [0]*self.arena.num_nodes
        self.utility_id = [0]*self.arena.num_nodes
        self.nodes_x = np.array([]) # just for placement of nodes on the screen
        self.arena.set_random_seed()
        self.arena.init_experiment()

        # start the GUI
        self.timestep = 0
        self.timestring = tk.StringVar()
        self.timestring.set(str(self.timestep))
        self.initialize()

        # Draw the arena
        self.draw_arena(True)

        # initialise running state
        self.isRunning = False


    ##########################################################################
    # GUI step function: advance the simulation by one time step
    def step(self):
        if not self.arena.experiment_finished():
            self.stop_button.config(state="normal")
            self.step_button.config(state="normal")
            self.run_button.config(state="disabled")
            self.reset_button.config(state="normal")
            self.arena.update()
            self.draw_arena()
            self.timestring.set( str(self.arena.num_steps) )
            self.master.update_idletasks()
        else:
            self.stop_button.config(state="disabled")
            self.step_button.config(state="disabled")
            self.run_button.config(state="disabled")
            self.reset_button.config(state="normal")
            self.draw_arena()
            self.stop()


    ##########################################################################
    # GUI run helper function
    def run_( self ):
        if self.isRunning:
            self.step()
            ms = int(10.0 * max(self.delay, 1.0))

            self.master.after(ms, self.run_)

    ##########################################################################
    # GUI run function: advance the simulation
    def run(self):
        if not self.isRunning:
            self.isRunning = True
            self.run_()

    ##########################################################################
    # GUI stop function: stops the simulation
    def stop(self):
        self.isRunning = False
        self.timestring.set( str(self.timestep) )
        self.stop_button.config(state="disabled")
        self.step_button.config(state="normal")
        self.run_button.config(state="normal")
        self.reset_button.config(state="normal")
        self.timestring.set( str(self.arena.num_steps) )
        self.master.update_idletasks()

    ##########################################################################
    # GUI reset function: reset the simulation
    def reset(self):
        self.isRunning = False
        self.stop_button.config(state="disabled")
        self.step_button.config(state="normal")
        self.run_button.config(state="normal")
        self.reset_button.config(state="disabled")
        self.arena.init_experiment()
        self.draw_arena(True)
        self.timestring.set( str(self.arena.num_steps) )
        self.master.update_idletasks()

    ##########################################################################
    # GUI intialize function: stup the tk environemt
    def initialize(self):
        self.toolbar = tk.Frame(self.master, relief='raised', bd=2)
        self.toolbar.pack(side='top', fill='x')

        self.step_button = tk.Button(self.toolbar, text="Step", command=self.step)
        self.run_button = tk.Button(self.toolbar, text="Run", command=self.run)
        self.stop_button = tk.Button(self.toolbar, text="Stop", command=self.stop)
        self.reset_button = tk.Button(self.toolbar, text="Reset", command=self.reset)
        self.step_button.pack(side='left')
        self.stop_button.pack(side='left')
        self.run_button.pack(side='left')
        self.reset_button.pack(side='left')

        self.stop_button.config(state="disabled")
        self.reset_button.config(state="disabled")

        self.scale = tk.Scale(self.toolbar, orient='h', from_=1, to=10, resolution=0.5, command=lambda d: setattr(self, 'delay', float(d)))
        self.scale.set(self.delay)
        self.scale.pack(side='left')

        self.label = tk.Label(self.toolbar, textvariable = self.timestring)
        self.label.pack(side='right')

        # print ("Canvas size", self.pixels_per_meter*self.arena.dimensions)
        a_width = self.pixels_per_meter
        a_height = self.pixels_per_meter/2


        self.w = tk.Canvas(self.master, width=int(a_width), height=int(a_height), background="#EEE")
        self.w.pack()
        self.length = self.pixels_per_meter/(5*self.arena.tree_depth*self.arena.tree_branches)
        agent_halfsize = int(Agent.size*self.length)
        x1, y1 = agent_halfsize*2, self.pixels_per_meter*0.0001 + agent_halfsize*2
        x2, y2 = agent_halfsize*2 + self.length, self.pixels_per_meter*0.0001 + self.length + agent_halfsize*2
        node = self.tree.catch_node(0)
        node.x,node.y = x1,y1
        self.nodes_id[0] = self.w.create_oval(x1,y1,x2,y2, fill="white")
        self.w.create_rectangle(x2+2,y1,x2+12,y2,fill="white", outline="black")
        self.utility_id[0] = self.w.create_rectangle(x2+2, agent_halfsize*2 + y2*(1 - node.utility/self.arena.max_targets_per_node),x2+12,y2,fill="black")
        self.nodes_x = np.append(self.nodes_x,x1)
        self.paintTree(self.arena.tree_depth,x2,y2)

        for a in self.arena.agents:
            agent_halfsize = int(Agent.size*self.length)
            agent_tag = "agent_%d" % a.id
            node = self.tree.catch_node(a.position)
            self.agents_id[a.id] = self.w.create_oval(node.x+self.length/2-agent_halfsize,node.y+self.length/2-agent_halfsize,node.x+self.length/2+agent_halfsize,node.y+self.length/2+agent_halfsize, fill="blue", tags=(agent_tag))
            self.w.tag_bind(agent_tag, "<ButtonPress-1>", lambda event, agent_tag = agent_tag: self.agent_selected(event, agent_tag))

    # #########################################################################
    # GUI draw function: standard draw of the arena and of the agent
    def draw_arena(self, init=False):
        self.w.bind("<Button-1>", self.unselect_agent)
        for a in self.arena.agents:
            node = self.tree.catch_node(a.position)
            agent_halfsize = int(Agent.size*self.length)
            self.w.coords(self.agents_id[a.id], (node.x+self.length/2-agent_halfsize,node.y+self.length/2-agent_halfsize,node.x+self.length/2+agent_halfsize,node.y+self.length/2+agent_halfsize))

    def paintTree(self,depth,x2,y2):
        agent_halfsize = int(Agent.size*self.length)
        y1 = y2 + 5
        y2 = y1 + self.length
        for b in range(self.arena.tree_branches):
            if b == 0:
                x1 = self.length + np.take(self.nodes_x,-1)
                x2 = x1 + self.length
            else:
                x1 = self.length + np.take(self.nodes_x,-1) + 14
                x2 = x1 + self.length

            node = self.tree.catch_node(len(self.nodes_x))
            node.x,node.y = x1,y1
            self.nodes_x = np.append(self.nodes_x,x1)
            self.nodes_id[node.id]= self.w.create_oval(x1,y1,x2,y2, fill="white")
            self.w.create_rectangle(x2+2,y1,x2+12,y2,fill="white", outline="black")
            self.utility_id[0] = self.w.create_rectangle(x2+2, y1 + (y2-y1)*(1 - node.utility/self.arena.max_targets_per_node),x2+12,y2,fill="black")
            self.paint_util(depth-1,x2,y2,b)
            # uso l'utilità del nodo, barra colorata

    def paint_util(self,depth,x2,y2,r):
        agent_halfsize = int(Agent.size*self.length)
        if depth > 0:
            y1 = y2 + 5
            y2 = y1 + self.length
            for b in range(self.arena.tree_branches):
                x1 = self.length + np.take(self.nodes_x,-1) + 14
                x2 = x1 + self.length
                node = self.tree.catch_node(len(self.nodes_x))
                node.x,node.y = x1,y1
                self.nodes_x = np.append(self.nodes_x,x1)
                self.nodes_id[node.id]= self.w.create_oval(x1,y1,x2,y2, fill="white")
                self.w.create_rectangle(x2+2,y1,x2+12,y2,fill="white", outline="black")
                self.utility_id[0] = self.w.create_rectangle(x2+2,  y1 + (y2-y1)*(1 - node.utility/self.arena.max_targets_per_node),x2+12,y2,fill="black")
                self.paint_util(depth-1,x2,y2,r)

    ##########################################################################
    # de-select an agent that was previously selected by a click
    def unselect_agent( self, event ):
        if not event.widget.find_withtag(tk.CURRENT):
            self.w.itemconfigure('selected',fill="blue")
            self.w.dtag('all','selected')
            for agent in self.arena.agents:
                agent.set_selected_flag(False)
        self.master.update_idletasks()

    ##########################################################################
    # select an agent through a mouse click
    def agent_selected( self, event, agent_tag ):
        self.w.itemconfigure('selected',fill="blue")
        self.w.dtag('all','selected')
        self.w.addtag('selected','withtag',agent_tag)
        self.w.itemconfigure('selected',fill="red")
        self.master.update_idletasks()
        a_str, a_id = agent_tag.split("_")
        self.arena.agents[int(a_id)].set_selected_flag(True)
