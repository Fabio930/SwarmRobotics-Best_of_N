# -*- coding: utf-8 -*-
import math, copy
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
        self.pixels_per_meter = 100
        # Initialize the arena and the agents
        self.arena = arena
        self.node_agents = [0]*self.arena.num_nodes
        self.utility_id = [0]*self.arena.num_nodes
        self.nodes_x = np.array([]) # just for placement of nodes on the screen
        self.arena.set_random_seed()
        self.arena.init_experiment()

        # start the GUI
        self.timestep = 0
        self.timestring = tk.StringVar()
        self.timestring.set(str(self.timestep))
        self.initialize()

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
        self.arena.set_random_seed()
        self.arena.init_experiment()
        self.draw_arena()
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
        a_width = self.pixels_per_meter * (self.arena.tree_depth * self.arena.tree_branches)
        a_height = self.pixels_per_meter*(1+self.arena.tree_depth)

        self.w = tk.Canvas(self.master, width=int(a_width), height=int(a_height), background="#EEE")
        self.w.pack()
        self.length = 40
        x1, y1 = 10, 20
        y2 = self.length + y1
        node = self.arena.tree.catch_node(0)
        node.x,node.y1, node.y2 = x1,y1,y2
        x2 = x1+10
        self.w.create_rectangle(x1,y1,x2,y2,fill="white", outline="black")
        self.w.create_text(x1,y1,anchor="sw",text="a")
        self.node_agents[0] = self.w.create_rectangle(x1,y1,x2,y2,fill="blue")
        self.w.create_text(x1,y2+2,anchor="nw",text="id:0")
        # self.nodes_id[0] = self.w.create_oval(x1,y1,x2,y2, fill="white")
        self.w.create_rectangle(x2+6,y1,x2+16,y2,fill="white", outline="black")
        self.w.create_text(x2+6,y1,anchor="sw",text="u")
        self.w.create_text(x2+30,y1,anchor="sw",text="a: % of agents")
        self.w.create_text(x2+30,y1+15,anchor="sw",text="u: % of utility")
        self.w.create_text(x2+30,y1+30,anchor="sw",text="id: node identifier")

        self.utility_id[0] = self.w.create_rectangle(x2+6, y1 + (y2-y1)*(1 - node.utility_mean/self.arena.MAX_utility),x2+16,y2,fill="green")
        self.nodes_x = np.append(self.nodes_x,x1)
        self.paintTree(self.arena.tree_depth,x2,y2)
        self.arena.tree_copy=copy.deepcopy(self.arena.tree)

    # #########################################################################
    # GUI draw function: standard draw of the arena and of the agent
    def draw_arena(self):
        for n in range(len(self.node_agents)):
            node = self.arena.tree.catch_node(n)
            sum = 0
            for a in range(len(node.committed_agents)):
                if node.committed_agents[a] is not None:
                    sum+=1
            if n==0:
                self.w.coords(self.node_agents[n], (node.x, node.y1 + (node.y2-node.y1)*(1 - sum/self.arena.num_agents),node.x+10,node.y2))
            else:
                self.w.coords(self.node_agents[n], (node.x+10, node.y1 + (node.y2-node.y1)*(1 - sum/self.arena.num_agents),node.x+20,node.y2))

    def paintTree(self,depth,x2,y2):
        y1 = y2 + 20
        y2 = y1 + self.length
        for b in range(self.arena.tree_branches):
            if b == 0:
                x1 = np.take(self.nodes_x,-1)
                x2 = x1 + 10
            else:
                x1 = 30 + np.take(self.nodes_x,-1) + 10
                x2 = x1 + 10

            node = self.arena.tree.catch_node(len(self.nodes_x))
            node.x,node.y1, node.y2 = x1,y1,y2
            self.nodes_x = np.append(self.nodes_x,x1)
            # self.nodes_id[node.id]= self.w.create_oval(x1,y1,x2,y2, fill="white")
            self.w.create_rectangle(x2,y1,x2+10,y2,fill="white", outline="black")
            self.node_agents[node.id] = self.w.create_rectangle(x2, y2,x2+10,y2,fill="blue")
            x2 = x2 + 10
            self.w.create_text(x1+10,y2+2,anchor="nw",text="id:"+str(node.id))

            self.w.create_rectangle(x2+6,y1,x2+16,y2,fill="white", outline="black")
            self.utility_id[node.id] = self.w.create_rectangle(x2+6, y1 + (y2-y1)*(1 - node.utility_mean/self.arena.MAX_utility),x2+16,y2,fill="green")
            self.paint_util(depth-1,x2,y2,b)

    def paint_util(self,depth,x2,y2,r):
        if depth > 0:
            y1 = y2 + 20
            y2 = y1 + self.length
            for b in range(self.arena.tree_branches):
                x1 = 30 + np.take(self.nodes_x,-1) + 10
                x2 = x1 + 10
                node = self.arena.tree.catch_node(len(self.nodes_x))
                node.x,node.y1, node.y2 = x1,y1,y2
                self.nodes_x = np.append(self.nodes_x,x1)
                # self.nodes_id[node.id]= self.w.create_oval(x1,y1,x2,y2, fill="white")
                self.w.create_rectangle(x2,y1,x2+10,y2,fill="white", outline="black")
                self.node_agents[node.id] = self.w.create_rectangle(x2, y2,x2+10,y2,fill="blue")
                x2 = x2 + 10
                self.w.create_text(x1+10,y2+2,anchor="nw",text="id:"+str(node.id))

                self.w.create_rectangle(x2+6,y1,x2+16,y2,fill="white", outline="black")
                self.utility_id[node.id] = self.w.create_rectangle(x2+6,  y1 + (y2-y1)*(1 - node.utility_mean/self.arena.MAX_utility),x2+16,y2,fill="green")
                self.paint_util(depth-1,x2,y2,r)
