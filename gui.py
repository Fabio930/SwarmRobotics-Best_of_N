# -*- coding: utf-8 -*-
import math, copy,matplotlib
from colour import Color
from matplotlib import cm
cmap = matplotlib.cm.get_cmap('rainbow')
import tkinter as tk
import numpy as np
from main.agent import Agent
from main.vectors import Vec2d

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
        self.pixels_per_meter = 100 if config_element.attrib.get("pixels_per_meter")  is None else int(config_element.attrib["pixels_per_meter"])

        # Initialize the arena and the agents
        self.arena = arena
        self.sideLength  = arena.arena_dimension
        self.node_agents = [0]*self.arena.num_nodes
        self.utility_id = [0]*self.arena.num_nodes
        self.agents_id = [0]*self.arena.num_agents;
        self.nodes_x = np.array([]) # just for placement of nodes on the screen
        self.leafs=[]
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
        t_width = 400
        t_height = 400

        self.w = tk.Canvas(self.master, width=int(t_width), height=int(t_height), background="#EEE")
        self.w.pack(side='left',fill='both',expand='True')
        self.length = 40
        x1, y1 = 8, 20
        y2 = self.length + y1
        node = self.arena.tree.catch_node(0)
        node.x,node.y1, node.y2 = x1,y1,y2
        x2 = x1+6
        self.w.create_rectangle(x1,y1,x2,y2,fill="white", outline="black")
        self.w.create_text(x1,y1,anchor="sw",text="a")
        self.node_agents[0] = self.w.create_rectangle(x1,y1,x2,y2,fill="blue")
        self.w.create_text(x1,y2+2,anchor="nw",text="id:0")
        # self.nodes_id[0] = self.w.create_oval(x1,y1,x2,y2, fill="white")
        self.w.create_rectangle(x2+6,y1,x2+12,y2,fill="white", outline="black")
        self.w.create_text(x2+6,y1,anchor="sw",text="u")
        self.w.create_text(x2+30,y1,anchor="sw",text="a: % of agents")
        self.w.create_text(x2+30,y1+15,anchor="sw",text="u: % of utility")
        self.w.create_text(x2+30,y1+30,anchor="sw",text="id: node identifier")

        self.v = tk.Canvas(self.master, width=int((self.sideLength+.1)*self.pixels_per_meter), height=int((self.sideLength+.1)*self.pixels_per_meter), background="#EEE")
        self.v.pack(side='right')
        self.v.create_rectangle(node.tl_corner.__getitem__(0)*self.pixels_per_meter,node.tl_corner.__getitem__(1)*self.pixels_per_meter,node.br_corner.__getitem__(0)*self.pixels_per_meter,node.br_corner.__getitem__(1)*self.pixels_per_meter,fill='white', outline="black")

        self.utility_id[0] = self.w.create_rectangle(x2+6, y1 + (y2-y1)*(1 - node.utility_mean/self.arena.MAX_utility),x2+12,y2,fill="green")
        self.nodes_x = np.append(self.nodes_x,x1)
        self.paintTree(self.arena.tree_depth,x2,y2)
        self.paintArena()
        self.arena.tree_copy=copy.deepcopy(self.arena.tree)

        for a in self.arena.agents:
            xpos = float(a.position.x*self.pixels_per_meter)
            ypos = float(a.position.y*self.pixels_per_meter)
            agent_halfsize = float(Agent.size*self.pixels_per_meter*.5)
            self.agents_id[a.id] = self.v.create_oval((xpos-agent_halfsize,ypos-agent_halfsize,xpos+agent_halfsize,ypos+agent_halfsize), fill="black")

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
                self.w.coords(self.node_agents[n], (node.x, node.y1 + (node.y2-node.y1)*(1 - sum/self.arena.num_agents),node.x+6,node.y2))
            else:
                self.w.coords(self.node_agents[n], (node.x+6, node.y1 + (node.y2-node.y1)*(1 - sum/self.arena.num_agents),node.x+12,node.y2))
        for i in range(self.arena.num_agents):
            a = self.arena.agents[i]
            xpos = float(a.position.x*self.pixels_per_meter)
            ypos = float(a.position.y*self.pixels_per_meter)
            agent_halfsize = float(a.size*self.pixels_per_meter*.5)
            self.v.coords(self.agents_id[i], (xpos-agent_halfsize,ypos-agent_halfsize,xpos+agent_halfsize,ypos+agent_halfsize))

    ##########################################################################
    def paintTree(self,depth,x2,y2):
        y1 = y2 + 20
        y2 = y1 + self.length
        for b in range(self.arena.tree_branches):
            if b == 0:
                x1 = np.take(self.nodes_x,-1)
                x2 = x1 + 6
            else:
                x1 = 15 + np.take(self.nodes_x,-1) + 6
                x2 = x1 + 6

            node = self.arena.tree.catch_node(len(self.nodes_x))
            node.x,node.y1, node.y2 = x1,y1,y2
            self.nodes_x = np.append(self.nodes_x,x1)
            self.w.create_rectangle(x2,y1,x2+6,y2,fill="white", outline="black")
            self.node_agents[node.id] = self.w.create_rectangle(x2, y2,x2+6,y2,fill="blue")
            x2 = x2 + 6
            self.w.create_text(x1+6,y2+2,anchor="nw",text=str(node.id))

            self.w.create_rectangle(x2+6,y1,x2+12,y2,fill="white", outline="black")
            self.utility_id[node.id] = self.w.create_rectangle(x2+6, y1 + (y2-y1)*(1 - node.utility_mean/self.arena.MAX_utility),x2+12,y2,fill="green")

            if node.child_nodes[0] is None:
                self.leafs.append([node,self.v.create_rectangle(node.tl_corner.__getitem__(0)*self.pixels_per_meter,node.tl_corner.__getitem__(1)*self.pixels_per_meter,node.br_corner.__getitem__(0)*self.pixels_per_meter,node.br_corner.__getitem__(1)*self.pixels_per_meter,fill='white', outline="black")])

            self.paint_util(depth-1,x2,y2)

    ##########################################################################
    def paint_util(self,depth,x2,y2):
        if depth > 0:
            y1 = y2 + 20
            y2 = y1 + self.length
            for b in range(self.arena.tree_branches):
                x1 = 20 + np.take(self.nodes_x,-1) + 6
                x2 = x1 + 6
                node = self.arena.tree.catch_node(len(self.nodes_x))
                node.x,node.y1, node.y2 = x1,y1,y2
                self.nodes_x = np.append(self.nodes_x,x1)
                self.w.create_rectangle(x2,y1,x2+6,y2,fill="white", outline="black")
                self.node_agents[node.id] = self.w.create_rectangle(x2, y2,x2+6,y2,fill="blue")
                x2 = x2 + 6
                self.w.create_text(x1+6,y2+2,anchor="nw",text=str(node.id))

                self.w.create_rectangle(x2+6,y1,x2+12,y2,fill="white", outline="black")
                self.utility_id[node.id] = self.w.create_rectangle(x2+6,  y1 + (y2-y1)*(1 - node.utility_mean/self.arena.MAX_utility),x2+12,y2,fill="green")

                if node.child_nodes[0] is None:
                    self.leafs.append([node,self.v.create_rectangle(node.tl_corner.__getitem__(0)*self.pixels_per_meter,node.tl_corner.__getitem__(1)*self.pixels_per_meter,node.br_corner.__getitem__(0)*self.pixels_per_meter,node.br_corner.__getitem__(1)*self.pixels_per_meter,fill='white', outline="black")])

                self.paint_util(depth-1,x2,y2)

    ##########################################################################
    def paintArena(self):
        delta=0.01
        for leaf in self.leafs:
            l=leaf[0].tl_corner.__getitem__(0)
            t=leaf[0].tl_corner.__getitem__(1)
            x0,x1,y0,y1=l,l+delta,t,t+delta
            self.grid=[[None]*len(leaf[0].gaussian_kernel[0])]*len(leaf[0].gaussian_kernel)
            for y in range(len(leaf[0].gaussian_kernel)):
                for x in range(len(leaf[0].gaussian_kernel[y])):
                    color=Color(rgb=cmap(leaf[0].gaussian_kernel[y][x]/self.arena.MAX_utility)[:-1])
                    self.grid[y][x]=self.v.create_rectangle(x0*self.pixels_per_meter,y0*self.pixels_per_meter,x1*self.pixels_per_meter,y1*self.pixels_per_meter,fill=color, outline="black")
                    x0=x1
                    x1=x0+delta
                y0=y1
                y1=y0+delta
                x0=l
                x1=l+delta
