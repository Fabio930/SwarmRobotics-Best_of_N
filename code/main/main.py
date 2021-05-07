# -*- coding: utf-8 -*-

import os, random, sys, getopt, importlib
import xml.etree.ElementTree as ET
from main.arena import ArenaFactory
from main.gui import GUIFactory
from main.results import Results
########################################################################################
## main functions
########################################################################################

def print_usage(errcode = None):
    print ('Usage: run_simulation -c <config_file> [-g]')
    sys.exit(errcode)

def start(argv):
    configfile = ''
    try:
        opts, args = getopt.getopt(argv,"hgc:",["config="])
    except getopt.GetoptError:
        print ('[FATAL] Error in parsing command line arguments')
        print_usage(2)
    for opt, arg in opts:
        if opt == '-h':
            print_usage()
        elif opt in ("-c", "--config"):
            configfile = arg

    if len(configfile)==0:
        print ('[FATAL] missing configuration file')
        print_usage(2)

    print ('Reading configuration file "%s"' % configfile)

    sys.path.insert(0, os.getcwd())

    # parse configuration file    run
    tree = ET.parse(configfile)

    # get the node for the arena configuration
    arena_config = tree.getroot().find('arena')
    if arena_config is None:
        print ("[ERROR] required tag <arena> in configuration file is missing")
        sys.exit(2)

    # dynamically load the library
    lib_pkg = arena_config.attrib.get("pkg")
    if lib_pkg is not None:
        importlib.import_module(".arena",lib_pkg)

    arena = ArenaFactory.create_arena(arena_config)
    results = Results(arena)
    gui_config = tree.getroot().find('gui')
    if gui_config is not None: #original file - used for GUI
        global tk
        import tkinter as tk
        print ('Graphical user interface is on')
        root = tk.Tk()
        root.lift()
        root.call('wm', 'attributes', '.', '-topmost', True)
        root.after_idle(root.call, 'wm', 'attributes', '.', '-topmost', False)

        # dynamically load the library
        lib_pkg    = gui_config.attrib.get("pkg")
        if lib_pkg is not None:
            importlib.import_module(lib_pkg + ".gui", lib_pkg)

        GUIFactory.create_gui(root, arena, gui_config )
        root.mainloop()
    else:
        num_runs = 0
        while num_runs < arena.num_runs:
            print ("************* Run number: ", num_runs, ' *************')
            num_runs += 1
            arena.run_id = num_runs
            if arena.random_seed is None:
                arena.set_random_seed()
            else:
                arena.run_id += arena.random_seed
                arena.set_random_seed(arena.run_id)
            arena.init_experiment()
            arena.run_experiment()
            results.update()
        if num_runs >= arena.num_runs:
            results.print_mean_on_file()
if __name__ == "__main__":
    start(sys.argv[1:])

#sys.argv is the list of commandline arguments passed to the Python program
