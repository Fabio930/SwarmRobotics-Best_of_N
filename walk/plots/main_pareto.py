# -*- coding: utf-8 -*-

import os, random, sys, getopt, importlib
from plots.pareto_plot import Plots_pareto
########################################################################################
## main functions
########################################################################################

def print_usage(errcode = None):
    print ('Usage: print_plots [-g]')
    sys.exit(errcode)

def start(argv):

    plotter = Plots_pareto()
    plotter.print()
if __name__ == "__main__":
    start(sys.argv[1:])

#sys.argv is the list of commandline arguments passed to the Python program
