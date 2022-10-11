# SwarmRobotics
This is a simple multi-agent simulation environment for mobile agents over a hierarchic environment.
Install it system-wide and use the 'run_simulation' script to launch the simulation.

Usage:
run_simulation -c <config_xml>

Example
run_simulation -c config/config.xml

The basic classes contain an example of the flocking behaviour by a group of agents.
The xml file provides the basic configuration options for the arena and the agents, as well as for the GUI.
Everything can be configured by overloading the base classes.
The xml file can also be used to switch off the GUI, by commenting the respective line.
