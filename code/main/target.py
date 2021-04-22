# -*- coding: utf-8 -*-
import sys,random
import numpy as np
########################################################################################
## Pysage Target
########################################################################################

##########################################################################
# factory to dynamically create targets
class TargetFactory:
    factories = {}
    def add_factory(id, target_factory):
        TargetFactory.factories[id] = target_factory
    add_factory = staticmethod(add_factory)

    def create_target(config_element, arena):
        target_pkg = config_element.attrib.get("pkg")
        if target_pkg is None:
            return Target.Factory().create(config_element, arena)
        id = target_pkg + ".target"
        target_type = config_element.attrib.get("type")
        if target_type is not None:
            id = target_pkg + "." + target_type + ".target"
        return TargetFactory.factories[id].create(config_element, arena)
    create_target = staticmethod(create_target)


##########################################################################
# the main target class
class Target:
    num_targets = 0
    root = None
    quality = 0.1

    class Factory:
        def create(self, config_element, arena): return Target(config_element, arena)


    ##########################################################################
    # Initialisation of the Target class
    def __init__(self, config_element, arena):

        if Target.num_targets == 0:
            # reference to the arena
            Target.root = arena
            if config_element.attrib.get("quality") is not None:
                Target.quality = float(config_element.attrib["quality"])
                if Target.quality >1 or Target.quality<0:
                    print ("[ERROR] attribute 'quality' in tag <target> must be in [0,1]")
                    sys.exit(2)

        self.assigned = [False,[0,0]]

        rnd1 = np.random.normal(0.5, 0.25)
        rnd2 = np.random.normal(0.75,0.25)
        rnd3 = np.random.normal(1,0.25)
        self.quality = Target.quality * np.random.choice([rnd1, rnd2, rnd3])
        if self.quality < 0:
            self.quality = 0.1

        # identification
        self.id = Target.num_targets

        self.committed_agents = np.array([])

        Target.num_targets += 1
        print("Target #"+str(self.id)+" initialized. Quality:"+str(self.quality))

    def committ_agent(self, agent):
        # returns 0 for succesfull committement or -1 if agent is already committed
        for a in self.committed_agents:
            if a.id == agent.id:
                return -1
        self.committed_agents = np.append(self.committed_agents,agent)
        return 0

    def uncommitt_agent(self,agent):
        # returns the index of the uncommitted agent or -1
        if len(self.committed_agents) > 0:
            INDEX = 0
            sem = False
            for a in self.committed_agents:
                if a.id == agent.id:
                    sem = True
                    break
                INDEX += 1
            if sem == True:
                self.committed_agents = np.delete(self.committed_agents,INDEX)
                return INDEX
            return -1
        return -1
