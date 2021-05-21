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

    class Factory:
        def create(self, config_element, arena): return Target(config_element, arena)


    ##########################################################################
    # Initialisation of the Target class
    def __init__(self, config_element, arena):

        self.assign = 0
        if Target.num_targets == 0:
            # reference to the arena
            Target.root = arena
        self.quality = 1
        if config_element.attrib.get("quality") is not None:
            self.quality = float(config_element.attrib["quality"])
            if  self.quality<0 or self.quality>1:
                print ("[ERROR] attribute 'quality' in tag <target> must be in [0,1]")
                sys.exit(2)

        # rnd1 = np.random.uniform(0.8,1)
        # self.quality = self.quality * rnd1
        # if self.quality < 0:
        #     self.quality = 0.1
        # if self.quality > 1:
        #     self.quality = 1

        # identification
        self.id = Target.num_targets

        Target.num_targets += 1
        print("Target #"+str(self.id)+" initialized. Quality:"+str(self.quality))
