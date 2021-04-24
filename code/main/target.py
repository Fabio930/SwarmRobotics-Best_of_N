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

        if Target.num_targets == 0:
            # reference to the arena
            Target.root = arena
        self.quality = 0.1
        if config_element.attrib.get("quality") is not None:
            self.quality = float(config_element.attrib["quality"])
            if  self.quality<0:
                print ("[ERROR] attribute 'quality' in tag <target> must be greater than 0")
                sys.exit(2)

        rnd1 = np.random.normal(0.5, 0.25)
        rnd2 = np.random.normal(0.75,0.25)
        rnd3 = np.random.normal(1,0.25)
        self.quality = self.quality * np.random.choice([rnd1, rnd2, rnd3])
        if self.quality < 0:
            self.quality = 0.1

        # identification
        self.id = Target.num_targets

        Target.num_targets += 1
        print("Target #"+str(self.id)+" initialized. Quality:"+str(self.quality))
