##########################################################################
class Filter:

    def __init__(self,alpha):
        self.utility = None
        self.alpha = float(alpha)

    def update_utility(self,sensed_utility):
        if self.utility == None:
            self.utility = sensed_utility
        else:
            if sensed_utility is not None:
                self.utility = self.alpha*self.utility + (1-self.alpha)*sensed_utility

    def set_alpha(self,new_alpha):
        self.alpha = new_alpha
