##########################################################################
class Filter:

    def __init__(self,alpha):
        self.utility = 0
        self.alpha = float(alpha)

    def update_utility(self,sensed_utility):
        self.utility = self.alpha*self.utility + (1-self.alpha)*sensed_utility

    def set_alpha(self,new_alpha):
        self.alpha = new_alpha

    def overwrite_utility(self,new_utility):
        self.utility = new_utility
