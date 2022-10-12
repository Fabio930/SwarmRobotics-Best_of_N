# @author Fabio Oddi <fabioddi24@gmail.com>

import numpy as np
import math
##########################################################################
class Filter:

    def __init__(self,alpha,ref=0):
        self.leaf = ref #ref==1 it is a leaf, ref==0 it is not
        self.utility = None
        self.alpha = float(alpha)
        self.data_1 = [0,0,0]
        self.data_2 = [0,0,0] #[CURRENTmean,variance,counter]
        self.switch = 0
        self.distance = 1

    def update_utility(self,sensed_utility,ref=None):
        if self.utility is not None:
            self.utility = self.utility*self.alpha + (1-self.alpha)*sensed_utility
        else:
            self.utility = sensed_utility
        if self.leaf == 1:
            if self.switch == 0:
                self.data_1[2] += 1
                self.data_1[0] = self.data_1[0] + (sensed_utility-self.data_1[0])/self.data_1[2]
                if self.data_1[2]>1:
                    self.data_1[1] = self.data_1[1]*((self.data_1[2]-2)/(self.data_1[2]-1)) + ((sensed_utility-self.data_1[0])**2)/self.data_1[2]
                self.switch = 1
            else:
                self.data_2[2] += 1
                self.data_2[0] = self.data_2[0] + (sensed_utility-self.data_2[0])/self.data_2[2]
                if self.data_2[2]>1:
                    self.data_2[1] = self.data_2[1]*((self.data_2[2]-2)/(self.data_2[2]-1)) + ((sensed_utility-self.data_2[0])**2)/self.data_2[2]
                self.switch = 0
            if self.data_1[2]>1 and self.data_2[2]>1:
                self.distance = 1 - np.sqrt(2*math.sqrt(self.data_1[1])*math.sqrt(self.data_2[1])/(.0000000000000001 + self.data_1[1] + self.data_2[1])) * np.exp(-.25*(((self.data_1[0]-self.data_2[0])**2)/(.0000000000000001 + self.data_1[1] + self.data_2[1])))
        if ref is not None:
            self.distance = ref
