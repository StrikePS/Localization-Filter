# Pranjal Sinha 

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

# world is 1d
# world has 'n' cells
# initial state of robot is at cell 'j'
# World is bounded, robot cannot move outside the world
# World has towers/landmarks that observe the robot
# each time step, robot can move forward or backward (left or right)
# each time step, towers can observe the robot (left or right)
# Action of robot is subject to error
# landmark positions are given as an array with 1 being a landmark.
# Observation of robot is subject to error

# Action Commands:

# for forward command
# p(not move| forward command) = 0.25
# p(move forward| forward command) = 0.5
# p(2 moves forward| forward command) = 0.25
# p(move backward| forward command) = 0.0

# since world is bounded, for boundary cells:
# at last cell, p(not move|forward command) = 1.0
# or: p(not move | forward command, last cell) = 1.0
# at second last cell, p(not move|forward command) = 0.25, p(move forward|forward command) = 0.75

# for backward command
# similar model as forward command


# Observation:

# The towers scans the environment. 
# The robot positions are relayed back to the robot by the towers with noise.
# Robot updates its current positon beliefs based on observation data.
# the towers relay positions as distance/range from the robot. 
# (If robot and landmark are adjacent, they are sperated by dist=1. if they are on the same cell, dist=0, et cetera)
# The probabilites of observing the robot at position x given the observation is given by a normal distribution.
# Normal distribution works since noise is considered Gaussian. 

# Try to attempt a belief update model for non-Gaussian noise!

# initial belief state   
# bel = np.hstack((np.zeros(9), 1, np.zeros(10))) for a 20 cell world and robot at cell 10.

class DiscreteFilter1D:

    # world is 1d
    # world has more than 3 cells.
    # world is bounded, robot cannot move outside the world
    
    # non boundary zone
    P_Move_F = P_Move_B = 0.5
    P_NotMove_F = P_NotMove_B = 0.25
    P_Move_F2 = P_Move_B2 = 0.25
    
    # boundary zone
    P_NotMove_F_b = 1.0 # last cell
    P_NotMove_B_b = 1.0 # first cell
    
    # semi boundary zone
    P_NotMove_F_sb = 0.25 # second last cell
    P_NotMove_B_sb = 0.25 # second cell
    P_Move_F_sb = 0.75 # second last cell
    P_Move_B_sb = 0.75 # second cell
    
    # default noise params for the observation
    Alpha = 0.005
    
    
    def __init__(self, x, bel, towers):
        """ Initializes the DiscreteFilter1D class with the given parameters
            x: 1d array of the robot position
            bel: 1d array of the robot belief state"""
        self.x = x
        self.bel = bel
        self.towers = towers
    
    def action_prob(index,x,u):
        """ Returns the probability of being at the current cell from all positions given an action
            Returns a 1d array of probabilities"""
        
        probs = np.zeros(len(x))
        
        if u == 0:
            probs[index] == 1
        
        elif u==1:
            #forward
            if index == len(x)-1:
                probs[index] = DiscreteFilter1D.P_NotMove_F_b
                probs[index-1] = DiscreteFilter1D.P_Move_F_sb
                probs[index-2] = DiscreteFilter1D.P_Move_F2
            elif index == len(x)-2:
                probs[index] = DiscreteFilter1D.P_NotMove_F_sb
                probs[index-1] = DiscreteFilter1D.P_Move_F
                probs[index-2] = DiscreteFilter1D.P_Move_F2
            elif index>1 and index<len(x)-2:
                probs[index] = DiscreteFilter1D.P_NotMove_F
                probs[index-1] = DiscreteFilter1D.P_Move_F
                probs[index-2] = DiscreteFilter1D.P_Move_F2
            elif index == 1:
                probs[index] = DiscreteFilter1D.P_NotMove_F
                probs[index-1] = DiscreteFilter1D.P_Move_F
            elif index == 0:
                probs[index] = DiscreteFilter1D.P_NotMove_F
                
        elif u==2:
            #backward
            if index == 0:
                probs[index] = DiscreteFilter1D.P_NotMove_B_b
                probs[index+1] = DiscreteFilter1D.P_Move_B_sb
                probs[index+2] = DiscreteFilter1D.P_Move_B2
            elif index == 1:
                probs[index] = DiscreteFilter1D.P_NotMove_B_sb
                probs[index+1] = DiscreteFilter1D.P_Move_B
                probs[index+2] = DiscreteFilter1D.P_Move_B2
            elif index>1 and index<len(x)-2:
                probs[index] = DiscreteFilter1D.P_NotMove_B
                probs[index+1] = DiscreteFilter1D.P_Move_B
                probs[index+2] = DiscreteFilter1D.P_Move_B2
            elif index == len(x)-2:
                probs[index] = DiscreteFilter1D.P_NotMove_B
                probs[index+1] = DiscreteFilter1D.P_Move_B
            elif index == len(x)-1:
                probs[index] = DiscreteFilter1D.P_NotMove_B
        
        probs = probs/np.sum(probs)
        probs.reshape(-1,1)
        return probs
    
    def gaussian_observation_noise(towers, ranges, alphas):
        arr = ranges.copy()
        if alphas == []:
            for i in range(len(arr)):
                arr[i] = np.random.normal(arr[i], arr[i]*DiscreteFilter1D.Alpha)
        else:
            for i in range(len(arr)):
                arr[i] = np.random.normal(arr[i], arr[i]*alphas[i])
        return arr
    
    def likelihood(x, towers, ranges, alphas):
        lh = 1
        noises = DiscreteFilter1D.gaussian_observation_noise(towers, ranges, alphas)
        for i in range(len(towers)):
            lh = lh*norm.pdf(ranges[i], loc=np.abs(towers[i]-x), scale=noises[i])
            
        return lh
            
    
    def move(x, u=0):
        """ Returns the new position of the robot given an action
            Reuturns a 1d array of the new position"""
        # u = 0 for no move
        # u = 1 for forward
        # u = 2 for backward
        arr = x.copy()
        pos = np.where(arr == 1)[0]
        if len(pos) == 0:
            raise ValueError("No position found in the array")
        if len(pos) > 1:
            raise ValueError("Multiple positions found in the array, expected only one")
        
        index = pos[0]
        
        if u == 0:
            return arr
        elif u == 1 and index>=0 and index<len(arr)-1:
            arr[index], arr[index+1] = 0,1
        elif u == 2 and index>0 and index<len(arr):
            arr[index], arr[index-1] = 0,1
        # else:
        #     print("invalid move command, staying in place")
        #     if index == 0 or index == len(arr)-1:
        #         print(("robot is at boundary, cannot move"))
        return arr
    
    def noisy_move(bel, u=0):
        """ Returns the beliefs of the robot being at position k given an action
            Returns a 1d array of beliefs"""
        # u = 0 for no move
        # u = 1 for forward
        # u = 2 for backward
        
        arr = bel.copy()
        
        if u == 0:
            return arr
        elif u == 1 or u == 2:
            for i in range(len(arr)):
                #arr[i] = np.sum(np.matmul(DiscreteFilter1D.prob(j, bel, u), bel) for j in range(len(arr)))
                arr[i] = np.matmul(DiscreteFilter1D.action_prob(i, bel, u), bel)
        
        norm = np.sum(arr)
        arr = arr/norm
        return arr
                
    def observation(x, towers):
        """ Returns the observed position of robot assuming no noise"""
        return x
    
    def noisy_observation(bel, towers, ranges, alphas=[]):
        """ Returns the beliefs of the robot being at position k given observations from towers
            Returns a 1d array of beliefs"""
            
        # towers is a 1d array of tower postions (index's of tower positions)
        # ranges is a 1d array of ranges from each tower (range is from respective tower position)
        # there is no direction of observation relayed.
        
        arr = bel.copy()
        for i in range(len(arr)):
            arr[i] = DiscreteFilter1D.likelihood(i, towers, ranges, alphas)*arr[i]
        
        arr = arr/np.sum(arr)
            
        return arr
        
    def update(self, d, k=0):
        """ Updates the position of the robot given an action
            Returns a 1d array of the actual new position and a probabilistic new position"""
        # k = 0 represesnts action data
        # k = 1 represents observation data
        x_noise_free = np.zeros(len(self.x))
        bel = np.zeros(len(self.bel))
        # action data update
        if k==0:
            x_noise_free = DiscreteFilter1D.move(self.x, d)
            bel = DiscreteFilter1D.noisy_move(self.bel, d)
        
        #observation data update
        # Implementation Here! make new functions for observation data update in the class
        
        if k==1:
            ranges = []
            alphas = []
            for i in range(len(self.towers)):
                ranges.append(np.abs(self.towers[i]-np.where(self.x==1)[0][0]))
                alphas.append(0.0025*i)
            x_noise_free = DiscreteFilter1D.observation(self.x, self.towers)
            bel = DiscreteFilter1D.noisy_observation(self.bel, self.towers, ranges, alphas)
        
        return x_noise_free, bel