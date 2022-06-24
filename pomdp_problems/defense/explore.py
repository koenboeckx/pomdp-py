""" 
Explore(n, s) problem

Description: 
    In a n-by-n gridworld, k blocks (obstacles) are placed. An agent enters and
    leaves the world throuhg the same entrance/exit. He can move freely,
    but is blocked by the obstacles. After each step, the agent receives an observation
    consisting of all squares with a line-of-sight distance s, except those squares of which
    the line-of-sight is blocked by an obstacle.
    Agent receives:
        * reward = -1 for each step
        * reward = 10 for each square observed
        * reward = 100 when all squares observed -> terminal state
        * (optionally) reward = 100 to leave through entrance -> terminal state
    Action space:
        * Move North, South, East, West

#TODO:
* create (pre-compute) visibility map (once obstacle positions are known)
    = dict{(x1, y1), (x2, y2) -> False/True} (for infinite max line-of-sight)
* 
"""

import pomdp_py
import random
import math
import numpy as np
import sys
import copy

EPSILON = 1e-9

def euclidean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class State(pomdp_py.State):
    def __init__(self, position, seen, terminal=False):
        """
        Args:
            position (tuple): (x,y) position of the agent on the grid.
            seen (list): list of squares already seen (0..n**2-1)
            terminal (bool, optional): Agent in terminal state. Defaults to False.
        """
        self.position = position
        self.seen = seen
        self.terminal = terminal

    def __hash__(self):
        return hash((self.position, self.seen, self.terminal))
    
    def __eq__(self, other):
        if isinstance(other, State):
            return self.position == self.position \
                and set(self.seen) == set(other.seen) \
                and self.terminal == self.terminal
        else:
            return False
    
    def __repr__(self) -> str:
        return f"State({str(self.position)} | {len(self.seen)} | {str(self.terminal)})"

class Action(pomdp_py.Action):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other
    def __str__(self):
        return self.name
    def __repr__(self):
        return "Action(%s)" % self.name

class MoveAction(Action):
    EAST = (1, 0)  # x is horizontal; x+ is right. y is vertical; y+ is up.
    WEST = (-1, 0)
    NORTH = (0, -1)
    SOUTH = (0, 1)
    def __init__(self, motion, name):
        if motion not in {MoveAction.EAST, MoveAction.WEST,
                          MoveAction.NORTH, MoveAction.SOUTH}:
            raise ValueError("Invalid move motion %s" % motion)
        self.motion = motion
        super().__init__("move-%s" % str(name))

MoveEast = MoveAction(MoveAction.EAST, "EAST")
MoveWest = MoveAction(MoveAction.WEST, "WEST")
MoveNorth = MoveAction(MoveAction.NORTH, "NORTH")
MoveSouth = MoveAction(MoveAction.SOUTH, "SOUTH")

class Observation(pomdp_py.Observation):
    def __init__(self, view):
        """
        Args:
            view (tuple): observations in field of view
            e=empty, d=obstacle, x=not visible
        """
        self.view = view
    def __hash__(self):
        return hash(self.view)
    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.view == other.view
        elif type(other) == str:
            return self.view == other
    def __str__(self):
        return str(self.view)
    def __repr__(self):
        return "Observation(%s)" % str(self.view)
    
    @staticmethod
    def from_state(self, state, obstacles, n=10, los=3):
        """Compute observation from state, taking position
        of obstacles into account

        Args:
            state (State): current agent state
            obstacles (list): list of obstacle positions
            n (int): gridworld size
            los (int): maximum line-of-sight

        Returns:
            Observation: current observation from state
        """
        view = []
        for i in range(n): # Q: where to best put n (here, obstacle, state)?
            for j in range(n):
                if euclidean_dist(state.position, (i, j)) <= los:
                    if 0 <= i < n and 0 <= j < n:
                        view.append(obstacles[(i,j)])
        return Observation(tuple(view))

class RSTransitionModel(pomdp_py.TransitionModel):

    """ The model is deterministic """

    def __init__(self, n, obstacles, los):
        """
        Args:
            n (int): size of gridworld (n x n)
            obstacles (list): list of obstacle positions
            los (int): maximum line-of-sight
        """
        self._n = n
        self.obstacles = obstacles
    
    def _move(self, position, action):
        """Execute move

        Args:
            position (tuple): current position (x, y) of agent
            action (Action): (move) action taken by agent

        Returns:
            tuple: new position (x', y')
        """
        expected = (position[0] + action.motion[0],
                    position[1] + action.motion[1])
        if self.obstacles[expected]: # bounce against obstacle -> no move
            return position
        else:
            return (max(0, min(position[0] + action.motion[0], self._n-1)),
                    max(0, min(position[1] + action.motion[1], self._n-1)))
    
    def sample(self, state, action):
        """ 
        terminal = False
        if len(set(state.seen)) == self._n**2:
            # Question: where update `seen`, since affected by observation
            #   maybe: first compute new post
            terminal = True
        """
        if state.terminal:
            next_terminal = True  # already terminated. So no state transition happens
        else:
            if isinstance(action, MoveAction):
                next_position = self._move_or_exit(state.position, action)