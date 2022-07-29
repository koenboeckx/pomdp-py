""" 
Explore(n, k, s) problem

Description: 
    In a n-by-n gridworld, k blocks (obstacles) are placed. An agent enters and
    leaves the world throuhg the same entrance/exit. He can move freely,
    but is blocked by the obstacles. After each step, the agent receives an observation
    consisting of all squares with a line-of-sight distance s, except those squares of which
    the line-of-sight is blocked by an obstacle.
    Agent receives:
        * reward = -1 for each step
        * reward = 10 for each square observed
        * (optionally) reward = 100 when all squares observed -> terminal state
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
            seen (set): set of squares already seen (0..n**2-1)
            terminal (bool, optional): Agent in terminal state. Defaults to False.
        """
        self.position = position
        self.seen = seen
        self.terminal = terminal

    def __hash__(self):
        return hash((self.position, tuple(self.seen), self.terminal))
    
    def __eq__(self, other):
        if isinstance(other, State):
            return self.position == self.position and self.terminal == self.terminal #\ and set(self.seen) == set(other.seen) \
                
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
    def __init__(self, view, coords):
        """
        Args:
            view (tuple): observations in field of view
            e=empty, d=obstacle, x=not visible
            coords (tuple): coords of elements in view
        """
        self.view = view
        self.coords = coords
    def __hash__(self):
        return hash(self.view)
    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.view == other.view
        elif type(other) == str:
            return self.view == other
    def __str__(self):
        return str(self.view) + ' |\n' + str(self.coords)
    def __repr__(self):
        return "Observation(%s)" % str(self.view)
    
    @staticmethod
    def from_position(position, obstacles, n=10, los=3):
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
        if not hasattr(Observation, 'vis_map'):
            Observation.make_visibility_map(n, obstacles)
        view, coords = [], []
        for i in range(n): 
            for j in range(n):
                if euclidean_dist(position, (i, j)) <= los:
                    if 0 <= i < n and 0 <= j < n:
                        coords.append((i,j))
                        if (i, j) in obstacles:
                            view.append('d')
                        elif not Observation.vis_map[position, (i,j)]: # view at (i, j) is blocked by obstacle(s)
                            view.append('x')
                        else: # observed empty spot
                            view.append('e')
        return Observation(tuple(view), tuple(coords))
    
    @staticmethod
    def make_visibility_map(n, obstacles, N=1000):        
        def is_visible(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            xs = np.linspace(x1, x2, N)
            ys = np.linspace(y1, y2, N)
            for x, y in zip(xs, ys):
                if (round(x), round(y)) in obstacles:
                    return False 
            return True 

        vis_map = {}
        for x1 in range(n):
            for y1 in range(n):
                for x2 in range(n):
                    for y2 in range(n):
                        vis_map[((x1, y1), (x2, y2))] = is_visible((x1, y1), (x2, y2))
        Observation.vis_map = vis_map



class TransitionModel(pomdp_py.TransitionModel):

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
        self.los = los # required because state is linked to observation (accum of observed squares = state.seen)
    
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
        if expected in self.obstacles: # bounce against obstacle -> no move
            return position
        else:
            return (max(0, min(position[0] + action.motion[0], self._n-1)),
                    max(0, min(position[1] + action.motion[1], self._n-1)))
    
    def probability(self, next_state, state, action, normalized=False, **kwargs):
        if next_state != self.sample(state, action):
            return EPSILON
        else:
            return 1.0 - EPSILON
    
    def sample(self, state, action):
        """ 
        terminal = False
        if len(set(state.seen)) == self._n**2:
            # Question: where update `seen`, since affected by observation
            #   maybe: first compute new post
            terminal = True
        """
        next_terminal = False
        if state.terminal:
            next_terminal = True  # already terminated. So no state transition happens
            next_position = state.position
            next_seen = state.seen
        else:
            if isinstance(action, MoveAction):
                next_position = self._move(state.position, action)
                observation = Observation.from_position(next_position, self.obstacles,
                                                        self._n, self.los)
                next_seen = state.seen | set([c for v, c in zip(observation.view, observation.coords) if v == 'e']) # only retain coords that are viewed
                if len(next_seen) == self._n**2 - len(self.obstacles):
                    next_terminal = True
        return State(next_position, next_seen, next_terminal)
    
    def argmax(self, state, action):
        """Returns the most likely next state"""
        return self.sample(state, action) # model is deterministic

class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, n, obstacles, los):
        """
        Args:
            n (int): size of gridworld (n x n)
            obstacles (list): list of obstacle positions
            los (int): maximum line-of-sight
        """
        self._n = n
        self.obstacles = obstacles
        self.los = los
    
    def probability(self, observation, next_state, action):
        if observation != self.sample(next_state, action):
            return EPSILON
        else:
            return 1.0 - EPSILON

    def sample(self, next_state, action, argmax=False):
        return Observation.from_position(next_state.position, self.obstacles,
                                        self._n, self.los)

class RewardModel(pomdp_py.RewardModel):
    def __init__(self):
        pass

    def sample(self, state, action, next_state, normalized=False, **kwargs):
        if state.terminal:
            return 0  # terminated. No reward
        diff = len(next_state.seen) - len(state.seen) # number of new observed squares
        return 10*diff - 1 # minus 1 for each step
    
    def argmax(self, state, action, next_state, normalized=False, **kwargs):
        raise NotImplementedError

    def probability(self, reward, state, action, next_state, normalized=False, **kwargs):
        raise NotImplementedError

class PolicyModel(pomdp_py.RolloutPolicy):
    """Simple policy model according to problem description.
    Used as rollout policy by MCTS
    """
    def __init__(self, n) -> None:
        self._all_actions = {MoveEast, MoveWest, MoveNorth, MoveSouth}
        self._n = n
    
    def sample(self, state, normalized=False, **kwargs):
        return random.sample(self.get_all_actions(state=state), 1)[0]
    
    def probability(self, action, state, normalized=False, **kwargs):
        raise NotImplementedError

    def argmax(self, state, normalized=False, **kwargs):
        """Returns the most likely reward"""
        raise NotImplementedError
    
    def get_all_actions(self, **kwargs):
        "returns all valid actions"
        state = kwargs.get("state", None)
        if state is None:
            return self._all_actions
        else:
            motions = set(self._all_actions)
            rover_x, rover_y = state.position
            if rover_x == 0:
                motions.remove(MoveWest)
            if rover_y == 0:
                motions.remove(MoveNorth)
            if rover_y == self._n - 1:
                motions.remove(MoveSouth)
            return motions

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state), 1)[0]

class ExploreProblem(pomdp_py.POMDP):
    @staticmethod
    def random_free_location(n, not_free_locs):
        """returns a random (x,y) location in nxn grid that is free."""
        while True:
            loc = (random.randint(0, n-1),
                   random.randint(0, n-1))
            if loc not in not_free_locs:
                return loc
    
    @staticmethod
    def generate_instance(n, k, type='random', **kwargs):
        """Returns an init_state and obstacle locations for an instance of Explore(n, k)

        Args:
            n (int): size of gridworld
            k (int): number of obstacles
            type (str, optional): How obstacles are placed. Defaults to 'random'.

        Returns:
            tuple: init_state (type: State), obstacles (type: list)
        """
        # TODO: imrove this such that there are never completely blocked squares?
        agent_position = [0, 1] #random.randint(0, n-1)] #initial position of agent
        obstacles = []
        if type == 'random':
            for _ in range(k):
                loc = ExploreProblem.random_free_location(n, obstacles + agent_position)
                obstacles.append(loc)
        elif type == 'preconfigured':
            filename = kwargs.get('filename')
            #with open('./defense/terrains/'+filename) as f:
            with open('/home/koen/Programming/pomdp-py/pomdp_problems/defense/terrains/'+filename) as f:
                for j, line in enumerate(f):
                    n = len(line)-1 # takes '/n' into account
                    for i, val in enumerate(line[:-1]):
                        if val == 'x': # add obstacle
                            obstacles.append((i,j))
                        if val == 'A':
                            agent_position = (i, j)

        else:
            raise ValueError(f"Initialization type {type} not recognized")
        
        init_state = State(tuple(agent_position), set(), False)
        return init_state, obstacles
    
    def print_state(self):
        string = "\n____MAP____\n"
        agent_position = self.env.state.position
        # true map
        for y in range(self._n):
            for x in range(self._n):
                char = "."
                if (x,y) in self._obstacles:
                    char = 'x'
                elif (x,y) == agent_position:
                    char = "A"
                elif (x, y) in self.env.state.seen:
                    char = 'o'
                string += char
            string += "\n"
        print(string)
        print(f"state = {self.env.state}")

    def __init__(self, n, k, los, init_state, obstacles, init_belief):
        self._n, self._k = n, k
        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(n),
                               TransitionModel(n, obstacles, los),
                               ObservationModel(n, obstacles, los),
                               RewardModel())
        env = pomdp_py.Environment(init_state, 
                                   TransitionModel(n, obstacles, los),
                                   RewardModel())
        self._obstacles = obstacles
        super().__init__(agent, env, name="ExploreProblem")

def init_particles_belief(n, k, num_particles=200, belief="uniform"):
    particles = []
    for _ in range(num_particles):
        if belief == 'uniform':
            state, _ = ExploreProblem.generate_instance(n, k)
        else:
            raise ValueError(f"Belief type {belief} not recognized")
        particles.append(state)
    init_belief = pomdp_py.Particles(particles)
    return init_belief

def test_planner(problem, planner, nsteps=3, discount=0.95):
    gamma = 1.0
    total_reward = 0
    total_discounted_reward = 0
    for i in range(nsteps):
        print(f"==== Step {i+1} ====")
        action = planner.plan(problem.agent)
        true_state = copy.deepcopy(problem.env.state)
        env_reward = problem.env.state_transition(action, execute=True)
        true_next_state = copy.deepcopy(problem.env.state)

        real_observation = problem.env.provide_observation(problem.agent.observation_model,
                                                            action)
        problem.agent.update_history(action, real_observation)
        planner.update(problem.agent, action, real_observation)
        
        # bookkeeping of intermediary results
        total_reward += env_reward
        total_discounted_reward += env_reward * gamma
        gamma *= discount

        print(f'True state: {true_state}')
        print(f"Action: {str(action)}")
        #print(f"Observation: {str(real_observation)}") # -> only activate for debugging
        print("Reward (Cumulative): %s" % str(total_reward))
        print("Reward (Cumulative Discounted): %s" % str(total_discounted_reward))
        if isinstance(planner, pomdp_py.POUCT):
            print("__num_sims__: %d" % planner.last_num_sims)
            print("__plan_time__: %.5f" % planner.last_planning_time)
        if isinstance(planner, pomdp_py.PORollout):
            print("__best_reward__: %d" % planner.last_best_reward)
        print("World:")
        problem.print_state()
        if true_state.terminal:
            break

def test_with_planner():
    n, k, los = 10, 15, 4 # change n manually
    #init_state, obstacles = ExploreProblem.generate_instance(n, k, type='random')
    init_state, obstacles = ExploreProblem.generate_instance(n, k, type='preconfigured', filename='terrain02.ter')
    init_belief = init_particles_belief(n, k, los)
    problem = ExploreProblem(n, k, los, init_state, obstacles, init_belief)

    print("*** Testing POMCP ***")
    pomcp = pomdp_py.POMCP(max_depth=10, discount_factor=0.95,  # max_depth (int) – Depth of the MCTS tree
                           num_sims=1000, exploration_const=20, # num_sims (int) = Number of simulations for each planning step
                           rollout_policy=problem.agent.policy_model,
                           num_visits_init=1)
    test_planner(problem, pomcp, nsteps=100, discount=0.95)

def test_no_planner():
    n, k, los = 20, 10, 100
    #init_state, obstacles = ExploreProblem.generate_instance(n, k)
    init_state, obstacles = ExploreProblem.generate_instance(n, k, type='preconfigured', filename='terrain02.ter')
    k = len(obstacles)
    n = 10
    init_belief = init_particles_belief(n, k, los)
    problem = ExploreProblem(n, k, los, init_state, obstacles, init_belief)
    problem.print_state()
    actions = [MoveEast, MoveSouth, MoveWest, MoveNorth]
    for iter in range(10):
        print(f'================ {iter} ================')
        action = random.choice(actions)
        action = MoveSouth
        print(f" *** Action = {action}")
        env_reward = problem.env.state_transition(action, execute=True)
        real_observation = problem.env.provide_observation(problem.agent.observation_model, action)
        problem.print_state()
        # print(real_observation) # -> only activate for debugging
        print(problem.env.state,'   ', env_reward)
        problem.agent.update_history(action, real_observation)


def test_load_map():
    init_state, obstacles = ExploreProblem.generate_instance(0, 0, type='preconfigured', filename='terrain01.ter')
    print(obstacles)
    print(init_state)
    print(len(obstacles))

if __name__ == "__main__":
    test_with_planner()
    #test_no_planner()
    #test_load_map()