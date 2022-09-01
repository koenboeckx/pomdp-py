# %%
import time, random

import numpy as np

# %%
class Game:
    # skeleton structure for game (in this case single palyer)
    def __init__(self, N) -> None:
        # goal is to find most efficient route through A (always moving rightward) to reach highest value in last column
        # position is terminal if last col is reached
        self.N = N
        self.A = np.zeros((N, N))
        self.A[:, -1] = np.random.randint(0, high=N, size=N)
    
    def move(self, position, step):
        "step in [-1, 0, 1]"
        assert step in self.all_moves(), f'step {step} not allowed'
        row, col  = position
        new_row, new_col = min(self.N, max(0, row+step)), col+1
        terminal = self.terminal((new_row, new_col))
        return (new_row, new_col), terminal
    
    def terminal(self, position):
        row, col  = position
        if col > self.N - 1:
            print('hold')
        return col == self.N - 1
    
    def all_moves(self):
        return [-1, 0, 1]
    
    def value(self, position):
        row, col  = position
        assert row in range(self.N), f'row {row} not allowed'
        assert col in range(self.N), f'column {col} not allowed'
        return self.A[row, col]        

# %%
class MCTS:
    def __init__(self, game, T=1, c=2):
        """T = time to explore before returning anwser
           c = exploration factor"""
        self.game = game
        self.T = T
        self.c = c

        self.visits = {}
        self.values = {}
    
    def run(self, root):
        self.visits[root], self.values[root] = 0, 0
        self.tree = {}
        start_time = time.time()
        while time.time() - start_time < self.T:
            leaf_node, trace = self.select(root)
            if self.game.terminal(leaf_node):
                value = self.game.value(leaf_node)
            else:
                children = self.expand(leaf_node)
                next_child, _ = children[0] # always start with first child from new children
                value = self.rollout(next_child)
            self.backprop(value, trace)
        root_vals = [(self.values[child]/self.visits[child], move) for child, move in self.tree[root]]
        _, move = sorted(root_vals)[-1] # pick move that leads to child with highest value/visit
        return move
    
    def select(self, node):
        trace = [node]
        while node in self.tree: # if not in tree -> expand (i.e. is leaf node)
            N = self.visits[node]
            uct = []
            for child, _ in self.tree[node]:
                if child not in self.visits or self.visits[child] == 0:
                    uct.append((np.infty, child))
                else:
                    uct.append((self.c * np.sqrt(np.log(N)/self.visits[child]), child))
            v, node = max(uct)
            trace.append(node)
        return node, trace

    def expand(self, node) -> None:
        children = [(self.game.move(node, move)[0], move) for move in self.game.all_moves()]
        self.tree[node] = children
        for child, _ in children:
            self.visits[child], self.values[child] = 0, 0
        return children

    def _rollout_policy(self, position):
        return random.choice(self.game.all_moves())

    def rollout(self, position) -> float:
        terminal = self.game.terminal(position)
        if terminal:
            self.game.value(position)
        while not terminal:
            move = self._rollout_policy(position)
            position, terminal = self.game.move(position, move)
        return self.game.value(position)

    def backprop(self, value, trace) -> None:
        for node in trace:
            self.visits[node] += 1
            self.values[node] += value


if __name__ == '__main__':
    game = Game(10)
    print(game.A)
    mcts = MCTS(game)
    position = (0, 0)
    while not game.terminal(position):
        move = mcts.run(position)
        print(move)
        position, _ = game.move(position, move)
    print(game.value(position))




