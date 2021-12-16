import numpy as np

from agents.agent import Agent

class Node:
    def __init__(self):
        self.children = []
        self.value = None
        self.reward = 0.0
        self.visits = 0

    def is_terminal(self):
        return len(self.children) == 0


class Tree:
    def __init__(self):
        self.__root = Node()
        
    

class MCTSEscaper(Agent):

    def __init__(self, start_state, end_state, env):
        super().__init__(start_state, env)
        self._end_state = end_state
        self._policy = None

    def prepare(self):
        #G_final = self._mcts(self._end_state)
        #self._policy = self._policy_mcts(G_final)
        pass
        
    def _plan(self):
        #return self._policy[self._state[0], self._state[1]]
        pass

    def is_obj_completed(self):
        return (self.get_state() == self._end_state).all()

    def print_completion_msg(self):
        print("Escaper escaped the maze!")

    def get_color(self):
        return 1.0
