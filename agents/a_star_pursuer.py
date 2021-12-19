import numpy as np

from agents.pursuer import *


class AStarPursuer(Pursuer):

    def __init__(self, start_state, env, heuristic):
        super().__init__(start_state, env)
        self._heuristic = heuristic

    def _plan(self):
        return self._policy()

    def _policy(self):
        actions = self._env._actions
        costs = np.full(len(actions), fill_value=np.inf)
        
        for i in range(len(actions)):
            new_state, is_moved = self._try2move(self.get_state(), actions[i])
            if is_moved:
                costs[i] = self._heuristic(new_state, self._env.get_escaper().get_state()) 
        return actions[np.argmin(costs)]
