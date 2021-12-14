import numpy as np

from agents.agent import Agent


class Pursuer(Agent):

    def __init__(self, start_state, env, escaper):
        super().__init__(start_state, env)
        self._escaper = escaper

    def prepare(self):
        pass

    def is_obj_completed(self):
        return (self._escaper.get_state() == self.get_state()).all()

    def print_completion_msg(self):
        print("Pursuer captured the escaper!")

    def get_color(self):
        return 0.6

    def _plan(self):
        actions = self._env._actions

        u_idx = self._policy()

        _, is_moved = self._try2move(self.get_state(), actions[u_idx])

        if is_moved:
            return actions[u_idx]

        else:
            return np.array([0, 0])

    def _policy(self):
        ds = np.array(self._escaper.get_state()) - np.array(self.get_state())

        theta = np.arctan2(ds[1], ds[0])
        theta = (theta + np.pi) / np.pi * 2

        u_idx = np.floor(theta)
        
        return int(u_idx % 4)
