import numpy as np

from scipy.ndimage.filters import gaussian_filter

from agents.agent import Agent


PURSUER_COST = 500


class VIEscaper(Agent):

    def __init__(self, start_state, end_state, env):
        super().__init__(start_state, env)
        self._end_state = end_state

        self._policy = None
    
    def prepare(self):
        G_final = vi(self._env._map_layout, self._env._actions, self._try2move, self._end_state)
        self._policy = policy_vi(self._env._map_layout, self._env._actions, self._try2move, G_final)

    def _plan(self):
        return self._policy[self._state[0], self._state[1]]

    def is_obj_completed(self):
        return (self.get_state() == self._end_state).all()

    def print_completion_msg(self):
        print("Escaper escaped the maze!")

    def get_color(self):
        return 1.0


class MaskedVIEscaper(VIEscaper):

    def __init__(self, start_state, end_state, env, pursuer_sigma):
        super().__init__(start_state, end_state, env)
        self._pursuer_sigma = pursuer_sigma

        self._G_final = None

    def prepare(self):
        self._G_final = vi(self._env._map_layout, self._env._actions, self._try2move,
                           self._end_state)

    def _plan(self):
        G_p = self._get_G_p()
        _policy = policy_vi(self._env._map_layout, self._env._actions, self._try2move, G_p)

        return _policy[self._state[0], self._state[1]]

    def _get_P(self):
        P = np.zeros_like(self._G_final)
        G_mask = self._G_final != -1

        for p in self._env.get_pursuers():
            state = p.get_state()

            if self._pursuer_sigma != -1:
                Pi = np.zeros_like(self._G_final)

                Pi[state[0], state[1]] = 1
                Pi = gaussian_filter(Pi, sigma=self._pursuer_sigma)
                Pi = Pi / Pi.max() * PURSUER_COST

                P[G_mask] += Pi[G_mask]

            else:
                P[state[0], state[1]] = PURSUER_COST

        return P

    def _get_G_p(self):
        return self._G_final + self._get_P()


"""
Support utils
"""


def vi(map_layout, actions, try2move_func, end_state, max_num_iter=100):
    max_value = map_layout.shape[0] * map_layout.shape[1]

    G_prev = np.ones((map_layout.shape[0], map_layout.shape[1])) * max_value
    G_curr = np.ones((map_layout.shape[0], map_layout.shape[1])) * max_value

    G_prev[end_state[0], end_state[1]] = 0

    G_final = np.ones((map_layout.shape[0], map_layout.shape[1])) * -1

    for i in np.arange(1, max_num_iter + 1):
        for y in np.arange(map_layout.shape[0]):
            for x in np.arange(map_layout.shape[1]):
                if G_prev[y, x] != max_value:
                    for u in actions:
                        new_state, is_moved = try2move_func(np.array([y, x]), u)

                        if is_moved and G_final[new_state[0], new_state[1]] == -1:
                            G_curr[new_state[0], new_state[1]] = min(G_curr[new_state[0], new_state[1]],
                                                                     G_prev[y, x] + 1)

                    G_final[y, x] = G_prev[y, x]

        if np.all(G_curr == max_value):
            print(f"VI converged on iteration: {i}")
            break

        G_prev = np.copy(G_curr)
        G_curr[:, :] = max_value

    if not np.all(G_curr == max_value):
        print("Warning! VI didn't converge")

    return G_final


def policy_vi(map_layout, actions, try2move_func, G_final):
    max_value = map_layout.shape[0] * map_layout.shape[1]
    policy = np.zeros((G_final.shape[0], G_final.shape[1], 2), dtype=np.int)

    for y in np.arange(G_final.shape[0]):
        for x in np.arange(G_final.shape[1]):
            if G_final[y, x] != -1:
                min_cost = max_value
                min_u = None

                for u in actions:
                    new_state, is_moved = try2move_func(np.array([y, x]), u)

                    if is_moved:
                        if G_final[new_state[0], new_state[1]] < min_cost:
                            min_cost = G_final[new_state[0], new_state[1]]
                            min_u = u

                if min_u is not None:
                    policy[y, x] = np.array(min_u)

    return policy