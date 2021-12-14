import numpy as np

from agents.agent import Agent


class VIEscaper(Agent):

    def __init__(self, start_state, end_state, env):
        super().__init__(start_state, env)
        self._end_state = end_state

        self._policy = None

    def prepare(self):
        G_final = self._vi(self._end_state)

        self._policy = self._policy_vi(G_final)

    def _plan(self):
        return self._policy[self._state[0], self._state[1]]

    def is_obj_completed(self):
        return (self.get_state() == self._end_state).all()

    def print_completion_msg(self):
        print("Escaper escaped the maze!")

    def get_color(self):
        return 1.0

    def _vi(self, end_state, max_num_iter=100):
        map_layout = self._env._map_layout
        actions = self._env._actions

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
                            new_state, is_moved = self._try2move(np.array([y, x]), u)

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

    def _policy_vi(self, G_final):
        map_layout = self._env._map_layout
        actions = self._env._actions

        max_value = map_layout.shape[0] * map_layout.shape[1]
        policy = np.zeros((G_final.shape[0], G_final.shape[1], 2), dtype=np.int)

        for y in np.arange(G_final.shape[0]):
            for x in np.arange(G_final.shape[1]):
                min_cost = max_value
                min_u = None

                for u in actions:
                    new_state, is_moved = self._try2move(np.array([y, x]), u)

                    if is_moved:
                        if G_final[new_state[0], new_state[1]] < min_cost:
                            min_cost = G_final[new_state[0], new_state[1]]
                            min_u = u

                if min_u is not None:
                    policy[y, x] = np.array(min_u)

        return policy