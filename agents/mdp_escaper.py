import numpy as np

from agents.agent import Agent


class MDPEscaper(Agent):

    def __init__(self, start_state, end_state, env, bellman_num_iter=5):
        super().__init__(start_state, env)
        self._end_state = end_state

        self._bellman_num_iter = bellman_num_iter

        self._p_actions = np.stack([env._actions,
                                    np.roll(env._actions, -1, 0),
                                    np.roll(env._actions, 1, 0)], 1)
        self._p_actions_prob = np.array([0.8, 0.1, 0.1])

        self._R_path = None

    def prepare(self):
        G_final = self._vi(self._end_state)
        max_value = G_final.max()

        self._R_path = -G_final + max_value
        self._R_path[self._R_path == (max_value + 1)] = -1.0
        self._R_path[self._end_state[0], self._end_state[1]] = 1000

    def _plan(self):
        U_curr = self._bellman()

        u = self._get_best_action(self.get_state(), U_curr, return_action=True)

        return u

    def is_obj_completed(self):
        return (self.get_state() == self._end_state).all()

    def print_completion_msg(self):
        print("Escaper escaped the maze!")

    def get_color(self):
        return 1.0

    def _vi(self, end_state, num_iter=100):
        map_layout = self._env._map_layout
        actions = self._env._actions

        max_value = map_layout.shape[0] * map_layout.shape[1]

        G_prev = np.ones((map_layout.shape[0], map_layout.shape[1])) * max_value
        G_curr = np.ones((map_layout.shape[0], map_layout.shape[1])) * max_value

        G_prev[end_state[0], end_state[1]] = 0

        G_final = np.ones((map_layout.shape[0], map_layout.shape[1])) * -1

        for i in np.arange(1, num_iter + 1):
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

    def _bellman(self, gamma=0.9):
        R = np.copy(self._R_path)

        for a in self._env.get_pursuers():
            s = a.get_state()
            R[s[0], s[1]] = -500

        U_prev = np.zeros_like(R)
        U_curr = np.zeros_like(R)

        for _ in np.arange(self._bellman_num_iter):
            for y in np.arange(R.shape[0]):
                for x in np.arange(R.shape[1]):
                    if R[y, x] != -1.0 and\
                            R[y, x] != -500:
                        U_curr[y, x] = R[y, x] + gamma * self._get_best_action(np.array([y, x]), U_prev)

            U_prev = np.copy(U_curr)
            U_curr[:, :] = 0.0

        return U_prev

    def _get_best_action(self, state, V, return_action=False):
        max_e_u_value = float('-inf')
        max_u = np.array([0, 0])

        for p_u in self._p_actions:
            e_u_value = 0
            is_feasible = True
            for i, (u, p) in enumerate(zip(p_u, self._p_actions_prob)):
                new_state, is_moved = self._try2move(np.array([state[0], state[1]]), u)

                if is_moved:
                    e_u_value += p * V[new_state[0], new_state[1]]

                else:
                    e_u_value += p * V[state[0],  state[1]]

                if i == 0 and not is_moved:
                    is_feasible = False
                    break

            if e_u_value > max_e_u_value and is_feasible:
                max_e_u_value = e_u_value
                max_u = p_u[0]

        if return_action:
            return max_u

        else:
            return max_e_u_value
