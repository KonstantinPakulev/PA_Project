import numpy as np

from agents.agent import Agent
from agents.vi_escaper import vi

PURSUER_COST = 500


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
        self._R_path = self._get_R_path()

    def _plan(self):
        R_p = self._get_R_p()
        U_p = self._bellman(R_p)

        u = self._get_best_action(self.get_state(), -U_p + 60 * self._R_path, return_action=True)

        return u

    def is_obj_completed(self):
        return (self.get_state() == self._end_state).all()

    def print_completion_msg(self):
        print("Escaper escaped the maze!")

    def get_color(self):
        return 1.0

    def _get_R_path(self):
        G_final = vi(self._env._map_layout, self._env._actions, self._try2move,
                     self._end_state)

        max_value = G_final.max()

        R_path = -G_final + max_value
        R_path[R_path == (max_value + 1)] = -1.0

        return R_path

    def _get_R_p(self):
        R_p = np.zeros_like(self._R_path)

        for p in self._env.get_pursuers():
            state = p.get_state()
            R_p[state[0], state[1]] = PURSUER_COST

        return R_p

    def _bellman(self, R, gamma=0.9):
        U_prev = np.zeros_like(R)
        U_curr = np.zeros_like(R)

        for _ in np.arange(self._bellman_num_iter):
            for y in np.arange(R.shape[0]):
                for x in np.arange(R.shape[1]):
                    if self._R_path[y, x] != -1.0:
                        U_curr[y, x] = R[y, x] + gamma * self._get_best_action(np.array([y, x]), U_prev)

            U_prev = np.copy(U_curr)
            U_curr[:, :] = 0.0

        return U_prev

    def _get_best_action(self, state, U, return_action=False):
        max_e_u_value = float('-inf')
        max_u = np.array([0, 0])

        for p_u in self._p_actions:
            e_u_value = 0
            is_feasible = True

            for i, (u, p) in enumerate(zip(p_u, self._p_actions_prob)):
                new_state, is_moved = self._try2move(np.array([state[0], state[1]]), u)

                if is_moved:
                    e_u_value += p * U[new_state[0], new_state[1]]

                else:
                    e_u_value += p * U[state[0], state[1]]

                if i == 0 and not is_moved:
                    is_feasible = False
                    break

            if is_feasible:
                if e_u_value > max_e_u_value:
                    max_e_u_value = e_u_value
                    max_u = p_u[0]

        if return_action:
            return max_u

        else:
            return max_e_u_value
