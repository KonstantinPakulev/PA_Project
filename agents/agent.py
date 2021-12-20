from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(self, start_state, env):
        self._env = env
        self._start_state = start_state
        self._state = start_state

    def reset(self):
        self._state = self._start_state

    @abstractmethod
    def prepare(self):
        ...

    def plan_and_move(self):
        u = self._plan()

        if u is None:
            raise ValueError("You need to implement 'plan' method")

        # print(u, self._state)

        self._state = self._state + u

        print(f'Actor \'{self.__class__}\' moved to {self._state}')

    @abstractmethod
    def is_obj_completed(self):
        ...

    @abstractmethod
    def print_completion_msg(self):
        ...

    def get_state(self):
        return self._state

    @abstractmethod
    def get_color(self):
        ...

    @abstractmethod
    def _plan(self):
        ...

    def _try2move(self, state, u):
        new_state = state + u

        is_moved = self._check_collision(self._env._map_layout, new_state)

        return new_state, is_moved

    @staticmethod
    def _check_collision(map_layout, state):
        if state[0] < 0 or \
           state[1] < 0 or \
           state[0] >= map_layout.shape[0] or \
           state[1] >= map_layout.shape[1]:

            return False

        if map_layout[state[0], state[1]] >= 1.0 - 1e-4:
            return False

        return True
