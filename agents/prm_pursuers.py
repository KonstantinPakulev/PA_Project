from math import sqrt

import numpy as np
from numpy.random import randint

from agents.agent import Agent
from environment import Environment


class PRMPursuers(Agent):

    def __init__(self, start_state, env):
        self.n_pursuers = len(start_state)
        start_state = tuple([State(s[0], s[1]) for s in start_state])
        super().__init__(start_state, env)
        self.PRM = {}
        self.n_samples = 900
        self.count_sampled = 0
        self.count_connected = 0
        print(f'Creating pursuers at {start_state}')

    def prepare(self):
        for i in range(self.n_samples):
            if i % 100 == 0 and i != 0:
                print(f'{i+1} samples processed for PRMPursuers')
            q = self.sample()
            self.draw(q)
        q = self._start_state
        self.draw(q)
        print(f'Created pursuers roadmap of size {len(self.PRM.keys())}')
        print(f'{self.count_sampled}, {self.count_connected}')
        # for k1 in self.PRM:
        #     print([self.dist_q(k1, k2) for k2 in self.PRM])

    def draw(self, q):
        if self._check_collision(self._env._map_layout, q):
            self.count_sampled += 1
            self.PRM[q] = set()
        old_keys = [*self.PRM]
        for node in old_keys:
            if node != q:
                dist = self.dist_q(q, node)
                if dist < self.n_pursuers * 9:
                    self.connect(node, q)

    def connect(self, q1, q2):
        last = self.steer(q1, q2)
        if last is None:
            return
        if not self._check_collision(self._env._map_layout, last):
            return
        if self.PRM.get(last) is None:
            self.count_connected += 1
            self.PRM[last] = set()
        self.PRM[last].add(q1)
        self.PRM[q1].add(last)

    def is_obj_completed(self):
        return any([(self._env.get_escaper().get_state()[0] == self_state[0]) and (
                    self._env.get_escaper().get_state()[1] == self_state[1])
                    for self_state in self.get_state()
                    ])

    def print_completion_msg(self):
        print("Pursuer captured the escaper!")

    def get_color(self):
        return 0.6

    def plan_and_move(self):
        u = self._plan()
        self._state = tuple(np.add(self._state, u))
        print(f'Actor \'{self.__class__}\' moved to {self._state}')

    def _plan(self):
        queue = [[self.get_state()]]
        visited = {self.get_state()}
        min_dist = np.inf
        while queue:
            path = queue.pop(0)
            node = path[-1]
            visited.add(node)
            dist = max([self.dist_s(p, self._env.get_escaper().get_state()) for p in node])
            if dist < min_dist:
                min_dist = dist
                print(f'new min dist: {min_dist}')
            if dist <= 15:
                goal = path[1] if len(path) > 1 else path[0]
                goal = list(goal)
                for i, g in enumerate(goal):
                    if self.dist_s(g, self._env.get_escaper().get_state()) < 7:
                        escaper = self._env.get_escaper().get_state()
                        goal[i] = State(escaper[0], escaper[1])
                goal = tuple(goal)
                return self.a_star_plan(goal)
            ajs = self.PRM.get(node)
            if ajs is None:
                self.draw(node)
            ajs = self.PRM.get(node, [])
            for adjacent in ajs:
                if adjacent not in visited:
                    new_path = list(path)
                    new_path.append(adjacent)
                    queue.append(new_path)
        raise ValueError(f"prm didn't converge, min distance: {min_dist}")

    def sample(self):
        x = self._env.get_x_size()
        y = self._env.get_y_size()
        return tuple([State(
            randint(0, x),
            randint(0, y)
        ) for _ in range(self.n_pursuers)])

    def a_star_plan(self, goal):
        u = []
        for s, g in zip(self.get_state(), goal):
            actions = self._env._actions
            costs = np.full(len(actions), fill_value=np.inf)

            for i in range(len(actions)):
                new_state, is_moved = self._try2move(s, actions[i])
                if is_moved:
                    costs[i] = Environment.h_euclidean(new_state, g)
            u.append(State(actions[np.argmin(costs)][0], actions[np.argmin(costs)][1]))
        return tuple(u)

    def steer(self, q1, q2):
        n = max([(s2 - s1).norm1() for s1, s2 in zip(q1, q2)])
        next = q1
        for i in range(1, n):
            step = tuple([((s2 - n1) // (n - i)) for s2, n1 in zip(q2, next)])
            if max([s.norm1() for s in step]) != 0:
                possible = tuple(np.add(next, step))
                if not self._check_collision(self._env._map_layout, possible):
                    if next == q1:
                        return None
                    return next
                next = possible
        return q2

    # tries to move in one direction
    def _try2move(self, state, u):
        new_state = state + u

        is_moved = Agent._check_collision(self._env._map_layout, new_state)

        return new_state, is_moved

    @staticmethod
    def _check_collision(map_layout, q):
        collision = not all(
            ([Agent._check_collision(map_layout, ind_state) for ind_state in q])
        )
        if collision:
            return False
        for i in range(len(q)):
            for j in range(i):
                if (q[i] - q[j]).norm2() < 2:
                    return False
        return True

    @staticmethod
    def dist_q(q1, q2):
        return sum([(s2 - s1).norm1() for s1, s2 in zip(q1, q2)])

    @staticmethod
    def dist_s(s1, s2):
        s1 = np.array([s1[0], s1[1]])
        s2 = np.array([s2[0], s2[1]])
        return np.linalg.norm(s2 - s1, 1)


class State:
    def __init__(self, x: int, y: int):
        self._x = x
        self._y = y

    def __hash__(self):
        return (self._x, self._y).__hash__()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def norm1(self):
        return abs(self.x) + abs(self.y)

    def norm2(self):
        return sqrt(self.x ** 2 + self.y ** 2)

    def __getitem__(self, item):
        if item == 0:
            return self._x
        elif item == 1:
            return self._y
        else:
            raise ValueError

    def __add__(self, other):
        return State(self.x + other[0], self.y + other[1])

    def __sub__(self, other):
        return State(self.x - other[0], self.y - other[1])

    def __floordiv__(self, other):
        return State(
            int(self.x / other), int(self.y / other)
        )

    def __repr__(self):
        return f'({self.x}, {self.y})'

    def __mul__(self, other):
        return self.x * other[0] + self.y * other[1]
