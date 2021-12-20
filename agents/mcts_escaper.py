import copy
import numpy as np
from collections import deque

from agents.agent import Agent
from agents.vi_escaper import vi, policy_vi


class MCTSEscaper(Agent):

    def __init__(self, start_state, end_state, env,
                 num_iter=100, num_sim_iter=4, time_penalty=False):
        super().__init__(start_state, env)
        self._end_state = end_state

        self._num_iter = num_iter
        self._num_sim_iter = num_sim_iter
        self._time_penalty = time_penalty

        self._R_path = None
        self._policy = None

    def prepare(self):
        G_final = vi(self._env._map_layout, self._env._actions, self._try2move, self._end_state)

        self._R_path = G_final
        self._policy = policy_vi(self._env._map_layout, self._env._actions, self._try2move, G_final)

    def _plan(self):
        tree = MCTSTree(self._env)

        root = MCTSNode(tree, None, self._env._agents,
                        None, 0, self._get_remaining_actions(self.get_state(), tree._visited))

        tree.set_root(root)

        for i in np.arange(self._num_iter):
            node = self._tree_policy(tree)

            reward = self._default_policy(node)

            self._backpropagate(node, reward)

        u = tree.get_best_action()

        return u

    def _tree_policy(self, tree):
        node = tree.select()

        if not node.is_terminal():
            sim_env = self._env.copy()

            u = node.get_action()

            agents = copy.deepcopy(node._agents)
            agents[0] = ActionEscaper(agents[0].get_state(), sim_env, u, self._end_state)

            for a in agents:
                a._env = sim_env

            sim_env.run(agents, 1)

            new_state = agents[0].get_state()

            r_value = self._R_path[new_state[0], new_state[1]]

            e_node = MCTSNode(tree, node, agents, u, r_value, self._get_remaining_actions(new_state, tree._visited))

            node.add_child(e_node)

            return e_node

        else:
            return node

    def _default_policy(self, node):
        reward = self._reward(node._agents)

        for i, a in enumerate(node._agents[::-1]):
            if a.is_obj_completed():
                if i == len(node._agents) - 1:
                    return 100 + reward

                else:
                    return -100

        sim_env = self._env.copy()

        agents = copy.deepcopy(node._agents)
        agents[0] = ActionEscaper(agents[0].get_state(), sim_env, self._policy, self._end_state)

        for a in agents:
            a._env = sim_env

        for i in np.arange(self._num_sim_iter):
            is_obj_completed = sim_env.run(agents, 1)

            if is_obj_completed:
                break

            reward += self._reward(agents)

        if is_obj_completed:
            if agents[0].is_obj_completed():
                reward += 1000

                if self._time_penalty:
                    reward -= (i + 1)

            else:
                return -100 + reward
        else:
            if self._time_penalty:
                reward -= (i + 1)

        return reward

    def _reward(self, agents):
        e_state = agents[0].get_state()

        dist = np.clip(self._R_path[e_state[0], e_state[1]], a_min=0.5, a_max=None)

        return 0.1 / dist

    @staticmethod
    def _backpropagate(node, reward):
        while node is not None:
            node._reward += reward
            node._n_visited += 1

            node = node._parent

    def is_obj_completed(self):
        return (self.get_state() == self._end_state).all()

    def print_completion_msg(self):
        print("Escaper escaped the maze!")

    def get_color(self):
        return 1.0

    def _get_remaining_actions(self, state, visited):
        remaining_actions = deque([])

        for u in self._env._actions:
                new_state, is_moved = self._try2move(state, u)

                if is_moved and not visited[new_state[0], new_state[1]]:
                    remaining_actions.append(u)

        return remaining_actions


class ActionEscaper(Agent):

    def __init__(self, start_state, env, policy, end_state):
        super().__init__(start_state, env)
        self._policy = policy
        self._end_state = end_state

    def prepare(self):
        pass

    def is_obj_completed(self):
        return (self.get_state() == self._end_state).all()

    def print_completion_msg(self):
        print("Escaper escaped the maze!")

    def get_color(self):
        return 1.0

    def _plan(self):
        if len(self._policy.shape) == 1:
            return self._policy

        else:
            return self._policy[self._state[0], self._state[1]]


class MCTSNode:

    def __init__(self, tree, parent, agents, action, r_value, remaining_actions):
        self._reward = 0.0
        self._n_visited = 0

        self._tree = tree
        self._parent = parent
        self._children = []

        self._agents = agents
        self._action = action
        self._r_value = r_value
        self._remaining_actions = remaining_actions

    def has_children(self):
        return len(self._children) != 0

    def is_terminal(self):
        return len(self._remaining_actions) == 0

    def get_action(self):
        return self._remaining_actions.popleft()

    def add_child(self, node):
        state = node._agents[0].get_state()
        self._tree._visited[state[0], state[1]] = True

        self._children.append(node)


class MCTSTree:

    def __init__(self, env):
        self._visited = np.zeros_like(env._map_layout, dtype=np.bool)

    def set_root(self, root):
        state = root._agents[0].get_state()
        self._visited[state[0], state[1]] = True

        self._root = root

    def select(self):
        node = self._root

        while node.is_terminal():
            if not node.has_children():
                break

            node = self._select_best_child(node)

        return node

    def get_best_action(self):
        return self._select_best_child(self._root)._action

    @staticmethod
    def _select_best_child(node):
        best_child = None
        best_child_value = float('-inf')
        best_child_r_value = float('inf')

        for c in node._children:
            c_value = c._reward / c._n_visited

            if c_value > best_child_value:
                best_child_value = c_value
                best_child_r_value = c._r_value
                best_child = c

            elif c_value == best_child_value:
                if c._r_value < best_child_r_value:
                    best_child_value = c_value
                    best_child_r_value = c._r_value
                    best_child = c

        return best_child
