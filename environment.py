import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

matplotlib.use("TkAgg")


class Environment:

    @staticmethod
    def from_file(layout_path='./data.npz'):
        layout = np.load(layout_path)

        map_layout = layout['environment']
        actions = layout['actions']

        return Environment(map_layout, actions)

    def __init__(self, map_layout, actions):
        self._map_layout = map_layout
        self._actions = actions

        self._agents = None

    def run_and_visualize(self, agents, end_state, num_iter, save_as=None):
        self._agents = agents

        for a in self._agents[::-1]:
            a.reset()
            a.prepare()

        fig = plt.figure()

        if save_as is not None:
            images = [[self._plot_env(end_state)]]

            for _ in np.arange(num_iter):
                is_obj_completed = self._run_iter()

                images.append([self._plot_env(end_state)])

                if is_obj_completed:
                    break

            ani = animation.ArtistAnimation(fig, images, interval=100, repeat=False)

            if not os.path.exists("videos"):
                os.mkdir("videos")

            ani.save(f"videos/{save_as}.mp4")

        else:
            self._plot_env(end_state)
            plt.pause(0.1)

            for _ in np.arange(num_iter):
                is_obj_completed = self._run_iter()

                plt.clf()
                self._plot_env(end_state)
                plt.pause(0.1)

                if is_obj_completed:
                    break

            plt.show()

    def run(self, agents, num_iter):
        self._agents = agents

        is_obj_completed = False

        for _ in np.arange(num_iter):
            is_obj_completed = self._run_iter(verbose=False)

            if is_obj_completed:
                break

        return is_obj_completed

    def get_escaper(self):
        return self._agents[0]

    def get_pursuers(self):
        return self._agents[1:]

    def copy(self):
        return Environment(self._map_layout, self._actions)

    def _run_iter(self, verbose=True):
        is_obj_completed = False

        for a in self._agents[::-1]:
            a.plan_and_move()

            if a.is_obj_completed():
                if verbose:
                    a.print_completion_msg()

                is_obj_completed = True
                break

        if not is_obj_completed:
            for a in self._agents[1::]:
                if a.is_obj_completed():
                    if verbose:
                        a.print_completion_msg()

                    is_obj_completed = True
                    break

        return is_obj_completed

    def _plot_env(self, end_state):
        map_layout = np.copy(self._map_layout)

        for a in self._agents:
            s = a.get_state()
            try:
                for oip in s:
                    map_layout[oip.x, oip.y] = a.get_color()
            except AttributeError:
                map_layout[s[0], s[1]] = a.get_color()

        map_layout[end_state[0], end_state[1]] = 0.3

        return plt.imshow(map_layout)

    @staticmethod
    def h_euclidean(src, dst):
        return np.linalg.norm(src - dst)
    
    @staticmethod
    def h_manhattan(src, dst):
        return np.linalg.norm(src - dst, ord=1)

    def get_x_size(self):
        return self._map_layout.shape[0]

    def get_y_size(self):
        return self._map_layout.shape[1]
