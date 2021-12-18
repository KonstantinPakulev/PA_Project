import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

matplotlib.use("TkAgg")


class Environment:

    def __init__(self, layout_path='./data.npz'):
        layout = np.load(layout_path)

        self._map_layout = layout['environment']
        self._actions = layout['actions']

        self._agents = None

    def run(self, agents, end_state, num_iter, save_as=None):
        self._agents = agents

        for a in self._agents[::-1]:
            a.reset()
            a.prepare()

        fig = plt.figure()

        if save_as is not None:
            images = [[self._plot_env(end_state)]]

            for i in np.arange(num_iter):
                is_obj_completed = self._run_iter()

                images.append([self._plot_env(end_state)])

                if is_obj_completed:
                    break

            ani = animation.ArtistAnimation(fig, images, interval=100, repeat=False)

            if not os.path.exists("video"):
                os.mkdir("video")

            ani.save(f"video/{save_as}.mp4")

        else:
            self._plot_env(end_state)
            plt.pause(0.1)

            for i in np.arange(num_iter):
                is_obj_completed = self._run_iter()

                plt.clf()
                self._plot_env(end_state)
                plt.pause(0.1)

                if is_obj_completed:
                    break

            plt.show()

    def get_escaper(self):
        return self._agents[0]

    def get_pursuers(self):
        return self._agents[1:]

    def _run_iter(self):
        is_obj_completed = False

        for a in self._agents[::-1]:
            a.plan_and_move()

            if a.is_obj_completed():
                a.print_completion_msg()

                is_obj_completed = True
                break

        if not is_obj_completed:
            for a in self._agents[1::-1]:
                if a.is_obj_completed():
                    a.print_completion_msg()

                    is_obj_completed = True
                    break

        return is_obj_completed

    def _plot_env(self, end_state):
        map_layout = np.copy(self._map_layout)

        for a in self._agents:
            s = a.get_state()
            map_layout[s[0], s[1]] = a.get_color()

        map_layout[end_state[0], end_state[1]] = 0.3

        return plt.imshow(map_layout)
