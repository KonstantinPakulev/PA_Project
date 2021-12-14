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

    def run(self, agents, end_state, num_iter, save_as=None):
        for a in agents:
            a.reset()
            a.prepare()

        fig = plt.figure()

        is_obj_completed = False

        if save_as is not None:
            images = [[self._plot_env(agents, end_state)]]

            for i in np.arange(num_iter):
                for a in agents:
                    a.plan_and_move()

                    if a.is_obj_completed():
                        a.print_completion_msg()

                        is_obj_completed = True
                        break

                images.append([self._plot_env(agents, end_state)])

                if is_obj_completed:
                    break

            ani = animation.ArtistAnimation(fig, images, interval=100, repeat=False)
            ani.save(f"{save_as}.mp4")

        else:
            self._plot_env(agents, end_state)

            for i in np.arange(num_iter):
                for a in agents:
                    a.plan_and_move()

                    if a.is_obj_completed():
                        a.print_completion_msg()

                        is_obj_completed = True
                        break

                plt.clf()
                self._plot_env(agents, end_state)
                plt.pause(0.1)

                if is_obj_completed:
                    break

            plt.show()

    def _plot_env(self, agents, end_state):
        map_layout = np.copy(self._map_layout)

        for a in agents:
            s = a.get_state()
            map_layout[s[0], s[1]] = a.get_color()

        map_layout[end_state[0], end_state[1]] = 0.3

        return plt.imshow(map_layout)
