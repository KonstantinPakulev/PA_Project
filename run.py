import numpy as np
from environment import Environment
from agents.vi_escaper import VIEscaper
from agents.pursuer import Pursuer


if __name__ == "__main__":
    vie_start_state = np.array([11, 6])
    p_start_state = np.array([7, 24])
    end_state = np.array([15, 29])

    env = Environment()

    vi_escaper = VIEscaper(vie_start_state, end_state, env)
    pursuer = Pursuer(p_start_state, env, vi_escaper)

    agents = [vi_escaper, pursuer]

    # env.run(agents, end_state, 41)
    env.run(agents, end_state, 41, save_as="vi")
