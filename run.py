import numpy as np
from environment import Environment
from agents.vi_escaper import VIEscaper
from agents.mdp_escaper import MDPEscaper
from agents.pursuer import Pursuer


if __name__ == "__main__":
    e_start_state = np.array([11, 6])
    p_start_state1 = np.array([7, 24])
    # p_start_state2 = np.array([0, 4])

    end_state = np.array([15, 29])

    env = Environment()

    # vi_escaper = VIEscaper(start_state, end_state, env)
    mdp_escaper = MDPEscaper(e_start_state, end_state, env)
    pursuer1 = Pursuer(p_start_state1, env)
    # pursuer2 = Pursuer(p_start_state2, env)
    # pursuer2

    agents = [mdp_escaper, pursuer1]

    # env.run(agents, end_state, 300)
    env.run(agents, end_state, 300, save_as="mdp")
