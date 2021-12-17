import numpy as np
from environment import Environment
from agents.vi_escaper import VIEscaper
from agents.mdp_escaper import MDPEscaper
from agents.pursuer import Pursuer


if __name__ == "__main__":
    e_start_state = np.array([11, 6])
    end_state = np.array([15, 29])

    p_start_states = [np.array([7, 20]),
                      np.array([5, 17]),
                      np.array([0, 5]),
                      np.array([26, 7])]

    env = Environment()

    # vi_escaper = VIEscaper(start_state, end_state, env)
    mdp_escaper = MDPEscaper(e_start_state, end_state, env)

    agents = [mdp_escaper]

    for p_ss in p_start_states:
        agents.append(Pursuer(p_ss, env))

    # env.run(agents, end_state, 300)
    env.run(agents, end_state, 300, save_as="mdp")
