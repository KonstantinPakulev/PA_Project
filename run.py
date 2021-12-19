import numpy as np
import argparse

from environment import Environment
from agents.vi_escaper import VIEscaper, MaskedVIEscaper
from agents.mdp_escaper import MDPEscaper
from agents.pursuer import Pursuer
from agents.a_star_pursuer import AStarPursuer

# --escaper vi --num_pursuers 0 --save_as vi
# --escaper vi --num_pursuers 1 --save_as vi_no_escape
# --escaper mvi --escaper_params 1.0 --num_pursuers 1 --save_as mvi_escape
# --escaper mvi --escaper_params 1.0 --num_pursuers 5 --save_as mvi_no_escape
# --escaper mdp --escaper_params 3 --num_pursuers 5 --save_as mdp_escape


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--escaper', help='escaper algorithm to use')
    parser.add_argument('--escaper_params', default=3, help='a string of parameters (if any) for escaper')
    parser.add_argument('--num_pursuers', type=int, default=1, help='number of pursuers from 0 to 5')
    parser.add_argument('--pursuer', type=str, default='default', help='pursuer algorithm to use')
    parser.add_argument('--save_as', nargs='?', help='save as mp4 with specified name')
    parser.add_argument('--num_iter', type=int, nargs='?', default=300)
    parser.add_argument('--heuristic_function', type=str, default='euclidean', nargs='?', help='heuristic function used by A* pursuer')

    args = parser.parse_args()

    env = Environment()

    e_start_state = np.array([11, 6])
    end_state = np.array([15, 29])
    

    if args.escaper == 'vi':
        escaper = VIEscaper(e_start_state, end_state, env)

    elif args.escaper == 'mvi':
        escaper = MaskedVIEscaper(e_start_state, end_state, env, float(args.escaper_params))

    elif args.escaper == 'mdp':
        escaper = MDPEscaper(e_start_state, end_state, env, int(args.escaper_params))

    else:
        raise ValueError(f"No such escaper algorithm: {args.escaper}")

    agents = [escaper]

    p_start_states = [np.array([7, 20]),
                      np.array([5, 17]),
                      np.array([0, 5]),
                      np.array([20, 7]),
                      np.array([26, 16])]
    
    if args.pursuer == 'default':
        pur = Pursuer
    elif args.pursuer == 'a_star':
        pur = AStarPursuer
        if args.heuristic_function == 'manhattan':
            h = Environment.h_manhattan
        elif args.heuristic_function == 'euclidean':
            h = Environment.h_euclidean

    for p_ss in p_start_states[:args.num_pursuers]:
        agents.append(pur(p_ss, env, h))

    if args.save_as is not None:
        env.run(agents, end_state, args.num_iter, save_as=args.save_as)

    else:
        env.run(agents, end_state, args.num_iter)
