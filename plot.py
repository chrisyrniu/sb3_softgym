import argparse
from learning_curve_plotter import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for Plotting Learning Curves')
    parser.add_argument('--log_dir', help='the path to the data', type=str, default='results')
    parser.add_argument('--eval_freq', help='evaluation frequence used during training', type=int, default=3000)
    parser.add_argument('--n_eval_episodes', help='the number of episodes for each evaluation during training', type=int, default=10)
    parser.add_argument('--eval_smooth_window_size', help='the sliding window size to smooth the evaluation rewards', type=int, default=10)
    parser.add_argument('--non_eval_sample_freq', help='the sample frequence of the rollout rewards for plotting ', type=int, default=1500)
    parser.add_argument('--non_eval_smooth_window_size', help='the sliding window size to smooth the sampled rollout rewards', type=int, default=10)
    parser.add_argument('--env_name', help='the environment name of the raw data', type=str, default='scoop_water')
    parser.add_argument('--show_legend', help='if show the legend in the figure', action='store_true', default=False)
    parser.add_argument('--n_steps', help='number of steps to plot', type=int, default=1500000)
    

    args = parser.parse_args()
    plotter = Learning_Curve_Plotter(log_dir=args.log_dir,
                                     eval_freq=args.eval_freq,
                                     n_eval_episodes=args.n_eval_episodes,
                                     eval_smooth_window_size=args.eval_smooth_window_size,
                                     non_eval_sample_freq=args.non_eval_sample_freq,
                                     non_eval_smooth_window_size=args.non_eval_smooth_window_size,
                                     env_name=args.env_name,
                                     show_legend=args.show_legend,
                                     n_steps=args.n_steps)
    plotter.process_data()
    plotter.plot()