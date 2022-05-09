import gym
import numpy as np
import copy
import argparse
import random
import os
import torch

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import HER, SAC

import softgym
print(softgym.__file__)
from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
import pyflex

# needs revision
def make_env(env_name, env_kwargs, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = normalize(SOFTGYM_ENVS[env_name](**env_kwargs))
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='DistributeWater')
    parser.add_argument('--headless', type=int, default=1, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--training_steps', type=int, default=50000, help='Number of total timesteps of training')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--buffer_size', type=int, default=1000000, help='buffer size')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--learning_starts', type=int, default=100, help='how many steps of the model to collect transitions for before learning starts')
    parser.add_argument('--device', type=str, default='cpu', help='the type of device to use (cpu|cuda)')
    parser.add_argument('--log_interval', help='the number of episodes before logging', type=int, default=100)
    parser.add_argument('--log_dir', help='the path to the log files', type=str, default='log_dir')
    parser.add_argument('--tb_log_name', help='the name of the run for TensorBoard logging', type=str, default='SAC')

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')

    # needs updating with vec_env
    env = make_env(args.env_name, env_kwargs, 0)
    env = DummyVecEnv([env])
    model = SAC("MlpPolicy", 
                env,
                learning_rate = args.learning_rate,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                learning_starts=args.learning_starts,
                device=args.device,
                tensorboard_log=args.log_dir,
                verbose=1)
    model.learn(total_timesteps=args.training_steps, log_interval=args.log_interval, tb_log_name=args.tb_log_name)