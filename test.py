import gym
import numpy as np
import argparse
import os.path as osp
import torch
from typing import Any, Callable, Dict, Optional, Type, Union

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from her_replay_buffer import HerReplayBuffer
from sac import SAC

import softgym
from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
import pyflex

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='DistributeWater')
    parser.add_argument('--headless', type=int, default=1, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--device', type=str, default='cpu', help='the type of device to use (cpu|cuda)')
    parser.add_argument('--load_file_dir', help='the path to model to test', type=str, default='save_dir/model.zip')
    parser.add_argument('--render', type=int, default=0, help='if render the tested environment')
    parser.add_argument('--num_episodes', help='number of episodes to test', type=int, default=20)
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--save_video', type=int, default=1, help='if save the video')
    parser.add_argument('--save_video_dir', type=str, default='save_dir', help='Path to the saved video')
    # Env args
    parser.add_argument('--loader_name', type=str, default='bowl', help='the type of the loader (bowl|bucket)')

    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = args.render
    env_kwargs['headless'] = args.headless
    env_kwargs['curr_mode'] = 0
    env_kwargs['eval'] = False
    env_kwargs['loader_name'] = args.loader_name

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')

    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    model = SAC.load(path=args.load_file_dir, device=args.device, env=env)
    seed = model.seed
    env.seed(seed)
    env.action_space.seed(seed)

    episode_reward_for_reg = []
    frames = [env.get_image(args.img_size, args.img_size)]
    for i in range(args.num_episodes):
        print(i)
        done = False
        episode_reward = 0
        obs = env.reset()
        t = 0
        while not done:
            t += 1
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
            episode_reward+= reward
            frames.extend(info['flex_env_recorded_frames'])
            if done:
                episode_reward_for_reg.append(episode_reward)
                break
    print('testing results:')
    print(episode_reward_for_reg)
    print(np.mean(episode_reward_for_reg))
    print(np.std(episode_reward_for_reg))

    if args.save_video:
        save_name = osp.join(args.save_video_dir, args.env_name + '.gif')
        save_numpy_as_gif(np.array(frames), save_name)
        print('Video generated and save to {}'.format(save_name))