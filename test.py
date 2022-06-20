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


# def make_vec_env(
    # env_name: Union[str,Any],
    # n_envs: int = 1,
    # seed: Optional[int] = None,
    # start_index: int = 0,
    # monitor_dir: Optional[str] = None,
    # wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    # env_kwargs: Optional[Dict[str, Any]] = None,
    # vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    # vec_env_kwargs: Optional[Dict[str, Any]] = None,
    # monitor_kwargs: Optional[Dict[str, Any]] = None,
    # wrapper_kwargs: Optional[Dict[str, Any]] = None):

    # env_kwargs = {} if env_kwargs is None else env_kwargs
    # vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    # monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    # wrapper_kwargs = {} if wrapper_kwargs is None else wrapper_kwargs

    # def make_env(rank):
    #     def _init():
    #         env = normalize(SOFTGYM_ENVS[env_name](**env_kwargs))
    #         if seed is not None:
    #             env.seed(seed + rank)
    #             env.action_space.seed(seed + rank)
    #         # Wrap the env in a Monitor wrapper
    #         # to have additional training information
    #         monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
    #         # Create the monitor folder if needed
    #         if monitor_path is not None:
    #             os.makedirs(monitor_dir, exist_ok=True)
    #         env = Monitor(env, filename=monitor_path, **monitor_kwargs)
    #         # Optionally, wrap the environment with the provided wrapper
    #         if wrapper_class is not None:
    #             env = wrapper_class(env, **wrapper_kwargs)
    #         return env

    #     return _init

    # # No custom VecEnv is passed
    # if vec_env_cls is None:
    #     # Default: use a DummyVecEnv
    #     vec_env_cls = DummyVecEnv

    # return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)


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
    # LoadWaterGoal args
    parser.add_argument('--curr_start_step', type=int, default=0)
    parser.add_argument('--curr_end_step', type=int, default=0)
    parser.add_argument('--curr_start_thresh', type=float, default=0.4)
    parser.add_argument('--curr_end_thresh', type=float, default=0.8)
    # LoadWaterAmount args
    parser.add_argument('--goal_sampling_mode', type=int, default=0, help='the mode for sampling the targeted amount of water')

    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = args.render
    env_kwargs['headless'] = args.headless

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')

    if args.env_name == "LoadWaterGoal" or args.env_name == "LoadWaterGoalHard":
        env_kwargs['curr_start_step'] = args.curr_start_step
        env_kwargs['curr_end_step'] = args.curr_end_step
        env_kwargs['curr_start_thresh'] = args.curr_start_thresh
        env_kwargs['curr_end_thresh'] = args.curr_end_thresh
    if args.env_name == "LoadWaterAmount" or args.env_name == "LoadWaterAmountHard":
        env_kwargs['goal_sampling_mode'] = args.goal_sampling_mode

    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    model = SAC.load(path=args.load_file_dir, device=args.device, env=env)
    seed = model.seed
    env.seed(seed)
    env.action_space.seed(seed)

    episode_reward_for_reg = []
    frames = [env.get_image(args.img_size, args.img_size)]
    for i in range(args.num_episodes):
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