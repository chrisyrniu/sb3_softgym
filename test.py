import gym
import numpy as np
import argparse
import os.path as osp
import os
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
    parser.add_argument('--use_cached_states', action='store_true', default=False)
    parser.add_argument('--save_cached_states', action='store_true', default=False)
    parser.add_argument('--cached_states_path', type=str, default='scoop_water_init_states.pkl', help='the path to save the cached states for softgym envs')
    parser.add_argument('--device', type=str, default='cpu', help='the type of device to use (cpu|cuda)')
    parser.add_argument('--load_file_dir', help='the path to model to test', type=str, default='save_dir/model.zip')
    parser.add_argument('--render', type=int, default=0, help='if render the tested environment')
    parser.add_argument('--num_episodes', help='number of episodes to test', type=int, default=20)
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--save_video', type=int, default=1, help='if save the video')
    parser.add_argument('--save_video_dir', type=str, default='save_dir', help='Path to the saved video')
    parser.add_argument('--seed', type=int, default=0)    
    # Env args
    parser.add_argument('--loader_name', type=str, default='bowl', help='the type of the loader (bowl|bucket)')
    parser.add_argument('--water_amount_goal', type=float, default=0.60, help='The water amount goal')
    parser.add_argument('--multi_amount_goals', type=int, default=0, choices=[0, 1, 2], help='The type of setting multiple water amount goals (0 for none, 1 for discrete values, 2 for a continuous range)')
    parser.add_argument('--pos_goal_lower', type=float, default=0.55, help='the lower bound in height of the postion goal area')
    parser.add_argument('--pos_goal_upper', type=float, default=0.75, help='the upper bound in height of the postion goal area')
    parser.add_argument('--loader_init_height', type=float, default=0.45, help='the initial height of the loader')
    parser.add_argument('--acc_limit_scale', type=float, default=1.0)
    parser.add_argument('--vel_limit_scale', type=float, default=1.0)


    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = args.use_cached_states
    env_kwargs['save_cached_states'] = args.save_cached_states
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['cached_states_path'] = args.cached_states_path
    env_kwargs['render'] = args.render
    env_kwargs['headless'] = args.headless
    env_kwargs['curr_mode'] = 0
    env_kwargs['eval'] = True
    env_kwargs['loader_name'] = args.loader_name
    env_kwargs['water_amount_goal'] = args.water_amount_goal
    env_kwargs['multi_amount_goals'] = args.multi_amount_goals
    env_kwargs['pos_goal_lower'] = args.pos_goal_lower
    env_kwargs['pos_goal_upper'] = args.pos_goal_upper
    env_kwargs['loader_init_height'] = args.loader_init_height
    env_kwargs['acc_limit_scale'] = args.acc_limit_scale
    env_kwargs['vel_limit_scale'] = args.vel_limit_scale

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')

    set_random_seed(args.seed)
    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    model = SAC.load(path=args.load_file_dir, device=args.device, env=env)
    model.set_random_seed(args.seed)

    episode_reward_for_reg = []
    water_amount_goals = []
    achieved_goals = []

    frames = []
    for i in range(args.num_episodes):
        print("episode", i)
        done = False
        episode_reward = 0
        obs = env.reset()
        if args.save_video:
            frames.append(env.get_image(args.img_size, args.img_size))
        t = 0
        in_loader_percent = []
        while not done:
            t += 1
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action, record_continuous_video=args.save_video, img_size=args.img_size)
            in_loader_percent.append([obs['observation'][8]])
            episode_reward+= reward
            if args.save_video:
                frames.extend(info['flex_env_recorded_frames'])
            if done:
                episode_reward_for_reg.append(episode_reward)
                water_amount_goals.append(info['amount_goal'])
                break
        achieved_goals.append(in_loader_percent)
    water_amount_goals = np.array(water_amount_goals)
    achieved_goals = np.array(achieved_goals)
    achieved_goals = np.mean(achieved_goals[:, -10:, 0], axis=1)
    error = np.mean(np.abs(water_amount_goals - achieved_goals))
    print('testing results')
    print('episode rewards', episode_reward_for_reg)
    print('episode reward mean', np.mean(episode_reward_for_reg))
    print('episode reward std', np.std(episode_reward_for_reg))
    print('water amount goals:', water_amount_goals)
    print('average water amounts from last ten steps:', achieved_goals)
    print('average error:', error)

    if args.save_video:
        if not os.path.exists(args.save_video_dir):
            os.makedirs(args.save_video_dir)
        save_name = osp.join(args.save_video_dir, args.env_name + '.gif')
        save_numpy_as_gif(np.array(frames), save_name)
        print('Video generated and save to {}'.format(save_name))