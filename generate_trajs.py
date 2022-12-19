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
import pickle

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
    parser.add_argument('--seed', type=int, default=0)
    # Env args
    parser.add_argument('--loader_name', type=str, default='bowl', help='the type of the loader (bowl|bucket)')
    parser.add_argument('--water_amount_goal', type=float, default=0.60, help='The water amount goal')

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
    env_kwargs['water_amount_goal'] = args.water_amount_goal

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')

    set_random_seed(args.seed)
    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    model = SAC.load(path=args.load_file_dir, device=args.device, env=env)
    model.set_random_seed(args.seed)
    print('use seed: ', args.seed)

    saved_info = {}
    saved_info['loader_type'] = args.loader_name
    if args.loader_name == 'bucket':
        saved_info['bucket_front_length'] = 0.08775 * 2
    elif args.loader_name == 'bowl':
        saved_info['bowl_radius'] = 0.115487415
    else:
        pass
    saved_info['num_episodes'] = args.num_episodes
    saved_info['episode_length'] = 75
    saved_info['tested_water_amount_goal'] = args.water_amount_goal
    saved_info['tank_height'] = 0.3
    saved_info['tank_border_thickness'] = 0.03
    saved_info['tank_length'] = 0.9
    saved_info['tank_width'] = 0.9

    loader_poss = []
    loader_vels = []
    loader_rots = []
    loader_rot_vels = []
    in_loader_percents = []
    waterlines = []
    targeted_poss = []
    targeted_water_amounts = []
    rwds = []

    episode_reward_for_reg = []
    frames = [env.get_image(args.img_size, args.img_size)]
    for i in range(args.num_episodes):
        print(i)
        done = False
        episode_reward = 0
        obs = env.reset()
        t = 0
        loader_pos = []
        loader_vel = []
        loader_rot = []
        loader_rot_vel = []
        in_loader_percent = []
        waterline = []
        targeted_pos = []
        targeted_water_amount = []
        rwd = []

        while not done:
            t += 1
            loader_pos.append(obs['observation'][:3])
            loader_vel.append(obs['observation'][4:7])
            loader_rot.append([obs['observation'][3]])
            loader_rot_vel.append(obs['observation'][7])
            in_loader_percent.append([obs['observation'][8]])
            waterline.append([obs['observation'][9]])
            targeted_pos.append(obs['desired_goal'][:3])
            targeted_water_amount.append([obs['desired_goal'][3]])

            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
            episode_reward+= reward
            rwd.append(reward)
            frames.extend(info['flex_env_recorded_frames'])
            if done:
                episode_reward_for_reg.append(episode_reward)
                break
        loader_pos = np.stack(loader_pos)
        loader_vel = np.stack(loader_vel)
        loader_rot = np.stack(loader_rot)
        loader_rot_vel = np.stack(loader_rot_vel)
        in_loader_percent = np.stack(in_loader_percent)
        waterline = np.stack(waterline)
        targted_pos = np.stack(targeted_pos)
        targeted_water_amount = np.stack(targeted_water_amount)
        rwd = np.array(rwd).reshape(-1, 1)

        loader_poss.append(loader_pos)
        loader_vels.append(loader_vel)
        loader_rots.append(loader_rot)
        loader_rot_vels.append(loader_rot_vel)
        in_loader_percents.append(in_loader_percent)
        waterlines.append(waterline)
        targeted_poss.append(targeted_pos)
        targeted_water_amounts.append(targeted_water_amount)
        rwds.append(rwd)
    
    loader_poss = np.stack(loader_poss)
    loader_vels = np.stack(loader_vels)
    loader_rots = np.stack(loader_rots)
    loader_rot_vels = np.stack(loader_rot_vels)
    in_loader_percents = np.stack(in_loader_percents)
    waterlines = np.stack(waterlines)
    targeted_poss = np.stack(targeted_poss)
    targeted_water_amounts = np.stack(targeted_water_amounts)
    rwds = np.stack(rwds)

    saved_info['loader_pos_trajs'] = loader_poss
    saved_info['loader_vel_trajs'] = loader_vels
    saved_info['loader_rot_trajs'] = loader_rots
    saved_info['loader_rot_vel_trajs'] = loader_rot_vels
    saved_info['in_loader_percent_trajs'] = in_loader_percents
    saved_info['waterline_trajs'] = waterlines
    saved_info['targeted_pos_trajs'] = targeted_poss
    saved_info['targeted_water_amount_trajs'] = targeted_water_amounts
    saved_info['reward_trajs'] = rwds
    saved_info['episode_reward_trajs'] = np.array(episode_reward_for_reg)

    with open(f'{args.loader_name}_targeted_amount_{args.water_amount_goal}_saved_trajs.pkl', 'wb') as handle:
        pickle.dump(saved_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(f'{args.loader_name}_targeted_amount_{args.water_amount_goal}_saved_trajs.pkl', 'rb') as handle:
        data = pickle.load(handle)

    print('testing results:')
    print(episode_reward_for_reg)
    print(np.mean(episode_reward_for_reg))
    print(np.std(episode_reward_for_reg))

    if args.save_video:
        save_name = osp.join(args.save_video_dir, args.loader_name + '_' + str(args.water_amount_goal) + '.gif')
        save_numpy_as_gif(np.array(frames), save_name)
        print('Video generated and save to {}'.format(save_name))