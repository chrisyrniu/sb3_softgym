import numpy as np
import argparse
import os.path as osp

from stable_baselines3.common.utils import set_random_seed
from sac import SAC

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
import pickle
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--env_name', type=str, default='ScoopWaterRobot2D')
    parser.add_argument('--headless', type=int, default=1, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--device', type=str, default='cuda', help='the type of device to use (cpu|cuda)')
    parser.add_argument('--load_file_dir', help='the path to model to test', type=str, default='save_dir/model.zip')
    parser.add_argument('--render', type=int, default=0, help='if render the tested environment')
    parser.add_argument('--num_episodes', help='number of episodes for each evaluation setting', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--save_video', type=int, default=1, help='if save the video')
    parser.add_argument('--save_dir', type=str, default='save_dir', help='Path to save the trajectories and videos')
    parser.add_argument('--seed', type=int, default=0)
    # Env args
    parser.add_argument('--loader_name', type=str, default='bucket', help='the type of the loader (bowl|bucket)')
    parser.add_argument('--water_amount_goal', type=float, default=0.60, help='The water amount goal')
    parser.add_argument('--position_goal', type=int, default=0, choices=[0, 1, 2, 3], help='0 means randomly sample positions from the workspace, 1,2,3 will give three preset position goals')
    parser.add_argument('--init_waterline', type=int, default=0, choices=[0, 1, 2, 3], help='0 means randomly sample initial waterlines, 1,2,3 will give three preset initial waterlines')

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = args.render
    env_kwargs['headless'] = args.headless
    env_kwargs['curr_mode'] = 0
    env_kwargs['eval'] = True
    env_kwargs['loader_name'] = args.loader_name
    env_kwargs['water_amount_goal'] = args.water_amount_goal
    env_kwargs['eval_position_goal'] = args.position_goal
    env_kwargs['eval_init_waterline'] = args.init_waterline

    set_random_seed(args.seed)
    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    model = SAC.load(path=args.load_file_dir, device=args.device, env=env)
    model.set_random_seed(args.seed)

    saved_info = {}
    saved_info['loader_type'] = args.loader_name
    scale = 1
    if args.loader_name == 'bucket':
        saved_info['bucket_front_length'] = 0.08775 * 2
        scale = 117 / saved_info['bucket_front_length']
        saved_info['bucket_front_length'] = 117
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
    frames = []
    for i in range(args.num_episodes):
        done = False
        episode_reward = 0
        obs = env.reset()
        frames.append(env.get_image(args.img_size, args.img_size))
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
    
    env._wrapped_env.close()
    del env

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

    scaled_keys = ['tank_height', 'tank_border_thickness', 'tank_length', 
                'tank_width', 'loader_pos_trajs', 'loader_vel_trajs', 
                'waterline_trajs', 'targeted_pos_trajs']
    for key in scaled_keys:
        saved_info[key] = saved_info[key] * scale
    
    print('---------')
    print('amount goal:', args.water_amount_goal, ',', 'position_goal:', args.position_goal, ',', 'init waterline', args.init_waterline)
    mean_water_amounts = np.mean(saved_info['in_loader_percent_trajs'][:, -20:, 0], axis=1)
    print('average water amounts from last 20 steps:', mean_water_amounts)
    error = np.round(np.mean(mean_water_amounts - args.water_amount_goal), 5)
    print('average error torwards the goal:', error)

    save_name = osp.join(args.save_dir, f'{args.loader_name}_amount_goal_{args.water_amount_goal}_pos_goal_{args.position_goal}_waterline_{args.init_waterline}_seed_{args.seed}_error_{error}.pkl')
    with open(save_name, 'wb') as handle:
        pickle.dump(saved_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if args.save_video:
        save_name = osp.join(args.save_dir, f'{args.loader_name}_amount_goal_{args.water_amount_goal}_pos_goal_{args.position_goal}_waterline_{args.init_waterline}_seed_{args.seed}_error_{error}.gif')
        save_numpy_as_gif(np.array(frames), save_name)
        # print('Video generated and save to {}'.format(save_name))