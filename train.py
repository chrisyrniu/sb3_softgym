import gym
import argparse
import os
# import os.path as osp
import torch
from typing import Any, Callable, Dict, Optional, Type, Union
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor
)
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
from stable_baselines3.common.utils import set_random_seed
from eval_checkpoint_callback import EvalCheckpointCallback


def make_vec_env(
    env_name: Union[str,Any],
    n_envs: int = 1,
    seed: Optional[int] = None,
    eval: Optional[bool] = False,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None):

    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    wrapper_kwargs = {} if wrapper_kwargs is None else wrapper_kwargs

    set_random_seed(seed)

    if eval:
        n_envs = 1

    def make_env(rank, eval):
        def _init():
            env = normalize(SOFTGYM_ENVS[env_name](**env_kwargs))
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            if eval:
                monitor_path = os.path.join(monitor_dir, "eval_" + str(rank)) if monitor_dir is not None else None
            else:
                monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index, eval) for i in range(n_envs)], **vec_env_kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='DistributeWater')
    parser.add_argument('--n_envs', help='the number of environments in parallel', type=int, default=1)
    parser.add_argument('--headless', type=int, default=1, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--training_steps', type=int, default=50000, help='Number of total timesteps of training')
    parser.add_argument('--train_freq', type=int, default=1, help='Update (call the train function) the model every train_freq steps (train_freq*n if do multi-processing)')
    parser.add_argument('--grad_steps', type=int, default=1, help='How many gradient steps to do for each train function call (after each rollout and it can only rollout one step at a time)')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--buffer_size', type=int, default=1000000, help='buffer size')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--learning_starts', type=int, default=100, help='how many steps of the model to collect transitions for before learning starts')
    parser.add_argument('--ent_coef', type=str, default='auto', help='entropy regularization coefficient')
    parser.add_argument('--device', type=str, default='cpu', help='the type of device to use (cpu|cuda)')
    parser.add_argument('--log_interval', help='the number of episodes before logging', type=int, default=100)
    parser.add_argument('--log_dir', help='the path to the log files', type=str, default='log_dir')
    parser.add_argument('--log_name', help='the name of the run for TensorBoard logging', type=str, default='SAC')
    parser.add_argument('--save_dir', help='the path to save models', type=str, default='save_dir')
    parser.add_argument('--seed', help='the seed number to use', type=int, default=0)
    # Evaluate args
    parser.add_argument('--min_reward', help='minimum reward to save the model', type=int)
    parser.add_argument('--n_eval_episodes', help='the number of episodes for each evaluation during training', type=int, default=5)
    parser.add_argument('--eval_freq', help='evaluation frequence of the model', type=int, default=1500)
    # HER args
    parser.add_argument('--her', type=int, default=0, help='Whether to use hindsight experience replay')
    parser.add_argument('--max_episode_length_her', type=int, default=75, help="The maximum length of an episode (only required for using HER)")
    parser.add_argument('--goal_selection_strategy', type=str, default='future', help='Strategy for sampling goals for replay (future, final or episode)')
    parser.add_argument('--n_sampled_goal', type=int, default=4, help='Number of virtual transitions to create per real transition, by sampling new goals.')
    parser.add_argument('--online_sampling', type=bool, default=True, help='If new transitions will not be saved in the replay buffer and will only be created at sampling time')
    # LoadWaterGoal args
    parser.add_argument('--curr_start_step', type=int, default=0, help='the training step to start curriculum learning')
    parser.add_argument('--curr_end_step', type=int, default=0, help='the training step to end curriculum learning')
    parser.add_argument('--curr_start_thresh', type=float, default=0.4, help='the (water amount) threshold to start curriculum learning')
    parser.add_argument('--curr_end_thresh', type=float, default=0.8, help='the (water amount) threshold to end curriculum learning')
    # LoadWaterAmount args
    parser.add_argument('--goal_sampling_mode', type=int, default=0, help='the mode for sampling the targeted amount of water')

    args = parser.parse_args()

    log_dir = args.log_dir + '/' + args.log_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_dir = args.save_dir + '/' + args.log_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    monitor_file_path = log_dir
    eval_monitor_file_path = log_dir

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')

    if args.env_name == "LoadWater":
        env_kwargs['goal_sampling_mode'] = args.goal_sampling_mode

    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['eval'] = True

    if args.n_envs == 1:
        args.vec_env_cls = DummyVecEnv
    else:
        args.vec_env_cls = SubprocVecEnv
    env = make_vec_env(args.env_name, 
                        n_envs=args.n_envs, 
                        seed=args.seed, 
                        env_kwargs=env_kwargs, 
                        vec_env_cls=args.vec_env_cls,
                        monitor_dir=monitor_file_path)
    eval_env = make_vec_env(args.env_name,
                        n_envs=1,
                        eval=True,
                        seed=args.seed,
                        env_kwargs=eval_env_kwargs,
                        vec_env_cls=DummyVecEnv,
                        monitor_dir=eval_monitor_file_path)
    callback = EvalCheckpointCallback(eval_env=eval_env, best_model_save_path=save_dir, n_eval_episodes=args.n_eval_episodes,
                                    eval_freq=args.eval_freq, minimum_reward=args.min_reward)
    # eval_env = make_vec_env(args.env_name, n_envs=1, seed=args.seed, env_kwargs=env_kwargs, vec_env_cls=DummyVecEnv)
    # eval_callback = EvalCallback(eval_env, best_model_save_path=args.save_dir,
    #                          log_path=args.log_dir, eval_freq=40,
    #                          deterministic=True, render=False)
    # callback = CheckpointCallback(save_freq=10000, save_path=save_dir,
    #                                      name_prefix='her_model')

    if args.her:
        args.policy = "MultiInputPolicy"
        args.replay_buffer = HerReplayBuffer
        args.replay_buffer_kwargs = dict(
            max_episode_length=args.max_episode_length_her,
            goal_selection_strategy=args.goal_selection_strategy,
            n_sampled_goal=args.n_sampled_goal,
            online_sampling=args.online_sampling,
        )
    else:
        args.policy = "MultiInputPolicy"
        args.replay_buffer = None
        args.replay_buffer_kwargs = {}

    model = SAC(args.policy, 
                env,
                learning_rate = args.learning_rate,
                train_freq=args.train_freq,
                gradient_steps=args.grad_steps,
                buffer_size=args.buffer_size,
                replay_buffer_class=args.replay_buffer,
                replay_buffer_kwargs = args.replay_buffer_kwargs,
                batch_size=args.batch_size,
                learning_starts=args.learning_starts,
                ent_coef=args.ent_coef,
                device=args.device,
                tensorboard_log=log_dir,
                verbose=1,
                seed=args.seed)
    model.learn(total_timesteps=args.training_steps, log_interval=args.log_interval, tb_log_name=args.log_name, callback=callback)
    model.save(path=save_dir+'model')