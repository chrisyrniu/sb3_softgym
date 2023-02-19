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
    method: str = 'random',
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

    def make_env(rank, eval, method, seed):
        def _init():
            env = normalize(SOFTGYM_ENVS[env_name](**env_kwargs))
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            if eval:
                monitor_path = os.path.join(monitor_dir, f"{method_name}_seed{seed}_eval_" + str(rank)) if monitor_dir is not None else None
            else:
                monitor_path = os.path.join(monitor_dir, f"{method_name}_seed{seed}_train_" + str(rank)) if monitor_dir is not None else None
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

    return vec_env_cls([make_env(i + start_index, eval, method, seed) for i in range(n_envs)], **vec_env_kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    # Env args
    parser.add_argument('--loader_name', type=str, default='bowl', help='the type of the loader (bowl|bucket)')
    parser.add_argument('--env_name', type=str, default='DistributeWater')
    parser.add_argument('--n_envs', help='the number of environments in parallel', type=int, default=1)
    parser.add_argument('--headless', type=int, default=1, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--cached_states_path', type=str, default='scoop_water_init_states.pkl', help='the path to save the cached states for softgym envs')
    # parser.add_argument('--achieved_amount_goal_zero_mask', action='store_true', default=False, 
    #                     help='When the the water amount goal is larger than 0, the reward will be -1 if the achieved amount is 0')
    parser.add_argument('--achieved_amount_zero_reward_coeff', type=float, default=1.0)
    # Train args
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
    parser.add_argument('--min_reward', help='minimum reward to save the model', type=float)
    parser.add_argument('--n_eval_episodes', help='the number of episodes for each evaluation during training', type=int, default=5)
    parser.add_argument('--eval_freq', help='evaluation frequence of the model', type=int, default=1500)
    # HER args
    parser.add_argument('--her', type=int, default=0, help='Whether to use hindsight experience replay')
    parser.add_argument('--max_episode_length_her', type=int, default=75, help="The maximum length of an episode (only required for using HER)")
    parser.add_argument('--goal_selection_strategy', type=str, default='future', help='Strategy for sampling goals for replay (future, final or episode)')
    parser.add_argument('--n_sampled_goal', type=int, default=4, help='Number of virtual transitions to create per real transition, by sampling new goals.')
    parser.add_argument('--online_sampling', type=bool, default=True, help='If new transitions will not be saved in the replay buffer and will only be created at sampling time')
    parser.add_argument('--water_amount_goal', type=float, default=0.60, help='The water amount goal')
    parser.add_argument('--multi_amount_goals', type=int, default=0, choices=[0, 1, 2], help='The type of setting multiple water amount goals (0 for none, 1 for discrete values, 2 for a continuous range)')
    # Curriculum Learning args
    parser.add_argument('--curr_mode', type=int, default=0, help='the curriculum learning mode to use (0: no curriculum, 1: base curriculum, 2: designed curriculum)')
    parser.add_argument('--curr_start', type=int, default=250000, help='the step to start the curriculum')
    parser.add_argument('--curr_end', type=int, default=650000, help='the step to end the curriculum')
    parser.add_argument('--virtual_water_amount_goal', type=float, default=0.0, help='the single virtual water amount goal for curriculum learning')
    # Env args
    parser.add_argument('--pos_goal_lower', type=float, default=0.55, help='the lower bound in height of the postion goal area')
    parser.add_argument('--pos_goal_upper', type=float, default=0.75, help='the upper bound in height of the postion goal area')
    parser.add_argument('--loader_init_height', type=float, default=0.45, help='the initial height of the loader')
    parser.add_argument('--pre_curr_pos_lower', type=float, default=0.16, help='the lower bound in height of the sampled position goal area before the curriculum starts')
    parser.add_argument('--pre_curr_pos_upper', type=float, default=0.45, help='the upper bound in height of the sampled position goal area before the curriculum starts')
    parser.add_argument('--post_curr_pos_lower', type=float, default=0.55, help='the lower bound in height of the sampled position goal area after the curriculum')
    parser.add_argument('--post_curr_pos_upper', type=float, default=0.75, help='the upper bound in height of the sampled position goal area after the curriculum')

    args = parser.parse_args()

    if args.her:
        if args.curr_mode == 0:
            method_name = "sac_her"
        elif args.curr_mode == 1:
            method_name = "sac_her_vanilla_curr"
        else:
            method_name = "sac_her_curr"
    else:
        if args.curr_mode == 0:
            method_name = "sac"
        elif args.curr_mode == 1:
            method_name = "sac_vanilla_curr"
        else:
            method_name = "sac_curr"

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
    env_kwargs['use_cached_states'] = True
    env_kwargs['save_cached_states'] = True
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['cached_states_path'] = args.cached_states_path
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless
    env_kwargs['curr_mode'] = args.curr_mode
    env_kwargs['eval'] = False
    env_kwargs['loader_name'] = args.loader_name
    env_kwargs['curr_start'] = args.curr_start
    env_kwargs['curr_end'] = args.curr_end
    env_kwargs['water_amount_goal'] = args.water_amount_goal
    env_kwargs['multi_amount_goals'] = args.multi_amount_goals
    env_kwargs['virtual_water_amount_goal'] = args.virtual_water_amount_goal
    env_kwargs['achieved_amount_zero_reward_coeff'] = args.achieved_amount_zero_reward_coeff
    env_kwargs['pos_goal_lower'] = args.pos_goal_lower
    env_kwargs['pos_goal_upper'] = args.pos_goal_upper
    env_kwargs['loader_init_height'] = args.loader_init_height
    env_kwargs['pre_curr_pos_lower'] = args.pre_curr_pos_lower
    env_kwargs['pre_curr_pos_upper'] = args.pre_curr_pos_upper
    env_kwargs['post_curr_pos_lower'] = args.post_curr_pos_lower
    env_kwargs['post_curr_pos_upper'] = args.post_curr_pos_upper

    

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations...')

    eval_env_kwargs = env_kwargs.copy()
    # eval_env_kwargs['use_cached_states'] = True
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
                        monitor_dir=monitor_file_path,
                        method=method_name)
    eval_env = make_vec_env(args.env_name,
                        n_envs=1,
                        eval=True,
                        seed=args.seed,
                        env_kwargs=eval_env_kwargs,
                        vec_env_cls=DummyVecEnv,
                        monitor_dir=eval_monitor_file_path,
                        method=method_name)
    callback = EvalCheckpointCallback(eval_env=eval_env, best_model_save_path=save_dir, n_eval_episodes=args.n_eval_episodes,
                                    eval_freq=args.eval_freq, minimum_reward=args.min_reward, method_name=method_name, seed=args.seed)

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