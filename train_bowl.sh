export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

python -u train.py \
  --env_name ScoopWater \
  --loader_name bowl \
  --loader_init_height 0.45 \
  --achieved_amount_zero_reward_coeff 1.0 \
  --her 1 \
  --curr_mode 2 \
  --curr_start 250000 \
  --curr_end 650000 \
  --water_amount_goal 0.70 \
  --multi_amount_goals 1 \
  --n_envs 1 \
  --headless 1 \
  --num_variations 1000 \
  --cached_states_path bowl_scoop_water_init_states.pkl \
  --training_steps 1500000 \
  --learning_rate 0.0003 \
  --train_freq 1 \
  --grad_steps 1 \
  --buffer_size 1000000 \
  --batch_size 512 \
  --learning_starts 1000 \
  --device cuda \
  --max_episode_length_her 75 \
  --goal_selection_strategy future \
  --n_sampled_goal 4 \
  --online_sampling True \
  --pos_goal_lower 0.40 \
  --pos_goal_upper 0.60 \
  --pre_curr_pos_lower 0.01 \
  --pre_curr_pos_upper 0.30 \
  --post_curr_pos_lower 0.40 \
  --post_curr_pos_upper 0.60 \
  --amount_curr_start 250000 \
  --amount_curr_end 450000 \
  --pre_curr_prob 0.51 \
  --post_curr_prob 0.85 \
  --log_interval 20 \
  --log_dir log_dir_new/amount_goal_exp \
  --log_name bowl_init045_area4060_s0130e4060_amount070m_s51e85_sac_her_curr_p2565a2545_seed0 \
  --min_reward -31 \
  --n_eval_episodes 10 \
  --eval_freq 3000 \
  --save_dir save_dir_new/amount_goal_exp \
  --seed 0

