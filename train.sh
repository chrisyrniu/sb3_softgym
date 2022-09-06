export OMP_NUM_THREADS=1

python -u train.py \
  --env_name LoadWater \
  --her 1 \
  --curr_mode 2 \
  --n_envs 1 \
  --headless 1 \
  --num_variations 1000 \
  --training_steps 1000000 \
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
  --log_interval 20 \
  --log_dir log_dir \
  --log_name sac_her_curr_seed0 \
  --min_reward -40 \
  --n_eval_episodes 10 \
  --eval_freq 3000 \
  --save_dir save_dir \
  --seed 0 \
  | tee train.log