export OMP_NUM_THREADS=1

python -u train_softgym.py \
  --env_name LoadWaterGoal \
  --n_envs 1 \
  --headless 1 \
  --num_variations 1 \
  --her 1 \
  --training_steps 1000000 \
  --learning_rate 0.0003 \
  --train_freq 1 \
  --grad_steps 1 \
  --buffer_size 1000000 \
  --batch_size 256 \
  --learning_starts 100 \
  --curr_start_step 150000 \
  --curr_end_step 650000 \
  --curr_start_thresh 0.4 \
  --curr_end_thresh 0.95 \
  --device cuda \
  --max_episode_length_her 75 \
  --goal_selection_strategy future \
  --n_sampled_goal 4 \
  --online_sampling True \
  --log_interval 20 \
  --log_dir log_dir \
  --log_name curr_learning_try \
  --save_dir save_dir \
  --seed 0 \
  | tee train_her.log