export OMP_NUM_THREADS=1

python -u test.py \
  --env_name ScoopWater \
  --loader_name bowl \
  --water_amount_goal 0.70 \
  --multi_amount_goals 1 \
  --headless 1 \
  --num_variations 128 \
  --use_cached_states \
  --save_cached_states \
  --cached_states_path bowl_evaluation_128.pkl \
  --device cuda \
  --load_file_dir save_dir_new/bowl/multi_goal/bowl_sac_her_vanilla_curr_multi_amount_goals_discrete_0.70_seed0/sac_her_vanilla_curr_seed0_best_model \
  --render 1 \
  --num_episodes 100 \
  --pos_goal_lower 0.50 \
  --pos_goal_upper 0.55 \
  --seed 0 \
  --save_video 0 \
  --save_video_dir videos/