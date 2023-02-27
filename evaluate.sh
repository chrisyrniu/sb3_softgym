export OMP_NUM_THREADS=1

python -u test.py \
  --env_name ScoopWater \
  --loader_name bucket \
  --water_amount_goal 0.70 \
  --multi_amount_goals 1 \
  --headless 1 \
  --num_variations 128 \
  --use_cached_states \
  --save_cached_states \
  --cached_states_path bucket_evaluation_128.pkl \
  --device cuda \
  --load_file_dir save_dir_new/amount_goal_exp/bucket_init045_area5575_s1645e5575_amount070m_s30e90_sac_her_curr_p2585a2070_seed0/sac_her_curr_seed0_best_model \
  --render 1 \
  --num_episodes 100 \
  --pos_goal_lower 0.55 \
  --pos_goal_upper 0.60 \
  --seed 0 \
  --save_video 0