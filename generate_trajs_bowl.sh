export OMP_NUM_THREADS=1

python -u generate_trajs.py \
  --env_name LoadWater \
  --loader_name bowl \
  --water_amount_goal 0.60 \
  --headless 1 \
  --num_variations 10 \
  --device cuda \
  --load_file_dir save_dir/bowl_multi_amount_goals/exp/bowl_sac_her_curr_multi_amount_goals_discrete_0.70_seed0/sac_her_curr_seed0_best_model \
  --render 1 \
  --num_episodes 5 \
  --save_video 1 \
  --save_video_dir save_dir/bowl_multi_amount_goals/exp/bowl_sac_her_curr_multi_amount_goals_discrete_0.70_seed0 \
  --seed 2 \
  | tee test.log