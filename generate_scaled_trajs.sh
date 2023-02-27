export OMP_NUM_THREADS=1

for water_amount_goal in 0.60 0.65 0.70 0.75 0.80
do
  for init_waterline in 1 2 3
  do
    for position_goal in 1 2 3
    do
      python -u generate_scaled_trajs.py \
        --env_name ScoopWaterRobot2D \
        --loader_name bucket \
        --headless 1 \
        --num_variations 1 \
        --device cuda \
        --load_file_dir save_dir_robot_fast/robot_bucket_init035_area3050_s1630e1650_amount070m_s51e85_sac_her_curr_p2590a2555_buffs200_seed0/sac_her_curr_seed0_best_model \
        --render 1 \
        --num_episodes 1 \
        --save_video 1 \
        --save_dir scaled_trajs_035 \
        --water_amount_goal ${water_amount_goal} \
        --position_goal ${position_goal} \
        --init_waterline ${init_waterline} \
        --loader_init_height 0.35 \
        --seed 0
    done
  done
done