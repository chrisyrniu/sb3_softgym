export OMP_NUM_THREADS=1

python -u test.py \
  --env_name LoadWater \
  --headless 1 \
  --num_variations 10 \
  --device cuda \
  --load_file_dir save_dir/bs256/sac_her_curr_seed0/sac_her_curr_seed0_best_model \
  --render 1 \
  --num_episodes 10 \
  --save_video 1 \
  --save_video_dir save_dir/bs256/sac_her_curr_seed0 \
  | tee test.log