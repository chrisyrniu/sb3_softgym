export OMP_NUM_THREADS=1

python -u test.py \
  --env_name DistributeWater \
  --headless 0 \
  --num_variations 1 \
  --device cuda \
  --load_file_dir save_dir/run1/model.zip \
  --render 1 \
  --num_episodes 5 \
  --save_video 1 \
  --save_video_dir save_dir/run1 \
  | tee test.log