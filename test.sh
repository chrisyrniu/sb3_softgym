export OMP_NUM_THREADS=1

python -u test.py \
  --env_name DistributeWater \
  --headless 1 \
  --num_variations 1 \
  --device cuda \
  --load_file_dir save_dir/run2/model.zip \
  --render 1 \
  --num_episodes 5 \
  --save_video 1 \
  --save_video_dir save_dir/run2 \
  | tee test.log