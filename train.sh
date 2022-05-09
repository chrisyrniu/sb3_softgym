export OMP_NUM_THREADS=1

python -u train_softgym.py \
  --env_name DistributeWater \
  --headless 1 \
  --num_variations 1 \
  --training_steps 50000 \
  --learning_rate 0.0003 \
  --buffer_size 1000000 \
  --batch_size 256 \
  --learning_starts 100 \
  --device cuda \
  --log_interval 100 \
  --log_dir log_dir \
  | tee train.log