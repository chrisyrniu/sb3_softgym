export OMP_NUM_THREADS=1

python -u train_softgym.py \
  --env_name DistributeWater \
  --n_envs 6 \
  --headless 1 \
  --num_variations 1 \
  --training_steps 200000 \
  --learning_rate 0.0003 \
  --train_freq 1 \
  --grad_steps 2 \
  --buffer_size 1000000 \
  --batch_size 256 \
  --learning_starts 100 \
  --device cuda \
  --log_interval 100 \
  --log_dir log_dir \
  --log_name 6_envs_train_freq_1_grad_steps_2 \
  --save_dir save_dir \
  --seed 0 \
  | tee train_vec.log