# h1
tensorboard   --logdir="legged_gym/logs/h1_41/Jun21_17-49-20_"   --port=6006

tensorboard   --logdir="logs/h1_41/Jun15_14-56-15_hidden_256_dof_cut_10"   --port=6007

tensorboard   --logdir="logs/h1_41/Jun15_11-34-46_hidden_256_dof_acc_2e-8_dof_pos_limits_.05"   --port=6008

# overview
python  legged_gym/legged_gym/scripts/train.py  --task=h1_41  --headless

# train
python  legged_gym/legged_gym/scripts/train.py  --task=h1_41  --headless  --max_iterations 1000 --sim_dev "cuda:0" --rl_device "cuda:1"  --run_name "hidden_256"

# resume
python  legged_gym/legged_gym/scripts/train.py  --task=h1_41  --headless --load_run="Jun17_17-20-54_lstm_256"  --checkpoint=-1  --resume  --sim_dev cuda:2  --rl_device cuda:3  --max_iterations=1000

# play
python  legged_gym/legged_gym/scripts/play.py  --task=h1_41  --load_run="Jun24_21-42-49_hid_256_noise_plane" --checkpoint=-1 --num_envs=4

python  legged_gym/legged_gym/scripts/play.py  --task=h1_41  --load_run="Jun15_22-21-22_no_lstm" --checkpoint=-1 --num_envs=4


# terrain_test
python  legged_gym/legged_gym/scripts/play.py  --task=h1_2  --load_run="Jun11_18-27-32_" --checkpoint=-1 --num_envs=4


#
        import pdb; pdb.set_trace()