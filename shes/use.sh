# h1
tensorboard   --logdir="legged_gym/logs/h1_41/Jun25_22-04-53_hidden_256_hf_encoder"   --port=6006

tensorboard   --logdir="legged_gym/logs/h1_41/Jun24_21-42-49_hid_256_noise_plane"   --port=6007

tensorboard   --logdir="logs/h1_41/Jun15_11-34-46_hidden_256_dof_acc_2e-8_dof_pos_limits_.05"   --port=6008


# past with wirsts and ankles may collides
python  legged_gym/legged_gym/scripts/train.py  --task=h1_51_hf  --headless  --max_iterations 2000 --sim_dev  cuda:2 --rl_device  cuda:3

python  legged_gym/legged_gym/scripts/train.py  --task=h1_51_hf  --headless --load_run="Jul05_11-47-50_"  --checkpoint=-1  --resume  --sim_dev cuda:2  --rl_device cuda:3  --max_iterations=1000

python  legged_gym/legged_gym/scripts/train.py  --task=h1_51_stair  --headless  --max_iterations 1000 --sim_dev  cuda:0 --rl_device  cuda:1  --load_run="Jul04_22-38-47_"  --checkpoint=-1  --resume



# overview
python  legged_gym/legged_gym/scripts/train.py  --task=h1_41   --num_envs 16

python  legged_gym/legged_gym/scripts/train.py  --task=go2_field  --headless  --max_iterations 2000 --sim_dev "cuda:6" --rl_device "cuda:7"

python  legged_gym/legged_gym/scripts/train.py  --task=h1_41_hf  --headless  --max_iterations 2000 --sim_dev  cuda:0 --rl_device  cuda:1

python  legged_gym/legged_gym/scripts/train.py  --task=h1_41_stair  --headless  --max_iterations 2000 --sim_dev "cuda:4" --rl_device "cuda:5"

# train
python  legged_gym/legged_gym/scripts/train.py  --task=h1_41_stair  --headless  --max_iterations 2000 --sim_dev "cuda:4" --rl_device "cuda:5"  --run_name "hidden_256_plane_height_field"


# resume
python  legged_gym/legged_gym/scripts/train.py  --task=h1_41_stair  --headless --load_run="Jul03_16-24-29_"  --checkpoint=-1  --resume  --sim_dev cuda:2  --rl_device cuda:3  --max_iterations=1000


# play
python  legged_gym/legged_gym/scripts/play.py  --task=h1_41_hf  --load_run="Jul05_14-15-31_" --checkpoint=-1 --num_envs=4

python  legged_gym/legged_gym/scripts/play.py  --task=h1_41_stair  --load_run="Jun26_17-04-29_hidden_256_plane_hf_measure" --checkpoint=-1 --num_envs=4



# terrain_test
python  legged_gym/legged_gym/scripts/play.py  --task=h1_2  --load_run="Jun11_18-27-32_" --checkpoint=-1 --num_envs=4

# import pdb; pdb.set_trace()
