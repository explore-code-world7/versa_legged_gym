
python  legged_gym/legged_gym/scripts/train.py  --task=h1_45_hf  --headless  --max_iterations 2000 --sim_dev  cuda:2 --rl_device  cuda:3


python  legged_gym/legged_gym/scripts/train.py  --task=h1_45_stair  --headless  --max_iterations 1000 --sim_dev  cuda:0 --rl_device  cuda:1  --load_run="Jul06_11-15-26_"  --checkpoint=-1  --resume
