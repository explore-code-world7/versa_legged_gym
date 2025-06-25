import torch

model_path = "legged_gym/logs/h1_41/Jun25_22-04-53_hidden_256_hf_encoder/model_1500.pt"

model = torch.load(model_path, map_location="cuda:0")
model_state_dict = model["model_state_dict"]
optimizer_state_dict = model["optimizer_state_dict"]
iter = model["iter"]
infos = model["infos"]
import pdb; pdb.set_trace()
print(model)