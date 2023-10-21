import os
import copy
import argparse
import torch

import models
from models.vit.build_model import build_model


parser = argparse.ArgumentParser()

parser.add_argument(
    '--model',
    default='imagenet21k_ViT-B_16')

parser.add_argument('--num_tokens', type=int, default=5)

args = parser.parse_args()

n_classes=2

if "vit" in args.model.lower():
    model = build_model(args, n_classes)
else:
    model_cls = getattr(models, args.model)
    model = model_cls(n_classes)
    
output_dir = "test/" + str(args).replace(", ", "/").replace("'", "").replace(
    "(", ""
).replace(")", "").replace("Namespace", "")

os.makedirs(output_dir, exist_ok=True)

if "vit" in args.model.lower():
    state_dict = model.get_state_dict()
else:
    state_dict = model.state_dict()
print(state_dict.keys())    

torch.save({i:copy.deepcopy(state_dict) for i in range(4)}, output_dir + "/model.pt")
