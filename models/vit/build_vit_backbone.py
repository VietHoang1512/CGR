#!/usr/bin/env python3
import numpy as np
import torch
import os
from .prompt import VisionTransformer, PromptedVisionTransformer





def build_vit_models(
    crop_size, args=None, model_root=None, load_pretrain=True, vis=False
):
    # TODO: Update this dictationary
    # image size is the size of actual image
    m2featdim = {
        "vitb16_224": 768,
        "vitb16": 768,
        "vitl16_224": 1024,
        "vitl16": 1024,
        "vitb8_imagenet21k": 768,
        "imagenet21k_ViT-B_16": 768,
        "vitb32_imagenet21k": 768,
        "vitl16_imagenet21k": 1024,
        "vitl32_imagenet21k": 1024,
        "vith14_imagenet21k": 1280,
    }
    model = PromptedVisionTransformer(args,
        crop_size, num_classes=-1, vis=vis
    )

    
    if load_pretrain:
        model.load_from(np.load(os.path.join(model_root, args.model+".npz")))

    return model, m2featdim[args.model]

