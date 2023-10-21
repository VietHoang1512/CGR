#!/usr/bin/env python3
"""
Model construction functions.
"""
from tabnanny import verbose
import torch

from .vit_models import ViT

# Supported model types


def build_model(args, n_classes):
    """
    build model here
    """
    model = ViT(args, n_classes)
    model, device = load_model_to_device(model)
    return model


def get_current_device():
    if torch.cuda.is_available():
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
    else:
        cur_device = torch.device('cpu')
    return cur_device


def load_model_to_device(model):
    cur_device = get_current_device()
    model = model.to(cur_device)
    return model, cur_device
