#!/usr/bin/env python3
"""
Model construction functions.
"""
from tabnanny import verbose
import torch

import logging

from .vit_models import ViT

# Supported model types


def build_model(args, n_classes):
    """
    build model here
    """
    model = ViT(args, n_classes)

    log_model_info(model, verbose=False)
    model, device = load_model_to_device(model)
    logging.info(f"Device used for model: {device}")

    return model


def log_model_info(model, verbose=False):
    """Logs model info"""
    if verbose:
        logging.info(f"Classification Model:\n{model}")
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total Parameters: {0}\t Gradient Parameters: {1}".format(
        model_total_params, model_grad_params))
    logging.info("tuned percent:%.3f"%(model_grad_params/model_total_params*100))


def get_current_device():
    if torch.cuda.is_available():
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
    else:
        cur_device = torch.device('cpu')
    return cur_device


def load_model_to_device(model):
    cur_device = get_current_device()
    model = model.cuda(device=cur_device)
    return model, cur_device
