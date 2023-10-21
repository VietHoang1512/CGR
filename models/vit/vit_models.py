#!/usr/bin/env python3

"""
ViT-related models
Note: models return logits instead of prob
"""
import torch
import torch.nn as nn

from collections import OrderedDict
from torchvision import models

from .build_vit_backbone import  build_vit_models
from .mlp import MLP
import logging


ROOT="./"
class ViT(nn.Module):
    """ViT-related model."""

    def __init__(self, args, n_classes, load_pretrain=True, vis=False):
        super(ViT, self).__init__()

        self.froze_enc = False
        

        self.build_backbone(args, load_pretrain, vis=vis)
        self.setup_side()
        self.setup_head(n_classes)

    def setup_side(self):
        self.side = None

    def build_backbone(self, args, load_pretrain, vis):
        self.enc, self.feat_dim = build_vit_models(
        224, args, ROOT, load_pretrain, vis
        )

        for k, p in self.enc.named_parameters():
            if "prompt" not in k:
                p.requires_grad = False



    def setup_head(self, n_classes):
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[n_classes], # noqa
            special_bias=True
        )

    def forward(self, x, return_feature=False):


        x = self.enc(x)  # batch_size x self.feat_dim

        if return_feature:
            return x, x
        x = self.head(x)

        return x
    
    def forward_cls_layerwise(self, x):
        cls_embeds = self.enc.forward_cls_layerwise(x)
        return cls_embeds

    def get_features(self, x):
        """get a (batch_size, self.feat_dim) feature"""
        x = self.enc(x)  # batch_size x self.feat_dim
        return x

    def get_state_dict(self):
        state_dict = {k:v for k, v in self.state_dict().items() if "prompt" in k or k.startswith("head")}
        return state_dict