#!/usr/bin/env python3
"""
vit with prompt: a clean version with the default settings of VPT
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv

from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
from scipy import ndimage

from .backbone import CONFIGS, Transformer, VisionTransformer, np2th



class PromptedTransformer(Transformer):
    def __init__(self, args, config, img_size, vis):

        super(PromptedTransformer, self).__init__(
            config, img_size, vis)
        
        self.vit_config = config
        
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])

        self.num_tokens = args.num_tokens  # number of prompted tokens

        DROPOUT = .1
        self.prompt_dropout = Dropout(DROPOUT)


        prompt_dim = config.hidden_size
        self.prompt_proj = nn.Identity()

        # initiate prompt:
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

        self.prompt_embeddings = nn.Parameter(torch.zeros(
            1, args.num_tokens, prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        

        total_d_layer = config.transformer["num_layers"]-1
        self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
            total_d_layer, self.num_tokens, prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                x[:, 1:, :]
            ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.encoder.eval()
            self.embeddings.eval()
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_deep_prompt(self, embedding_output):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]

        for i in range(num_layers):
            if i == 0:
                hidden_states, weights = self.encoder.layer[i](embedding_output)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))

                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompt_emb,
                        hidden_states[:, (1+self.num_tokens):, :]
                    ), dim=1)


                hidden_states, weights = self.encoder.layer[i](hidden_states)

            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward(self, x):
        # this is the default version:
        embedding_output = self.incorporate_prompt(x)

        encoded, attn_weights = self.forward_deep_prompt(
            embedding_output)

        return encoded, attn_weights


class PromptedVisionTransformer(VisionTransformer):
    def __init__(
        self, args, img_size=224, num_classes=21843, vis=False
    ):
        super(PromptedVisionTransformer, self).__init__(
            args.model, img_size, num_classes, vis)

        vit_cfg = CONFIGS[args.model]
        self.transformer = PromptedTransformer(args,vit_cfg, img_size, vis)

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)

        x = x[:, 0]

        logits = self.head(x)

        if not vis:
            return logits
        return logits, attn_weights

# Parameters to be updated: {('enc.transformer.prompt_embeddings', torch.Size([1, 10, 768])), ('enc.transformer.deep_prompt_embeddings', torch.Size([11, 10, 768])), ('head.last_layer.bias', torch.Size([2])), ('head.last_layer.weight', torch.Size([2, 768]))}
class PromptGenerator(nn.Module):
    def __init__(
        self,
        args, 
        config, 
        preference_dim=4,
        preference_embedding_dim=64,
    ):
        super().__init__()
            
        total_d_layer = config.transformer["num_layers"]-1
        prompt_dim = config.hidden_size

        self.preference_embedding_matrix = nn.Embedding(
            num_embeddings=preference_dim, embedding_dim=preference_embedding_dim
        )        
        
        self.layer_to_shape = {('enc.transformer.prompt_embeddings', torch.Size([1, args.num_tokens, prompt_dim])), ('enc.transformer.deep_prompt_embeddings', torch.Size([total_d_layer , args.num_tokens, prompt_dim])), ('head.last_layer.bias', torch.Size([2])), ('head.last_layer.weight', torch.Size([2, prompt_dim]))}
        
        for layer_name, shape in self.layer_to_shape:
            length = torch.prod(shape)
            layer = nn.Linear(preference_embedding_dim, length)
            # nn.init.xavier_uniform_(layer.weight)
            setattr(self, layer_name, layer)
            
    def forward(self, preference):
        # preference embedding
        pref_embedding = torch.zeros(
            (self.preference_embedding_dim,), device=preference.device
        )
        for i, pref in enumerate(preference):
            pref_embedding += (
                self.preference_embedding_matrix(
                    torch.tensor([i], device=preference.device)
                ).squeeze(0)
                * pref
            )       
        for layer_name, shape in self.layer_to_shape:
            layer = getattr(self, layer_name)
            pref_embedding = layer(pref_embedding).view(shape) 