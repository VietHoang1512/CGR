import os
import logging
import numpy as np
import torch
import torch.nn as nn

from weight_methods import MGDA, BGDA, PCGrad, CAGrad, IMTL, EW, EPO

gradient_aggregators = {
    "mgda": MGDA,
    "pcgrad": PCGrad,
    "cagrad": CAGrad,
    "imtl": IMTL,
    "ew": EW,
    "epo": EPO,
    "bgda": BGDA
}


class LossComputer:
    def __init__(self, criterion,  loader, pref=None):
        self.criterion = criterion

        self.n_groups = loader.dataset.n_groups

        self.pref = pref
        self.similarity = nn.CosineSimilarity(dim=0, eps=1e-6)
        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).cuda()/self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).cuda()
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().cuda()
        self.conflict = torch.zeros(self.n_groups).cuda()
        
        self.group_losses = []
        self.group_accs = []
        self.group_conflicts = []
        
        self.reset_stats(init=True)

    def groupdro(self, yhat, y, group_idx, model):
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(
            per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg((torch.argmax(yhat,1)==y).float(), group_idx)

        self.update_exp_avg_loss(group_loss, group_count)
        actual_loss = self.compute_robust_loss(
            group_loss, group_count)

        grads = []
        for loss_idx, loss in enumerate(group_loss):
            model.zero_grad()
            loss.backward(retain_graph=True)
            grad = self.get_gradient(model)
            grads.append(grad)
        grads = torch.stack(grads)

        model.zero_grad()
        actual_loss.backward()
        final_grad = self.get_gradient(model)
        for idx, grad in enumerate(grads):
            similarity = self.similarity(grad, final_grad).item()
            self.conflict[idx] = (similarity < 0)
        self.set_gradient(model, final_grad)
        
        self.update_stats(group_loss, group_acc, group_count, self.conflict.clone())

    def compute_robust_loss(self, group_loss, group_count, step_size=0.01):
        adjusted_loss = group_loss
        self.adv_probs = self.adv_probs * torch.exp(step_size*adjusted_loss.data)
        self.adv_probs = self.adv_probs/(self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss

    def compute_grads(self,  yhat, y, group_idx, model, args):

        if self.pref is not None:
            assert args.moo_method == "epo", "Preference can only be used with EPO"

        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(
            per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg((torch.argmax(yhat,1)==y).float(), group_idx)
        

        grads = []
        for loss_idx, loss in enumerate(group_loss):
            model.zero_grad()
            loss.backward(retain_graph=True)
            grad = self.get_gradient(model)
            grads.append(grad)
        grads = torch.stack(grads)
        if args.moo_method=="erm":
            model.zero_grad()
            group_loss.sum().backward()
            final_grad = self.get_gradient(model)
        else:
            gradient_aggregate = gradient_aggregators[args.moo_method]
            final_grad = gradient_aggregate(grads, group_loss, self.pref)
        
        for idx, grad in enumerate(grads):
            similarity = self.similarity(grad, final_grad).item()
            self.conflict[idx] = (similarity < 0)
        self.set_gradient(model, final_grad)
        
        self.update_stats(group_loss, group_acc, group_count, self.conflict.clone())

    def get_gradient(self, model):
        grad = []
        for _, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad.append(param.grad.data.clone().flatten())
                else:
                    grad.append(torch.zeros_like(param.data).flatten())
        return torch.cat(grad)

    def set_gradient(self, model, grad):
        offset = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                numel = param.data.numel()
                param.grad = grad[offset:offset+numel].view(param.data.shape)
                offset += numel
        assert offset == grad.numel(
        ), f"Size mismatched {offset} != {grad.numel()}"

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(
            self.n_groups).unsqueeze(1).long().cuda()).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()  # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count, gamma=0.1):
        prev_weights = (1 - gamma*(group_count > 0).float()
                        ) * (self.exp_avg_initialized > 0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss*curr_weights
        self.exp_avg_initialized = (
            self.exp_avg_initialized > 0) + (group_count > 0)

    def reset_stats(self, init=False):

        if not init:
            self.group_losses.append(self.avg_group_loss.detach().clone().cpu())
            self.group_accs.append(self.avg_group_acc.detach().clone().cpu())
            self.group_conflicts.append(self.avg_group_conflict.detach().clone().cpu())

        self.processed_data_counts = torch.zeros(self.n_groups).cuda()
        self.avg_group_loss = torch.zeros(self.n_groups).cuda()
        self.avg_group_acc = torch.zeros(self.n_groups).cuda()
        self.avg_group_conflict = torch.zeros(self.n_groups).cuda()
        self.avg_per_sample_loss = 0.
        self.avg_acc = 0.
        self.batch_count = 0.

    def update_stats(self, group_loss, group_acc, group_count, conflict):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom == 0).float()
        prev_weight = self.processed_data_counts/denom
        curr_weight = group_count/denom
        
        self.avg_group_loss = prev_weight*self.avg_group_loss + curr_weight*group_loss
        self.avg_group_acc = prev_weight*self.avg_group_acc + curr_weight*group_acc
        self.avg_group_conflict = prev_weight*self.avg_group_conflict + curr_weight*conflict
        
        # batch-wise average actual loss
        denom = self.batch_count + 1


        # counts
        self.processed_data_counts += group_count

        self.batch_count += 1

        # avg per-sample quantities
        group_frac = self.processed_data_counts / \
            (self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc
        

        
    def log_stats(self):

        for group_idx in range(self.n_groups):
            logging.info(
                f'  {group_idx}  '
                f'[n = {int(self.processed_data_counts[group_idx])}]:\t'
                f'loss = {self.avg_group_loss[group_idx]:.3f}  '
                f'acc = {self.avg_group_acc[group_idx]:.3f}\n')

    def save_stats(self, output_dir):
        stats = {
            "group_losses": torch.stack(self.group_losses),
            "group_accs": torch.stack(self.group_accs),
            "group_conflicts": torch.stack(self.group_conflicts)
        }
        torch.save(stats, os.path.join(output_dir, "stats.pt"))