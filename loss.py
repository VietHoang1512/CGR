import numpy as np
import torch
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


    def loss(self, yhat, y, group_idx=None):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(
            per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg(
            (torch.argmax(yhat, 1) == y).float(), group_idx)
        return group_loss


    def compute_grads(self, losses, model, args):
        if self.pref is not None:
            assert args.moo_method == "epo", "Preference can only be used with EPO"

        grads = []
        for loss_idx, loss in enumerate(losses):
            model.zero_grad()
            loss.backward(retain_graph=(loss_idx < (len(losses)-1)))
            grad = self.get_gradient(model)
            grads.append(grad)
        grads = torch.stack(grads)
        gradient_aggregate = gradient_aggregators[args.moo_method]
        grad = gradient_aggregate(grads, losses, self.pref)
        self.set_gradient(model, grad)
        return losses.mean()

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
