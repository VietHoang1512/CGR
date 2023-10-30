import os
import gc
from tqdm import tqdm
import logging
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import torch.nn.functional as F

import utils
from loss import LossComputer
from models.vit.schedulers import make_scheduler


def eval(model, test_loader_dict, return_auc=False, device="cuda"):
    model.eval()
    results_dict = {}
    with torch.no_grad():
        for test_name, test_loader in test_loader_dict.items():
            acc_groups = {
                g_idx: utils.AverageMeter() for g_idx in range(test_loader.dataset.n_groups)
            }
            probs, ys = [], []
            for batch in tqdm(test_loader):
                x, y, group, *_ = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)

                utils.update_dict(acc_groups, y, group, logits)
                if return_auc:
                    prob = F.softmax(logits, dim=-1)[:, 1].view(-1)
                    probs.append(prob.detach().cpu())
                    ys.append(y.detach().cpu())

            if return_auc:
                probs = np.concatenate(probs)
                ys = np.concatenate(ys)
                roc_auc = roc_auc_score(ys, probs)
                acc_groups["roc_auc"] = roc_auc
            results_dict[test_name] = acc_groups
    return results_dict


def run_epoch(epoch, model, loss_computer, optimizer, train_loader, test_loader_dict, get_ys_func, args, log_every=50, scheduler=None, best_val_wga=0, best_test_wga=0):

    model.train()



    prog_bar_loader = tqdm(train_loader)

    n_groups = train_loader.dataset.n_groups
    train_acc_groups = {g_idx: utils.AverageMeter()
                        for g_idx in range(n_groups)}

    train_group_ratio = train_loader.dataset.group_ratio
    for batch_idx, batch in enumerate(prog_bar_loader):

        batch = tuple(t.cuda() for t in batch)
        x = batch[0]
        y = batch[1]
        g = batch[2]

        outputs = model(x)
        utils.update_dict(train_acc_groups, y, g, outputs)

        optimizer.zero_grad()
        if args.moo_method=="groupdro":
            loss_computer.groupdro(outputs, y, g, model)

        else:
            loss_computer.compute_grads(outputs, y, g, model, args)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        if (batch_idx+1) == len(prog_bar_loader) or (batch_idx+1) % log_every == 0:

            loss_computer.log_stats()
            loss_computer.reset_stats()

            results_dict = eval(model, test_loader_dict,
                                return_auc="isic" in args.dataset.lower())

            for ds_name, acc_groups in results_dict.items():
                utils.log_test_results(
                    epoch, acc_groups, get_ys_func, ds_name, train_group_ratio)

            val_wga = utils.get_results(
                results_dict["val"], get_ys_func, train_group_ratio
            )["worst_accuracy"]

            test_wga = utils.get_results(
                results_dict["test"], get_ys_func, train_group_ratio
            )["worst_accuracy"]

            if val_wga > best_val_wga:
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_val.pt"))
                logging.info(f"New best validation WGA: {val_wga}")
                best_val_wga = val_wga

            if test_wga > best_test_wga:
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_test.pt"))
                logging.info(f"New best test WGA: {test_wga}")
                best_test_wga = test_wga

    if scheduler:
        scheduler.step()
        
    
    return train_acc_groups, best_val_wga, best_test_wga


def train(model, criterion, train_loader, test_loader_dict, get_ys_func, args):

    model = model.cuda()
    utils.log_model_info(model)

    if args.preference is not None:
        args.preference = torch.from_numpy(
            np.array(args.preference, dtype=np.float32)).cuda()
        args.preference /= args.preference.sum()
    logging.info(f'Preference: {args.preference}\n')

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)

    # optimizer = torch.optim.Adam(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=args.lr,
    #     weight_decay=args.weight_decay)

    if args.scheduler:
        args.scheduler = make_scheduler(
            optimizer, "cosine", args.warmup, args.n_epochs
        )
    logging.info(f"Using {args.scheduler} scheduler\n")

    # Double check
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add((name, param.shape))
            print(name, param.shape, param.requires_grad)

    logging.info(f"Parameters to be updated: {enabled}\n")

    best_val_wga = 0
    best_test_wga = 0
    
    loss_computer = LossComputer(
        criterion,
        train_loader,
        pref=args.preference)

    for epoch in range(args.n_epochs):
        logging.info(f"Epoch: [{epoch+1}/{args.n_epochs}]")

        train_acc_groups, val_wga, test_wga = run_epoch(epoch, model, loss_computer, optimizer, train_loader, test_loader_dict, get_ys_func,
                                                        args, log_every=args.log_every, scheduler=args.scheduler, best_val_wga=best_val_wga, best_test_wga=best_test_wga)
        utils.log_test_results(epoch, train_acc_groups, get_ys_func,  "train")

        if val_wga > best_val_wga:
            best_val_wga = val_wga

        if test_wga > best_test_wga:
            best_test_wga = test_wga

        for param_group in optimizer.param_groups:
            curr_lr = param_group['lr']
            logging.info('Current lr: %f\n' % curr_lr)
            
        loss_computer.save_stats(args.output_dir)

        # if args.scheduler and args.model != 'bert':
        #     val_loss = val_loss_computer.avg_group_loss
            # scheduler step to update lr at the end of epoch
            # scheduler.step(val_loss)

        # if epoch % args.save_step == 0 and epoch > 0:
        #     # torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))
        #     torch.save(model.state_dict(), os.path.join(args.log_dir, "%d_model_weights.pt" % epoch))

        # if args.save_last:
        #     # torch.save(model, os.path.join(args.log_dir, 'last_model.pth'))
        #     torch.save(model.state_dict(), os.path.join(args.log_dir, "last_model_weights.pt"))

