import os
import gc
import torch


import numpy as np
from tqdm import tqdm

from loss import LossComputer
from utils import log_to_tb
from pytorch_transformers import AdamW, WarmupLinearSchedule

from models.vit.schedulers import make_scheduler

def run_epoch(epoch, model, criterion, optimizer, loader, loader_name, loss_computer, logger, writer, csv_logger, args,
              is_training, tag, show_progress=False, log_every=50, scheduler=None):
    """
    scheduler is only used inside this function if model is bert.
    """

    if is_training:
        model.train()
        if args.model == 'bert':
            model.zero_grad()
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader[loader_name])
    else:
        prog_bar_loader = loader[loader_name]

    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):

            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]

            if args.model == 'bert':
                input_ids = x[:, :, 0]
                input_masks = x[:, :, 1]
                segment_ids = x[:, :, 2]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=y
                )[1] # [1] returns logits
            else:
                outputs = model(x)

            loss_main = loss_computer.loss(outputs, y, g, is_training)

            if is_training:
                if args.model == 'bert':
                    model.zero_grad()
                    # loss_main.backward()
                    loss_main = loss_computer.compute_grads(loss_main, model, args)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scheduler.step()
                    optimizer.step()
                    model.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss_main = loss_computer.compute_grads(loss_main, model, args)
                    
                    # loss_main.backward()
                    optimizer.step()
                    model.zero_grad()
                    if scheduler:
                        scheduler.step()
                    

            if is_training and (batch_idx+1) % log_every==0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                log_to_tb(writer, epoch, loss_computer.get_stats(model, args), loss_computer.n_groups, tag="train")
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()
                
                logger.write(f'\nValidation:\n')
                val_loss_computer = LossComputer(
                    criterion,
                    dataset=loader['val_data'],
                    )
                run_epoch(
                    epoch, model, criterion, optimizer,
                    loader, 'val_loader',
                    val_loss_computer,
                    logger, writer, csv_logger, args,
                    is_training=False, tag="val")
                
                # Test set; don't print to avoid peeking
                if loader['test_data'] is not None:
                    logger.write(f'\nTest:\n')
                    test_loss_computer = LossComputer(
                        criterion,
                        dataset=loader['test_data'],
                        )

                    run_epoch(
                        epoch, model, criterion, optimizer,
                        loader, 'test_loader',
                        test_loss_computer,
                        logger, writer, csv_logger, args,
                        is_training=False, tag="test")                

        if (not is_training) or loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            log_to_tb(writer, epoch, loss_computer.get_stats(model, args), loss_computer.n_groups, tag=tag)
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()


def train(model, criterion, dataset, 
          logger, writer, train_csv_logger, val_csv_logger, test_csv_logger,
          args, epoch_offset):


    model = model.cuda()

    train_loss_computer = LossComputer(
        criterion,
        dataset=dataset['train_data'])

    # BERT uses its own scheduler and optimizer
    if 'bert' in args.model:
        print("BERT model")
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            eps=args.adam_epsilon)
        t_total = len(dataset['train_loader']) * args.n_epochs
        print(f'\nt_total is {t_total}\n')
        scheduler = WarmupLinearSchedule(
            optimizer,
            warmup_steps=args.warmup_steps,
            t_total=t_total)
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay)
        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                'min',
                factor=0.1,
                patience=5,
                threshold=0.0001,
                min_lr=0,
                eps=1e-08)
        elif 'vit' in args.model.lower():
            scheduler = make_scheduler(
                optimizer
            )
        else:
            scheduler = None

    # Double check
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    logger.write(f"Parameters to be updated: {enabled}")

    best_val_acc = 0
    for epoch in range(epoch_offset, epoch_offset+args.n_epochs):
        logger.write('\nEpoch [%d]:\n' % epoch)
        logger.write(f'Training:\n')
        run_epoch(
            epoch, model, criterion, optimizer,
            dataset, 'train_loader',
            train_loss_computer,
            logger, writer, train_csv_logger, args,
            is_training=True, tag="train",
            show_progress=True,
            log_every=args.log_every,
            scheduler=scheduler)

        logger.write(f'\nValidation:\n')
        val_loss_computer = LossComputer(
            criterion,
            dataset=dataset['val_data'],
            )
        run_epoch(
            epoch, model, criterion, optimizer,
            dataset, 'val_loader',
            val_loss_computer,
            logger, writer, val_csv_logger, args,
            is_training=False, tag="val")
        
        # Test set; don't print to avoid peeking
        if dataset['test_data'] is not None:
            logger.write(f'\nTest:\n')
            test_loss_computer = LossComputer(
                criterion,
                dataset=dataset['test_data'],
                )

            run_epoch(
                epoch, model, criterion, optimizer,
                dataset, 'test_loader',
                test_loss_computer,
                logger, writer, test_csv_logger, args,
                is_training=False, tag="test")

        # Inspect learning rates
        if (epoch+1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
                logger.write('Current lr: %f\n' % curr_lr)

        if args.scheduler and args.model != 'bert':
            val_loss = val_loss_computer.avg_group_loss
            scheduler.step(val_loss) # scheduler step to update lr at the end of epoch

        # if epoch % args.save_step == 0 and epoch > 0:
        #     # torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))
        #     torch.save(model.state_dict(), os.path.join(args.log_dir, "%d_model_weights.pt" % epoch))

        # if args.save_last:
        #     # torch.save(model, os.path.join(args.log_dir, 'last_model.pth'))
        #     torch.save(model.state_dict(), os.path.join(args.log_dir, "last_model_weights.pt"))

        if args.save_best:
            if args.robust or args.reweight_groups:
                curr_val_acc = min(val_loss_computer.avg_group_acc)
            else:
                curr_val_acc = val_loss_computer.avg_acc
            logger.write(f'Current validation accuracy: {curr_val_acc}\n')
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                # torch.save(model, os.path.join(args.log_dir, 'best_model.pth'))
                torch.save(model.state_dict(), os.path.join(args.log_dir, "best_model_weights.pt"))
                np.save(os.path.join(args.log_dir, "best_epoch_num.npy"), [epoch])
                logger.write(f'Best model saved at epoch {epoch}\n')


        logger.write('\n')

        gc.collect()
