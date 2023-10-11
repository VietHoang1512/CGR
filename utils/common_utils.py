import torch
import numpy as np
import argparse
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import dataset
from utils import logging_utils


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_y_s(g, n_spurious):
    y = g // n_spurious
    s = g % n_spurious
    return y, s


def update_dict(acc_groups, y, g, logits):
    preds = torch.argmax(logits, axis=1)
    correct_batch = preds == y
    g = g.cpu()
    for g_val in np.unique(g):
        mask = g == g_val
        n = mask.sum().item()
        corr = correct_batch[mask].sum().item()
        acc_groups[g_val].update(corr / n, n)


def get_results(acc_groups, get_ys_func, reweight_ratio=None):
    # TODO(izmailovpavel): add mean acc on train group distribution: DONE
    groups = acc_groups.keys()
    results = {
        f"accuracy_{get_ys_func(g)[0]}_{get_ys_func(g)[1]}": acc_groups[g].avg
        for g in groups
    }
    accs = np.array([acc_groups[g].avg for g in groups])
    all_correct = sum([acc_groups[g].sum for g in groups])
    all_total = sum([acc_groups[g].count for g in groups])
    results.update({"worst_accuracy": min(results.values())})
    results.update({"mean_accuracy": all_correct / all_total})
    if reweight_ratio is not None:
        results.update(
            {"avg_accuracy": (np.array(accs) * np.array(reweight_ratio)).sum()}
        )
    return results


def get_data(args):
    if args.data_transform == "None":
        transform_cls = lambda *args, **kwargs: None
    else:
        transform_cls = getattr(dataset, args.data_transform)
    train_transform = transform_cls(train=True)

    test_transform = transform_cls(train=False)

    dataset_cls = getattr(dataset, args.dataset)

    trainset = dataset_cls(
        basedir=args.data_dir, split="train", transform=train_transform
    )

    # dataset.remove_minority_groups(trainset, args.num_minority_groups_remove)

    test_data_dir = args.test_data_dir if args.test_data_dir else args.data_dir
    testset_dict = {
        split: dataset_cls(basedir=test_data_dir, split=split, transform=test_transform)
        for split in ["val", "test"]
    }

    # collate_fn = data.get_collate_fn(
    #     mixup=args.mixup, num_classes=trainset.n_classes)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 4,
        "pin_memory": True,
    }
    train_group_weights = len(trainset)/trainset.group_counts
    train_weights = train_group_weights[trainset.group_array]

    # Replacement needs to be set to True, otherwise we'll run out of minority samples
    train_loader = DataLoader(
        trainset, shuffle=False, sampler = WeightedRandomSampler(train_weights, len(trainset), replacement=True), **loader_kwargs  # collate_fn=collate_fn,
    )
    test_loader_dict = {
        name: DataLoader(ds, shuffle=False, **loader_kwargs)
        for name, ds in testset_dict.items()
    }

    get_ys_func = partial(get_y_s, n_spurious=testset_dict["test"].n_spurious)
    logging_utils.log_data(
        trainset,  testset_dict["val"], testset_dict["test"], get_ys_func=get_ys_func
    )

    return train_loader, test_loader_dict, get_ys_func
