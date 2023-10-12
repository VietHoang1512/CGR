import os
import logging
import sys
import json
import yaml
import torchvision
from torch.utils.tensorboard import SummaryWriter

import utils


def write_dict_to_tb(writer, dict, prefix, step):
    if prefix[-1] != "/":
        prefix += "/"
    for key, value in dict.items():
        writer.add_scalar(f"{prefix}{key}", value, step)


def prepare_logging(args):
    args.output_dir = "outputs/"
    
    for k, v in vars(args).items():
        print(k, "=", v)
        if "/" not in str(v):
            args.output_dir += f"/{k}={v}"
    
    os.system("rm -rf " + args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        filename=f"{args.output_dir}/train.log",
        level=logging.DEBUG,
        filemode="w",
        datefmt="%H:%M:%S",
        format="%(asctime)s :: %(levelname)-8s \n%(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("Output directory:" + args.output_dir)

    with open(os.path.join(args.output_dir, "config.yaml"), "w") as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)



    with open(os.path.join(args.output_dir, "command.sh"), "w") as f:
        f.write(" ".join(sys.argv))
        f.write("\n")

    writer = SummaryWriter(log_dir=args.output_dir)

    with open(os.path.join(args.output_dir, "config.yaml"), "w") as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)

    return writer


def log_after_epoch(
    writer, epoch, loss_meter, acc_groups, get_ys_func, tag, images=None
):
    logging.info(f"Epoch {epoch}\t Loss: {loss_meter.avg}")
    results = utils.get_results(acc_groups, get_ys_func)
    logging.info(f"\ntrain result:")
    logging.info(str(results))

    write_dict_to_tb(writer, results, tag, epoch)

    if images is not None:
        images = images[:4]
        images_concat = torchvision.utils.make_grid(
            images, nrow=2, padding=2, pad_value=1.0
        )
        writer.add_image("data/", images_concat, epoch)




def log_test_results(epoch, acc_groups, get_ys_func, tag,  reweight_ratio=None):
    results = utils.get_results(acc_groups, get_ys_func, reweight_ratio)
    logging.info(f"\n{tag} result:")
    logging.info(str(results))



def log_data(train_data, val_data=None, test_data=None, get_ys_func=None):
    for data, name in [(train_data, "Train"), (val_data, "Val"), (test_data, "Test")]:
        if data:
            logging.info(f"{name} Data (total {len(data)})\n")
            print("N groups ", data.n_groups)
            for group_idx in range(data.n_groups):
                y_idx, s_idx = get_ys_func(group_idx)
                logging.info(
                    f"    Group {group_idx} (y={y_idx}, s={s_idx}):"
                    f" n = {data.group_counts[group_idx]:.0f}\n"
                )


def log_optimizer(writer, optimizer, epoch):
    group = optimizer.param_groups[0]
    hypers = {name: group[name] for name in ["lr", "weight_decay"]}
    write_dict_to_tb(writer, hypers, "optimizer_hypers", epoch)

def log_model_info(model, verbose=False):
    """Logs model info"""
    if verbose:
        logging.info(f"Classification Model:\n{model}")
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total Parameters: {0}\t Gradient Parameters: {1}".format(
        model_total_params, model_grad_params))
    logging.info("Tuned percent:%.3f"%(model_grad_params/model_total_params*100))