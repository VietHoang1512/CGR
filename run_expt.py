import os
import argparse
import torch
import logging
from train import train
import utils
import models


def get_parser():
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,
        default="./dataset/waterbird/waterbird_complete95_forest2water2",
        help="Train dataset directory",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default=None,
        required=False,
        help="Test data directory (default: <data_dir>)",
    )
    parser.add_argument(
        "--data_transform",
        type=str,
        required=False,
        default="AugWaterbirdsCelebATransform",
        choices=[
            "None",
            "AugDominoTransform",
            "NoAugDominoTransform",
            "SimCLRDominoTransform",
            "MaskedDominoTransform",
            "AugWaterbirdsCelebATransform",
            "SimCLRWaterbirdsCelebATransform",
            "NoAugWaterbirdsCelebATransform",
            "NoAugNoNormWaterbirdsCelebATransform",
            "MaskedWaterbirdsCelebATransform",
            "ImageNetAugmixTransform",
            "ImageNetRandomErasingTransform",
            "SimCLRCifarTransform",
            "AlbertTokenizeTransform",
            "BertTokenizeTransform",
            "BertMultilingualTokenizeTransform",
            "DebertaTokenizeTransform",
            "WaterbirdsForCLIPTransform",
        ],
        help="Data preprocessing transformation",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="SpuriousCorrelationDataset",
        choices=[
            "SpuriousCorrelationDataset",
            "MultiNLIDataset",
            "FakeSpuriousCIFAR10",
            "WildsFMOW",
            "WildsCivilCommentsCoarse",
            "WildsCivilCommentsCoarseNM",
            "DeBERTaMultiNLIDataset",
            "BERTMultilingualMultiNLIDataset",
        ],
        help="Dataset type",
    ) 


    # Model
    parser.add_argument(
        '--model',
        default='resnet50')
    parser.add_argument('--train_from_scratch',
                        action='store_true', default=False)

    parser.add_argument(
        "--moo_method",
        type=str,
        choices=[
            "mgda",
            "pcgrad",
            "cagrad",
            "imtl",
            "ew",
            "epo"
        ],
        default="imtl",
        help="MTL weight method",
    )
    parser.add_argument('--num_tokens', type=int, default=5)

    # Optimization
    parser.add_argument('--n_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--warmup', default=5, type=int)

    # EPO:

    
    parser.add_argument("--preference", nargs='+',  default=None, help="preference vector")

    # parser.add_argument('--gamma', type=float, default=0.1)
    # parser.add_argument('--minimum_variational_weight', type=float, default=0)
    # Misc
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--show_progress', default=False, action='store_true')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int, default=10)
    parser.add_argument('--save_best', action='store_true', default=False)
    parser.add_argument('--save_last', action='store_true', default=False)


    return parser


def main(args):


    # BERT-specific configs copied over from run_glue.py

    args.max_grad_norm = 1.0

    writer = utils.prepare_logging(args)

    utils.set_seed(args.seed)

    # Data
    train_loader, test_loader_dict, get_ys_func = utils.get_data(args)
    n_classes = train_loader.dataset.n_classes

    # log_data(data, logger)
    if "vit" in args.model.lower():
        logging.info("Using ViT")
        from models.vit.build_model import build_model
        model = build_model(args, n_classes)

    else:
        logging.info("Using DFR model")
        model_cls = getattr(models, args.model)
        model = model_cls(n_classes)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    train(model, criterion, train_loader, test_loader_dict, get_ys_func, args)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
