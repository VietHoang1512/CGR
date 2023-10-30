import os
import numpy as np
import argparse
import torch
import logging
import cv2
from tqdm.auto import tqdm
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from PIL import Image
import utils
import copy

def get_parser():
    parser = argparse.ArgumentParser()

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
        help="Data preprocessing transformation",
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="MetaShiftDataset",
        choices=[
            "SpuriousCorrelationDataset",
            "MultiNLIDataset",
            "FakeSpuriousCIFAR10",
            "WildsFMOW",
            "WildsCivilCommentsCoarse",
            "WildsCivilCommentsCoarseNM",
            "DeBERTaMultiNLIDataset",
            "BERTMultilingualMultiNLIDataset",
            "MetaShiftDataset",
            "ISICDataset"
        ],
        help="Dataset type",
    ) 
    parser.add_argument('--alpha', type=float, default=.0,  help="Reweight group power")
    parser.add_argument('--batch_size', type=int, default=32)
    
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="./outputs/gradcam/",
        help="Output directory",
    )

    # Model
    parser.add_argument(
        '--model',
        default='resnet50')

    parser.add_argument(
        "--erm_ckpt",
        type=str,
        required=True,
        help="Checkpoint to load",
    )
    parser.add_argument(
        "--groupdro_ckpt",
        type=str,
        required=True,
        help="Checkpoint to load",
    )
    parser.add_argument(
        "--our_ckpt",
        type=str,
        required=True,
        help="Checkpoint to load",
    )

    parser.add_argument('--num_tokens', type=int, default=10)
    parser.add_argument('--multi_prompt', action='store_true', default=False)

    # Misc
    parser.add_argument('--seed', type=int, default=0)

    return parser


def main(args):

    utils.set_seed(args.seed)

    # writer = utils.prepare_logging(args)

    n_classes = 2
    
    target_resolution = (224, 224)
    resize_resolution = (256, 256)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
        
    if args.model.lower().startswith("clip-"):
        from models.clip.text_prompt import TextPrompt

        args.model = args.model[5:]

        # TODO: add support for other datasets
        metadata_map = {
            # Padding for str formatting
            "generic-spurious": ["bird on land", "bird on water"],
            "spurious": ["land background", "water background"],
            "target": ["landbird", "waterbird"],
        }
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        if "336" in args.model:
            target_resolution = (336, 336)
            resize_resolution = target_resolution
        model = TextPrompt(args, metadata_map)

    elif "vit" in args.model.lower():
        logging.info("Using ViT")
        from models.vit.build_model import build_model
        model = build_model(args, n_classes)

    else:
        import models
        logging.info("Using DFR model")
        model_cls = getattr(models, args.model)
        model = model_cls(n_classes)

    erm_model = copy.deepcopy(model)
    erm_model.load_state_dict(torch.load(args.erm_ckpt))
    erm_model.eval()
    groupdro_model = copy.deepcopy(model)
    groupdro_model.load_state_dict(torch.load(args.groupdro_ckpt))
    groupdro_model.eval()
    our_model = copy.deepcopy(model)
    our_model.load_state_dict(torch.load(args.our_ckpt))
    our_model.eval()

     
    train_loader, test_loader_dict, _ = utils.get_data(args)

    
    train_dataset = train_loader.dataset
    val_dataset = test_loader_dict["val"].dataset
    test_dataset = test_loader_dict["test"].dataset
    dataset_dict = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    transform = transforms.Compose([
                transforms.Resize(resize_resolution),
                transforms.CenterCrop(target_resolution),
            ])

    for dataset_name, dataset in dataset_dict.items():
    
        output_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        for idx in tqdm(range(len(dataset))):
            img, y, g, img_filepath = dataset.get_datum(idx, transform=transform)
            
            targets = [ClassifierOutputTarget(y)]
            
            input_tensor = preprocess_image(img, mean=mean, std=std)
            img = np.float32(img) / 255

            # ERM
            erm_target_layers = [erm_model.layer4]
            with GradCAM(model=erm_model, target_layers=erm_target_layers, use_cuda=True) as cam:
                grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
                erm_cam_image = show_cam_on_image(
                    img, grayscale_cams[0, :], use_rgb=True)

            groupdro_target_layers = [groupdro_model.layer4]
            with GradCAM(model=groupdro_model, target_layers=groupdro_target_layers, use_cuda=True) as cam:
                grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
                groupdro_cam_image = show_cam_on_image(
                    img, grayscale_cams[0, :], use_rgb=True)

            our_target_layers = [our_model.layer4]
            with GradCAM(model=our_model, target_layers=our_target_layers, use_cuda=True) as cam:
                grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
                our_cam_image = show_cam_on_image(
                    img, grayscale_cams[0, :], use_rgb=True)

            images = np.hstack((np.uint8(255*img), erm_cam_image, groupdro_cam_image, our_cam_image))
            images_out = Image.fromarray(images)
            # save images
            img_filename = img_filepath.split(
                "/")[-1].replace(".jpg", "_gradcam.jpg")
            images_out.save(os.path.join(output_dir, img_filename))

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)