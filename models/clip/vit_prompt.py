import copy
from tqdm.auto import tqdm
import torch
import torch.nn as nn

import clip


# ImageNet prompt templates used by CLIP
imagenet_templates = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]


def classname_to_prompt(classname):
    prompt = "a photo of " + "a" if classname[0] in "aeiou" else "an" + " {}."
    return prompt.format(classname)


class ViTPrompt(nn.Module):
    def __init__(self, args, metadata_map):
        super(ViTPrompt, self).__init__()

        clip_model, _ = clip.load(args.model, device="cpu")
        self.clip_model = clip_model
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.prompt = nn.Linear(args.embd_dim, args.prompt_length, bias=False)

        self.zeroshot_weights = dict()
        pbar = tqdm(metadata_map.items())
        for k, v in pbar:
            self.zeroshot_weights[k] = self.get_zeroshot_weights(
                v, args.multi_prompt)
            pbar.set_description(
                f"Loaded zero shot weight for {k}, shape: {self.zeroshot_weights[k].shape}")

    @property
    def feature_dim(self):
        return self.image_encoder.output_dim

    def to(self, device):
        self = super().to(device)
        self.zeroshot_weights = {k: v.to(device)
                                 for k, v in self.zeroshot_weights.items()}
        return self

    # def extract_vector(self, image):
    #     image_features = self.image_encoder(image.type(self.dtype))
    #     image_features = image_features / \
    #         image_features.norm(dim=-1, keepdim=True)

    #     return image_features

    def forward(self, image, prompt=True, key="target", return_features=False):
        if prompt:
            image_features = self.image_encoder(
                image.type(self.dtype), self.prompt.weight)
        else:
            image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ self.zeroshot_weights[key]
        if return_features:
            return logits, image_features
        return logits

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def get_zeroshot_weights(self, texts, multi_prompt=False, avg=True):
        """
        Zero-shot classifier for pre-trained CLIP models.
        """
        zeroshot_weights = []
        for text in texts:
            if multi_prompt:
                # format with class
                prompts = [template.format(text)
                           for template in self.templates]
            else:
                prompts = [classname_to_prompt(text)]
            prompts = clip.tokenize(prompts)  # tokenize
            class_embeddings = self.clip_model.encode_text(
                prompts
            )  # embed with text encoder
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            if not avg:
                zeroshot_weights.append(class_embeddings)
            else:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
        return zeroshot_weights.detach().clone()


if __name__ == "__main__":
    net = ViTPrompt()
