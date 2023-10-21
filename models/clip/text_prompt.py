import torch
import torch.nn as nn

from . import clip


def template_prompt(args):
    if "waterbird" in args.data_dir:
        return "a type of bird, a photo of a"
    raise NotImplementedError(
        "template prompt not implemented for this dataset" + args.data_dir)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(
            dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, args, clip_model, metadata_map):
        super().__init__()

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.n_ctx = args.num_tokens

        template = template_prompt(args)
        template_n_ctx = len(template.split(" "))
        assert self.n_ctx >= template_n_ctx, f"#tokens ({self.n_ctx}) should larger equal than #initial prompt tokens ({template_n_ctx}, {template})"

        template_tokens = clip.tokenize(template)
        with torch.no_grad():
            template_embedding = clip_model.token_embedding(
                template_tokens).type(dtype)

        ctx_vectors = torch.empty(args.num_tokens, ctx_dim, dtype=dtype)
        # nn.init.normal_(ctx_vectors, std=0.02) 

        print("ctx vectors size: ", ctx_vectors.shape)

        ctx_vectors[self.n_ctx - template_n_ctx:,
                    :] = template_embedding[0, 1:1 + template_n_ctx, :]

        prompt_prefix = " ".join(["X"] * (args.num_tokens-template_n_ctx))
        prompt_prefix = f"{prompt_prefix} {template}"

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = metadata_map["target"]

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {self.n_ctx}")

        self.n_cls = len(classnames)

        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])


        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(
                dtype)
            print("embedding size: ", embedding.shape)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:,
                                                       1 + self.n_ctx:, :])  # CLS, EOS

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor


    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        # neb = self.original_embedding
        # prompts = torch.cat([prompts, neb], dim=0)
        return prompts


class TextPrompt(nn.Module):
    def __init__(self, args, metadata_map):
        super().__init__()
        clip_model, _ = clip.load(args.model, device="cpu")
        clip_model.float()
        self.prompt_learner = PromptLearner(args, clip_model, metadata_map)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        for k, p in self.named_parameters():
            if "prompt_learner" not in k:
                p.requires_grad = False

    # @autocast()
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        text_features = text_features / text_features.norm(dim=-1,
                                                           keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
