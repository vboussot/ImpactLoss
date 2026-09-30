"""Export DINOv2 ViT-S/14 as IMPACT TorchScript models that take a 2D slice of any size.

    python Dinov2.py        ->  DinoV2_Small.pt (in the current directory), Dino/DinoV2_Small.pt on the Hub

Weights and code: facebookresearch/dinov2 (Apache-2.0) through torch.hub, the code pinned at DINOV2_COMMIT.

forward(x [B, 3, H, W], nb_layers [1], stats [4] = [min, max, mean, std], direction) -> List[Tensor [B, C, h, w]]

It returns the PATCH tokens of the whole slice, the position embedding interpolated to its grid: the hub's
get_intermediate_layers(x, n=[2, 5, 8, 11], reshape=True, norm=True), i.e. blocks 3, 6, 9, 12 through the final
LayerNorm, the first nb_layers of them (the blocks past the deepest one asked for are not run). The feature DINO-Reg
registers with (x_norm_patchtokens, there of ViT-L/14 reg4). Every token attends to the whole slice, so the model
has no bounded field of view (fov null in models.json).

As the model on the Hub does: x is scaled to [0, 1] with stats' min and max (x's own when stats is
not 4 values) and standardised with the ImageNet mean and std; direction is ignored. Then each axis is resampled
(bilinear) to the nearest multiple of 14, 14 at least, and left untouched when it already is one: the h x w patch
grid then spans exactly the input's extent, which is where IMPACT puts a feature map smaller than its input
(spacing = extent / size, itk-impact's AllocateFeatureImage; KonfAI interpolates it over the tile). Padding or
cropping instead would shift the features by up to 13 pixels at the far edge. Features come back on the patch
grid, as VGG16's and SAM 2.1's deeper layers come back on their strided grids.

The blocks are traced in evaluation mode and the wrapper is scripted, so the grid, the position embedding and
nb_layers follow the input at run time and the forward keeps its default arguments (the image alone works). The
position embedding is interpolated as the hub's interpolate_pos_encoding does it (the 0.1 offset of ViT-S, the
native 37 x 37 grid returned as is), in float32 and cast to the tokens' dtype, so float16 runs too.
"""
import math
import sys
from typing import Final, List

import torch
import torch.nn.functional as F

DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"  # facebookresearch/dinov2 main, 2026-06-03
NB_OUTPUTS = 4  # outputs: blocks depth / 4, 2 depth / 4, ...


def load_hub_model(name: str = "dinov2_vits14") -> torch.nn.Module:
    model = torch.hub.load(f"facebookresearch/dinov2:{DINOV2_COMMIT}", name, trust_repo=True, skip_validation=True)
    return model.eval()


class StandardizeImageNet(torch.nn.Module):
    """The Hub model's own normalisation, unchanged."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        if stats.numel() == 4:
            minv = stats[0]
            maxv = stats[1]
        else:
            minv = x.min()
            maxv = x.max()

        x = (x - minv) / (maxv - minv + 1e-6)
        return (x - self.mean) / self.std


class DinoV2(torch.nn.Module):
    patch: Final[int]
    grid: Final[int]
    offset: Final[float]
    antialias: Final[bool]
    prefix: Final[int]

    def __init__(self, vit: torch.nn.Module, stages: List[torch.nn.Module]) -> None:
        super().__init__()
        self.standardize_imagenet = StandardizeImageNet()
        self.patch = int(vit.patch_size)
        self.grid = int(round(math.sqrt(vit.pos_embed.shape[1] - 1)))
        self.offset = float(vit.interpolate_offset)
        self.antialias = bool(vit.interpolate_antialias)
        self.prefix = 1 + int(vit.num_register_tokens)
        self.patch_embed = vit.patch_embed.proj
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        # No registers (ViT-S) = an empty [1, 0, C] block, so the concatenation needs no branch.
        registers = vit.register_tokens if vit.register_tokens is not None else vit.cls_token[:, :0]
        self.register_tokens = torch.nn.Parameter(registers.detach().clone())
        self.stages = torch.nn.ModuleList(stages)
        self.norm = vit.norm

    def interpolate_pos_encoding(self, h: int, w: int, dtype: torch.dtype) -> torch.Tensor:
        if h == self.grid and w == self.grid:
            return self.pos_embed.to(dtype)
        pos_embed = self.pos_embed.float()
        patch_pos_embed = pos_embed[:, 1:].reshape(1, self.grid, self.grid, -1).permute(0, 3, 1, 2)
        if self.offset > 0.0:
            patch_pos_embed = F.interpolate(
                patch_pos_embed,
                scale_factor=[(h + self.offset) / self.grid, (w + self.offset) / self.grid],
                mode="bicubic",
                align_corners=False,
                antialias=self.antialias,
            )
        else:
            patch_pos_embed = F.interpolate(
                patch_pos_embed, size=[h, w], mode="bicubic", align_corners=False, antialias=self.antialias
            )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        return torch.cat((pos_embed[:, :1], patch_pos_embed), dim=1).to(dtype)

    def tokens(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1)
        tokens = tokens + self.interpolate_pos_encoding(h, w, tokens.dtype)
        registers = self.register_tokens.expand(tokens.shape[0], -1, -1)
        return torch.cat((tokens[:, :1], registers, tokens[:, 1:]), dim=1)

    def forward(
        self,
        x: torch.Tensor,
        nb_layers_tensor: torch.Tensor = torch.tensor([1]),
        stats: torch.Tensor = torch.tensor([]),
        direction: torch.Tensor = torch.tensor([]),
    ) -> List[torch.Tensor]:
        nb_layers = int(nb_layers_tensor.item())
        x = self.standardize_imagenet(x, stats)
        batch = x.shape[0]
        h = max((x.shape[2] + self.patch // 2) // self.patch, 1)
        w = max((x.shape[3] + self.patch // 2) // self.patch, 1)
        if x.shape[2] != h * self.patch or x.shape[3] != w * self.patch:
            x = F.interpolate(x, size=[h * self.patch, w * self.patch], mode="bilinear", align_corners=False)

        tokens = self.tokens(x, h, w)
        outputs: List[torch.Tensor] = []
        for stage in self.stages:
            tokens = stage(tokens)
            patches = self.norm(tokens)[:, self.prefix :]
            outputs.append(patches.reshape(batch, h, w, -1).permute(0, 3, 1, 2).contiguous())
            if len(outputs) == nb_layers:  # a return, not a break: TorchScript unrolls a ModuleList loop
                return outputs
        return outputs


def export(hub_name: str = "dinov2_vits14", stem: str = "DinoV2_Small") -> None:
    vit = load_hub_model(hub_name)
    depth = len(vit.blocks)
    ends = [depth * (i + 1) // NB_OUTPUTS for i in range(NB_OUTPUTS)]
    with torch.no_grad():
        example = vit.prepare_tokens_with_masks(torch.rand(1, 3, 224, 224))
    stages = [
        torch.jit.trace(torch.nn.Sequential(*vit.blocks[start:end]), example)
        for start, end in zip([0] + ends[:-1], ends)
    ]
    torch.jit.script(DinoV2(vit, stages).eval()).save(f"{stem}.pt")
    print(f"{stem}.pt: {hub_name} (patch tokens, blocks {ends})")


if __name__ == "__main__":
    export(*sys.argv[1:])
