"""Export the DINOv3 ConvNeXt encoders as IMPACT TorchScript models.

Weights: the DINOv3 ConvNeXt checkpoints, which are covered by the DINOv3 License Agreement, NOT by an
open-source licence: read it before redistributing anything built here. Download them yourself and put
them beside this script.

The trunk is traced in evaluation mode and the wrapper around it is SCRIPTED rather than traced: tracing
drops the default values the forward declares, and a model whose four arguments are all required cannot
be called by a caller that passes only the image and its statistics.

    python Dinov3.py [layers]     layers: how many ConvNeXt stages to keep (default 3, all four)
"""
import sys

import torch
from convnext import ConvNeXt

convnext_sizes = {
    "Tiny": dict(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        checkpoint="dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
    ),
    "Small": dict(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        checkpoint="dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
    ),
}


class DinoV3(torch.nn.Module):

    def __init__(self, trunk: torch.nn.Module):
        super().__init__()
        self.model = trunk

    def forward(self, x: torch.Tensor, nb_layers_tensor: torch.Tensor = torch.tensor([4]), stats: torch.Tensor = torch.tensor([]), direction: torch.Tensor = torch.tensor([])) -> list[torch.Tensor]:
        return self.model(x, nb_layers_tensor, stats, direction)


if __name__ == "__main__":
    layers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    for size, args in convnext_sizes.items():
        state_init = torch.load(args["checkpoint"], map_location="cpu")
        model = ConvNeXt(depths=args["depths"][:layers + 1], dims=args["dims"][:layers + 1])
        state = {}
        for key in model.state_dict().keys():
            state[key] = state_init[key]
        model.load_state_dict(state)
        model.eval()
        example = torch.zeros((1, 3, 512, 512))
        traced = torch.jit.trace(model, (example, torch.tensor([layers + 1]), torch.tensor([]), torch.tensor([])))
        scripted_module = torch.jit.script(DinoV3(traced))
        scripted_module.save(f"./DinoV3_{size}.pt")
