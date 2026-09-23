"""Export the anatomix ViT encoder (32 features) as an IMPACT TorchScript model.

Weights: neeldey/anatomix on the HuggingFace Hub, anatomix-dev-vit.pth (MIT, (c) 2024 Neel Dey), the
backbone their anatomix-register.py uses by default. Its architecture is PrimusV2 from
dynamic-network-architectures (Apache-2.0, (c) 2022 DKFZ), which anatomix configures and extends.

The positional embedding is sized for a 128-voxel cube, so the model takes 128x128x128 inputs only:
score it patch by patch at that size (PatchSize 128 128 128).

Same two rules as the U-Net export beside it: the network is traced in evaluation mode, and the wrapper
is scripted so its arguments keep their defaults.

    python AnatomixDevViT.py
"""
import torch
from anatomix.model.load_from_hf import ANATOMIX_VARIANTS, _load_handling_compile
from anatomix.model.vit3d import PrimusV2
from huggingface_hub import hf_hub_download


class Normalize(torch.nn.Module):

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        if stats.numel() == 4:
            return (x - stats[0]) / (stats[1] - stats[0] + 1e-6)
        vmin = x.min()
        return (x - vmin) / (x.max() - vmin + 1e-6)


class AnatomixViT(torch.nn.Module):

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.normalize = Normalize()
        self.model = model

    def forward(self, x: torch.Tensor, nb_layers: torch.Tensor = torch.tensor([1]), stats: torch.Tensor = torch.tensor([]), direction: torch.Tensor = torch.tensor([])) -> list[torch.Tensor]:
        x = self.normalize(x, stats)
        return [self.model(x)]


if __name__ == "__main__":
    example = torch.ones((1, 1, 128, 128, 128))  # the only input size this model accepts
    model = PrimusV2(**ANATOMIX_VARIANTS["anatomix-dev-vit"]["vit_kwargs"])
    model = _load_handling_compile(
        model,
        torch.load(hf_hub_download("neeldey/anatomix", "anatomix-dev-vit.pth", repo_type="model"), map_location="cpu"),
    )
    model.eval()
    scripted_module = torch.jit.script(AnatomixViT(torch.jit.trace(model, example)))
    scripted_module.save("./AnatomixDevViT.pt")
