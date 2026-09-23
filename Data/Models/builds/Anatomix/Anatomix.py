"""Export the anatomix U-Net encoder (16 features) as an IMPACT TorchScript model.

Weights: neeldey/anatomix on the HuggingFace Hub, anatomix.pth (MIT, (c) 2024 Neel Dey).

The network is put in evaluation mode before tracing, so its BatchNorms use the learnt statistics
rather than each batch's own, and the wrapper around it is SCRIPTED rather than traced: tracing drops
the default values declared below, and a model whose four arguments are all required cannot be called
by a caller that passes only the image and its statistics.

    python Anatomix.py
"""
import torch
from anatomix.model.network import Unet
from huggingface_hub import hf_hub_download


class Normalize(torch.nn.Module):

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        if stats.numel() == 4:
            return (x - stats[0]) / (stats[1] - stats[0] + 1e-6)
        vmin = x.min()
        return (x - vmin) / (x.max() - vmin + 1e-6)


class Anatomix(torch.nn.Module):

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.normalize = Normalize()
        self.model = model

    def forward(self, x: torch.Tensor, nb_layers: torch.Tensor = torch.tensor([1]), stats: torch.Tensor = torch.tensor([]), direction: torch.Tensor = torch.tensor([])) -> list[torch.Tensor]:
        x = self.normalize(x, stats)
        return [self.model(x)]


if __name__ == "__main__":
    example = torch.ones((1, 1, 128, 128, 128))
    model = Unet(
        dimension=3,  # Only 3D supported for now
        input_nc=1,  # number of input channels
        output_nc=16,  # number of output channels
        num_downs=4,  # number of downsampling layers
        ngf=16,  # channel multiplier
    )

    model.load_state_dict(
        torch.load(hf_hub_download("neeldey/anatomix", "anatomix.pth", repo_type="model"), map_location="cpu"),
        strict=True,
    )
    model.eval()
    scripted_module = torch.jit.script(Anatomix(torch.jit.trace(model, example)))
    scripted_module.save("./Anatomix.pt")
