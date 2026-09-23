"""Export the DINOv2 ViT-S/14 encoder as an IMPACT TorchScript model.

Weights: facebookresearch/dinov2 through torch.hub (Apache-2.0).

The trunk is traced in evaluation mode and the wrapper around it is SCRIPTED rather than traced:
tracing drops the default values the forward declares, and a model whose four arguments are all
required cannot be called by a caller that passes only the image and its statistics.

    python Dinov2.py
"""
import torch


class StandardizeImageNet(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        if stats.numel() == 4:
            minv = stats[0]
            maxv = stats[1]
        else:
            minv = x.min()
            maxv = x.max()

        x = (x - minv) / (maxv - minv + 1e-6)
        return (x - self.mean) / self.std


class Trunk(torch.nn.Module):
    """Tokens, blocks, norm; the class token as a 1x1 feature map. Traced, so it may use Python control flow."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.prepare_tokens_with_masks(x, None)
        for blk in self.model.blocks:
            x = blk(x)

        x = self.model.norm(x)
        return x[:, 0].unsqueeze(2).unsqueeze(2)


class DinoV2(torch.nn.Module):

    def __init__(self, trunk: torch.nn.Module):
        super().__init__()
        self.standardize_imagenet = StandardizeImageNet()
        self.model = trunk

    def forward(self, x: torch.Tensor, nb_layers_tensor: torch.Tensor = torch.tensor([1]), stats: torch.Tensor = torch.tensor([]), direction: torch.Tensor = torch.tensor([])) -> list[torch.Tensor]:
        return [self.model(self.standardize_imagenet(x, stats))]


if __name__ == "__main__":
    example = torch.zeros((1, 3, 14, 14))
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    scripted_module = torch.jit.script(DinoV2(torch.jit.trace(Trunk(model), example)))
    scripted_module.save("./Small.pt")
