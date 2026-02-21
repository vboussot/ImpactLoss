from typing import Any
import torch
from torchvision import models

class StandardizeImageNet(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            
    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        if stats.shape[0] == 4:
            minv = stats[0]
            maxv = stats[1]
        else:
            minv = x.min()
            maxv = x.max()
        
        x = (x - minv) / (maxv - minv + 1e-6)
        return (x - self.mean) / self.std
    
class VGG16(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        layers = [4,9,16,23,30]
        self.features = torch.nn.ModuleList([features[int(a):int(b)] for a, b in zip([0]+layers[:-1], layers)])
        self.standardize_imagenet = StandardizeImageNet()

    def forward(self, x: torch.Tensor, nb_layers_tensor: torch.Tensor = torch.tensor([1]), stats: torch.Tensor = torch.tensor([]), direction: torch.Tensor = torch.tensor([])) -> list[torch.Tensor]:
        nb_layers = int(nb_layers_tensor.item())
        x = self.standardize_imagenet(x, stats)
        results: list[torch.Tensor] = []
        if nb_layers >= 1:
            x = self.features[0](x)
            results.append(x)
        if nb_layers >= 2:
            x = self.features[1](x)
            results.append(x)
        if nb_layers >= 3:
            x = self.features[2](x)
            results.append(x)
        if nb_layers >= 4:
            x = self.features[3](x)
            results.append(x)
        if nb_layers >= 5:
            x = self.features[4](x)
            results.append(x)
        return results
    
model = VGG16()
sm = torch.jit.script(model)
sm.save("VGG16.pt")
