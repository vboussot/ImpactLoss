import torch
from anatomix.model.network import Unet


class Normalize(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        if stats.shape[0] == 4:
            return (x - stats[0]) / (stats[1] - stats[0] + 1e-6)
        else:
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
    example = torch.ones((1,1,128,128,128))
    model = Unet(
        dimension=3,  # Only 3D supported for now
        input_nc=1,  # number of input channels
        output_nc=16,  # number of output channels
        num_downs=4,  # number of downsampling layers
        ngf=16,  # channel multiplier
    )
    
    model.load_state_dict(
        torch.load("./lib/anatomix/model-weights/anatomix.pth"),
        strict=True,
    )
    model = Anatomix(model)
    traced_script_module = torch.jit.trace(model, (example, torch.tensor([1]), torch.tensor([]), torch.tensor([])))
    traced_script_module.save("./Anatomix.pt")