import torch
from anatomix.model.network import Unet

class Anatomix(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = Unet(
            dimension=3,  # Only 3D supported for now
            input_nc=1,  # number of input channels
            output_nc=16,  # number of output channels
            num_downs=4,  # number of downsampling layers
            ngf=16,  # channel multiplier
        )
        self.model.load_state_dict(
            torch.load("./model-weights/anatomix.pth"),
            strict=True,
        )

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        return [self.model(input)]
    
example = torch.zeros((1,1,128,128,128))
model = Anatomix()
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("./Anatomix.pt")