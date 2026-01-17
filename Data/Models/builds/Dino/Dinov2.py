import torch

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
    
class DinoV2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.standardize_imagenet = StandardizeImageNet()

    def forward(self, x: torch.Tensor, nb_layers_tensor: torch.Tensor = torch.tensor([1]), stats: torch.Tensor = torch.tensor([]), direction: torch.Tensor = torch.tensor([])) -> list[torch.Tensor]:
        x = self.standardize_imagenet(x, stats)
        x = self.model.prepare_tokens_with_masks(x, None)
        for blk in self.model.blocks:
            x = blk(x)

        x = self.model.norm(x)
        return [x[:, 0].unsqueeze(2).unsqueeze(2)]

if __name__ == "__main__":
    example = torch.zeros((1,3,14,14)) 
    model = DinoV2()
    traced_script_module = torch.jit.trace(model, (example, torch.tensor([1]), torch.tensor([]), torch.tensor([])))
    traced_script_module.save("./Small.pt")
    
