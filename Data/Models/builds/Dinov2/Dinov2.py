import torch

class DinoV2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        x = self.model.prepare_tokens_with_masks(input, None)
        for blk in self.model.blocks:
            x = blk(x)

        x = self.model.norm(x)
        return [x[:, 0].unsqueeze(2).unsqueeze(2)]

example = torch.zeros((1,3,14,14)) 
model = DinoV2()
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("./Small_1_Layers.pt")