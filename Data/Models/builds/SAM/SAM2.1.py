from sam2.build_sam import build_sam2
import torch
import requests
from tqdm import tqdm
import os

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
    
class SAM(torch.nn.Module):
    def __init__(self, model_cfg: str, checkpoint: str):
        super().__init__()
        self.model = build_sam2(model_cfg, checkpoint).image_encoder.to("cpu").trunk
        self.standardizeImageNet = StandardizeImageNet()

    def forward(self, x: torch.Tensor, nb_layers_tensor: torch.Tensor = torch.tensor([4]), stats: torch.Tensor = torch.tensor([]), direction: torch.Tensor = torch.tensor([])) -> list[torch.Tensor]:
        x = self.standardizeImageNet(x, stats)
        nb_layers = nb_layers_tensor.item()

        x = self.model.patch_embed(x)
        x = x + self.model._get_pos_embed((x.shape[1], x.shape[2]))

        outputs = []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if (i == self.model.stage_ends[-1]) or (
                i in self.model.stage_ends and self.model.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)
                if len(outputs) == nb_layers:
                    return outputs
        return outputs
    
url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    
if __name__ == "__main__":
    models = {"SAM2.1_Tiny": ("sam2.1_hiera_tiny.pt", "sam2.1_hiera_t.yaml"), "SAM2.1_Small": ("sam2.1_hiera_small.pt", "sam2.1_hiera_s.yaml")}
    
    for name, (checkpoint, model_cfg) in models.items():
        with open(checkpoint, "wb") as f:
            with requests.get(url+checkpoint, stream=True) as r:
                r.raise_for_status()

                total_size = int(r.headers.get('content-length', 0))
                progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
                for chunk in r.iter_content(chunk_size=8192 * 16):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
                progress_bar.close()
        
        model = SAM("configs/sam2.1/"+model_cfg, checkpoint)
        example = torch.zeros((1,3,29,29))
        sm = torch.jit.script(model)
        sm.save(name+".pt")
        #os.remove(checkpoint)
        