from sam2.build_sam import build_sam2
import torch
import types
import os
import requests
from tqdm import tqdm

class SAM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
        with open("sam2.1_hiera_tiny.pt", "wb") as f:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()

                total_size = int(r.headers.get('content-length', 0))
                progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
                for chunk in r.iter_content(chunk_size=8192 * 16):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
                progress_bar.close()
        checkpoint = "./sam2.1_hiera_tiny.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        self.model = build_sam2(model_cfg, checkpoint).image_encoder.to("cpu").trunk
        os.remove("sam2.1_hiera_tiny.pt")
        
    def forward(self, sample: torch.Tensor) -> list[torch.Tensor]:
        layers = self.model(sample)
        return [layers[0], layers[1], layers[2], layers[3]]
    
model = SAM()
example = torch.zeros((1,3,512,512))
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("./Tiny_4_Layers.pt")