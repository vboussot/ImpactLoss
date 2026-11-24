import torch
import requests
from tqdm import tqdm
import zipfile
import shutil
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "TS"))
import TS  # noqa: E402  (the network, its normalisations by given statistics and the key map, as a module for TorchScript)


class UnetCPP(TS.Unet):
    """MRSegmentator's nnU-Net, TotalSegmentator MR's network of depth 6, without the reorientation: ``direction`` is
    taken and ignored, as the first export did. ``norms`` and ``norm_input`` are TS.Unet's."""

    def __init__(self, nb_class: int) -> None:
        super().__init__(nb_class, 6, mri=True)

    def forward(self, x: torch.Tensor, nb_layers_tensor: torch.Tensor = torch.tensor([1]), stats: torch.Tensor = torch.tensor([]), direction: torch.Tensor = torch.tensor([]), norms: list[torch.Tensor] | None = None) -> list[torch.Tensor]:
        return self._run(x, int(nb_layers_tensor.item()), stats, torch.empty(0), TS._given(norms), -1)[0]

    @torch.jit.export
    def norm_input(self, x: torch.Tensor, stats: torch.Tensor = torch.tensor([]), direction: torch.Tensor = torch.tensor([]), norms: list[torch.Tensor] | None = None) -> torch.Tensor:
        given = TS._given(norms)
        captured = self._run(x, 0, stats, torch.empty(0), given, len(given))[1]
        assert captured is not None, "every normalisation has its statistics"
        return captured


def download(url: str) -> dict[str, torch.Tensor]:
    with open(url.split("/")[-1], 'wb') as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()

            total_size = int(r.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
            for chunk in r.iter_content(chunk_size=8192 * 16):
                progress_bar.update(len(chunk))
                f.write(chunk)
            progress_bar.close()
    with zipfile.ZipFile(url.split("/")[-1], 'r') as zip_f:
        zip_f.extractall(url.split("/")[-1].replace(".zip", ""))
    os.remove(url.split("/")[-1])
    state_dict = torch.load(next(Path(url.split("/")[-1].replace(".zip", "")).rglob("checkpoint_final.pth"), None), weights_only=False)["network_weights"]
    shutil.rmtree(url.split("/")[-1].replace(".zip", ""))
    return state_dict

def convert_torchScript_full(model_name: str, model: torch.nn.Module, url: str):
    state_dict = download(url)
    tmp = {}

    model_state_dict_key = model.state_dict().keys()
    with open("Destination_Unet_1.txt") as f2:
        it = iter(state_dict.keys())
        for l1 in f2:
            key = next(it)
            while "decoder.seg_layers" in key:
                if "decoder.seg_layers.4" in key :
                    break
                key = next(it)
                    
            while "all_modules" in key or "decoder.encoder" in key:
                key = next(it)
            if TS.new_key(l1.replace("\n", ""), model.depth) not in model_state_dict_key:
                break
            tmp[TS.new_key(l1.replace("\n", ""), model.depth)] = state_dict[key]

    model.load_state_dict(tmp)

    sm = torch.jit.script(model)
    sm.save(model_name)

weight_url = "https://github.com/hhaentze/MRSegmentator/releases/download/v1.2.0/weights.zip"
       
if __name__ == "__main__":
    convert_torchScript_full("MRSeg.pt", UnetCPP(41), weight_url)