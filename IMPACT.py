# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025 Valentin Boussot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import shutil
from tqdm import tqdm
import os
import requests
from pathlib import Path

class IMPACT(torch.nn.Module):
    """
    IMPACT Loss Module
    ------------------
    A similarity metric based on pretrained TorchScript feature extractors.
    Computes a weighted L1 distance between feature maps of two images.

    Parameters
    ----------
    model_name : str
        Name of the TorchScript model (e.g. "TS/M730_2_Layers").
    shape : list[int]
        Expected spatial size. Use [0, 0, 0] for no resample.
    in_channels : int
        Number of input channels (before adapting to the model).
    weights : list[float]
        Per-layer feature weights.
    """
    def __init__(
        self, model_name: str, shape: list[int] = [0, 0], in_channels: int = 1, weights: list[float] = [1]
    ) -> None:
        super().__init__()
        if model_name is None:
            return
        self.in_channels = in_channels
        self.loss = torch.nn.L1Loss()
        self.weights = weights
        
        # Download and load the TorchScript model
        self.model_path = download_url(
            model_name,
            "https://huggingface.co/VBoussot/impact-torchscript-models/resolve/main/",
        )
        self.model: torch.nn.Module = torch.jit.load(self.model_path)  # nosec B614
        
        # Handle spatial dimensions
        self.dim = len(shape)
        self.shape = shape if all(s > 0 for s in shape) else None
        self.modules_loss: dict[str, dict[torch.nn.Module, float]] = {}
        
        # Sanity check on model I/O
        try:
            dummy_input = torch.zeros((1, self.in_channels, *(self.shape if self.shape else [224] * self.dim)))
            out = self.model(dummy_input)
            if not isinstance(out, (list, tuple)):
                raise TypeError(f"Expected model output to be a list or tuple, but got {type(out)}.")
            if len(weights) != len(out):
                raise ValueError(f"Mismatch between number of weights ({len(weights)}) and model outputs ({len(out)}).")
        except Exception as e:
            msg = (
                f"[Model Sanity Check Failed]\n"
                f"Input shape attempted: {dummy_input.shape}\n"
                f"Expected output length: {len(weights)}\n"
                f"Error: {type(e).__name__}: {e}"
            )
            raise RuntimeError(msg) from e
        
        # Lazy model loading (will reload at first forward call)
        self.model = None

    def preprocessing(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Resize and adjust the tensor channels to match model expectations.
        """
        if self.shape is not None and not all(
            tensor.shape[-i - 1] == size for i, size in enumerate(reversed(self.shape[2:]))
        ):
            if tensor.dtype == torch.uint8:
                mode = "nearest"
            elif len(tensor.shape) < 4:
                mode = "bilinear"
            else:
                mode = "trilinear"
            
            tensor = torch.nn.functional.interpolate(
                tensor, mode=mode, size=tuple(self.shape), align_corners=False
            ).type(torch.float32)
        # Replicate channels if needed (e.g., 1 → 3)
        if tensor.shape[1] != self.in_channels:
            tensor = tensor.repeat(tuple([1, self.in_channels] + [1 for _ in range(self.dim)]))
        return tensor

    def _compute(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the feature-based L1 loss between two tensors.
        """
        loss = torch.zeros((1), requires_grad=True).to(output.device, non_blocking=False).type(torch.float32)
        output = self.preprocessing(output)
        target = self.preprocessing(target)
        self.model.to(output.device)
        for weight, output_features, target_features in zip(self.weights, self.model(output), self.model(target)):
            if weight == 0:
                continue
            loss = loss + weight * self.loss(output_features, target_features)
        return loss

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Main forward method.
        If the model is 2D and inputs are 3D, compute slice-wise average loss.
        """
        if self.model is None:
            self.model = torch.jit.load(self.model_path)  # nosec B614
        loss = torch.zeros((1), requires_grad=True).to(output.device, non_blocking=False).type(torch.float32)
        # 3D → 2D model case: iterate over slices
        if len(output.shape) == 5 and self.dim == 2:
            for i in range(output.shape[2]):
                loss += self._compute(output[:, :, i, ...], target[:, :, i, ...])
            loss /= output.shape[2]
        else:
            loss = self._compute(output, target)
        return loss.to(output)

def download_url(model_name: str, url: str) -> str:
    """
    Download a TorchScript model from HuggingFace if not cached locally.
    """
    model_name = f"{model_name}.pt"
    base_path = (Path("~/.IMPACT/") / "models").expanduser()
    os.makedirs(base_path, exist_ok=True)

    subdirs = Path(model_name).parent
    model_dir = base_path / subdirs
    model_dir.mkdir(exist_ok=True)
    filetmp = model_dir / ("tmp_" + str(Path(model_name).name))
    file = model_dir / Path(model_name).name
    if file.exists():
        return str(file)
    try:
        print(f"[IMPACT] Downloading {model_name} to {file}")
        with requests.get(url + model_name, stream=True, timeout=10) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(filetmp, "wb") as f:
                with tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {model_name}",
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        shutil.copy2(filetmp, file)
        print("Download finished.")
    except Exception as e:
        raise e
    finally:
        if filetmp.exists():
            os.remove(filetmp)
    return str(file)

if __name__ == "__main__":
    # Example usage with a 3D model applied to a 3D volume
    loss = IMPACT("TS/M730_2_Layers", [0,0,0], 1, [1,1])
    A = torch.rand(1,1,128,128,128)
    B = torch.rand(1,1,128,128,128)
    l = loss(A,B)
    print(l)
