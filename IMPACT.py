# Copyright (c) 2025 Valentin Boussot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from huggingface_hub import hf_hub_download

class IMPACTReg(torch.nn.Module):
    """
    IMPACT-Reg Loss Module
    ------------------
    A similarity metric based on pretrained TorchScript feature extractors.
    Computes a weighted L1 distance between feature maps of two images.

    Parameters
    ----------
    model_name : str
        Name of the TorchScript model (e.g. "TS/M730").
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
        self.nb_layer = len(weights)
        
        # Download and load the TorchScript model
        self.model_path = hf_hub_download(
            repo_id="VBoussot/impact-torchscript-models", filename=model_name, repo_type="model", revision=None
        )  # nosec B615

        self.model: torch.nn.Module = torch.jit.load(self.model_path, map_location=torch.device("cpu"))  # nosec B614
        self.dim = len(shape)
        self.shape = shape if all(s > 0 for s in shape) else None
        
        dummy_input = torch.zeros((1, self.in_channels, *(self.shape if self.shape else [128] * self.dim))).to(0)
        try:
            out = self.model.to(0)(dummy_input, torch.tensor([self.nb_layer]))
            if not isinstance(out, (list, tuple)):
                raise TypeError(f"Expected model output to be a list or tuple, but got {type(out)}.")
            if self.nb_layer != len(out):
                raise ValueError(
                    f"Loss 'L1': mismatch between the number of weights "
                    f"({self.nb_layer}) and the number of model outputs "
                    f"({len(out)}). Each output must have a corresponding weight."
                )
        except Exception as e:
            msg = (
                f"[Model Sanity Check Failed]\n"
                f"Input shape attempted: {dummy_input.shape}\n"
                f"Error: {type(e).__name__}: {e}"
            )
            raise RuntimeError(msg) from e
        self.model = None

    def preprocessing(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Resize and adjust the tensor channels to match model expectations.

        Note:
            This method does not perform any intensity normalization.
            Input images must be provided in their raw (non-normalized) form,
            i.e., in their original intensity space (e.g., HU for CT, native values for MRI).

            If images have already been normalized upstream, they must be de-normalized
            before being passed to this pipeline.

            Intensity normalization or standardization is the responsibility of the IMPACT models.
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
        return [tensor, torch.tensor([self.nb_layer]), torch.tensor([tensor.min(), tensor.max(), tensor.mean(), tensor.std()])]
    

    def _compute(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the feature-based L1 loss between two tensors.
        """
        loss = torch.zeros(1, device=output.device, dtype=torch.float32)

        output = self.preprocessing(output)
        target = self.preprocessing(target)
        
        for weight, output_features, target_features in zip(self.weights, self.model(*output), self.model(*target)):
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
        self.model.to(output.device)
        self.model.eval()
        
        loss = torch.zeros(1, device=output.device, dtype=torch.float32)
        # 3D -> 2D model case: iterate over slices
        if len(output.shape) == 5 and self.dim == 2:
            for i in range(output.shape[2]):
                loss = loss + self._compute(output[:, :, i, ...], target[:, :, i, ...])
            loss = loss / output.shape[2]
        else:
            loss = self._compute(output, target)
        return loss

if __name__ == "__main__":
    # Example usage with a 3D model applied to a 3D volume
    loss = IMPACTReg("TS/M730.pt", [64,64,64], 1, [1,1])
    A = torch.rand(1,1,128,128,128)
    B = torch.rand(1,1,128,128,128)
    l = loss(A,B)
    print(l)
