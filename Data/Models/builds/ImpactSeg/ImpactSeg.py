"""Export the IMPACT-Seg networks as IMPACT TorchScript models.

Weights: IMPACT-Seg (C. Hémon et al., https://github.com/HEMONCedric/IMPACTSeg), as its KonfAI apps on
https://huggingface.co/HCedric/IMPACTSeg hold them: fold 1 of fine-3d and of fine-2.5d, and the coarse model.

    python ImpactSeg.py BUNDLE    # BUNDLE: a download of https://huggingface.co/HCedric/IMPACTSeg

writes Fine3D.pt, Fine2.5D.pt and Coarse.pt here. Each is called as the TS models are,
forward(x, nb_layers, stats, direction), and returns its first nb_layers layers: each encoder stage's output, the last
decoder block's, then the one-hot of the label the app gives each voxel.

- stats ([min, max, mean, std]) standardizes x by its mean and std, as the apps' Standardize does (x's own without);
- direction reorients x onto RAS, as the apps' Canonical does, and the layers back onto x's axes;
- any size: a stride halves an odd side rounding up (its convolution does, and its skip's pooling is told to), and
  each decoder step cuts its upsampled map back to its skip, as TS's convolutions round up and crop. A side that
  divides by 32 (16 for the coarse one) runs as the app's network does. An input no side of which is longer than
  that reaches the deepest stage as one voxel, which an instance normalisation cannot normalise by its own
  statistics: stop before it (nb_layers) or give norms, as for the TS models;
- the 3-D models take TS's norms (the statistics of their instance normalisations), norm_input and norm_count.

Fine2.5D runs the 2-D network on each axial slice with its two neighbours on either side as its five channels, as the
app reads them (extend_slice 4, reflected past the first and the last slice), and returns its layers as volumes
[B, C, Z, Y / s, X / s]. Its BatchNorms use their learnt statistics: it has no norms to take.
"""

import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "TS"))
import TS  # noqa: E402  (StatNorm, Standardize, the direction's axes and the norms helpers, as a module for TorchScript)

FINE_IGNORED = [115, 116, 127, 128, 129]  # the fine apps' ignore_classes, which the released models do not predict


def _flipped(x: torch.Tensor, flips: tuple[bool, bool, bool, int, int, int]) -> torch.Tensor:
    if flips[0]:
        x = x.flip(2)
    if flips[1]:
        x = x.flip(3)
    if flips[2]:
        x = x.flip(4)
    return x


class Canonical(TS.Canonical):
    """The apps' Canonical (KonfAI's, onto RAS) from the direction: output axis i takes the image axis j that
    diag(-1, -1, 1) @ direction carries most onto it, flipped where that runs backwards. TS's reads the transposed
    direction and flips the axes before permuting them, which is the same only on a direction that keeps the axes in
    their order."""

    def get_permute_and_flip(self, direction: torch.Tensor) -> tuple[bool, bool, bool, int, int, int] | None:
        if len(direction.shape) != 2 or direction.size(0) != 3 or direction.size(1) != 3:
            return None
        onto = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.double, device=direction.device)) @ direction.to(torch.double)
        source: list[int] = []
        flip: list[bool] = []
        for i in range(3):
            j = int(torch.argmax(onto[i].abs()).item())
            source.append(4 - j)  # image axis j (x, y, z) is tensor dimension 4 - j
            flip.append(bool(onto[i, j].item() < 0))
        return flip[2], flip[1], flip[0], source[2], source[1], source[0]

    def forward(self, x: torch.Tensor, permute_and_flip: tuple[bool, bool, bool, int, int, int] | None) -> torch.Tensor:
        if permute_and_flip is None:
            return x
        return _flipped(x.permute([0, 1, permute_and_flip[3], permute_and_flip[4], permute_and_flip[5]]), permute_and_flip)

    def inverse(self, x: torch.Tensor, permute_and_flip: tuple[bool, bool, bool, int, int, int] | None) -> torch.Tensor:
        """``x`` back onto the image's axes, given get_inverse_permute_and_flip of the forward's."""
        if permute_and_flip is None:
            return x
        return _flipped(x, permute_and_flip).permute([0, 1, permute_and_flip[3], permute_and_flip[4], permute_and_flip[5]])


class BatchNorm(torch.nn.Module):
    """BatchNorm by its learnt statistics, as the 2.5D app runs it: nothing to take from outside."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(num_features))
        self.bias = torch.nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0))

    def forward(self, x: torch.Tensor, stat: torch.Tensor | None) -> torch.Tensor:
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, 1e-5)


class Unused(torch.nn.Module):
    """The skip normalisation of a block that does not project its skip."""

    def forward(self, x: torch.Tensor, stat: torch.Tensor | None) -> torch.Tensor:
        return x


def _conv(dim: int):
    return torch.nn.Conv3d if dim == 3 else torch.nn.Conv2d


class ConvNorm(torch.nn.Module):
    """A 3-voxel convolution, its normalisation and a LeakyReLU: the stem and each decoder block."""

    def __init__(self, dim: int, in_channels: int, out_channels: int, norm) -> None:
        super().__init__()
        self.conv = _conv(dim)(in_channels, out_channels, 3, 1, 1)
        self.norm = norm(out_channels)
        self.act = torch.nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor, norms: list[torch.Tensor], k: int, stop: int) -> tuple[torch.Tensor, int, bool]:
        """The output and the count of normalisations done; at ``stop``, the input of that normalisation instead."""
        x = self.conv(x)
        if k == stop:
            return x, k, True
        return self.act(self.norm(x, TS._stat(norms, k))), k + 1, False


class ResBlock(torch.nn.Module):
    """nnU-Net's BasicBlockD, as KonfAI's ResidualBlockD: a strided block pools its skip, a channel change projects
    it, and the skip's normalisation runs first."""

    def __init__(self, dim: int, in_channels: int, out_channels: int, stride: int, norm) -> None:
        super().__init__()
        conv = _conv(dim)
        self.dim, self.stride = dim, stride
        self.project = in_channels != out_channels
        self.skip_conv = conv(in_channels, out_channels, 1, 1, 0, bias=False) if self.project else torch.nn.Identity()
        self.skip_norm = norm(out_channels) if self.project else Unused()
        self.conv1 = conv(in_channels, out_channels, 3, stride, 1)
        self.norm1 = norm(out_channels)
        self.act1 = torch.nn.LeakyReLU(0.01)
        self.conv2 = conv(out_channels, out_channels, 3, 1, 1)
        self.norm2 = norm(out_channels)
        self.act2 = torch.nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor, norms: list[torch.Tensor], k: int, stop: int) -> tuple[torch.Tensor, int, bool]:
        skip = x
        if self.stride > 1:
            # Averaged over the window's voxels inside the map, as a rounded-up stride reads past an odd side; a side
            # of one voxel keeps it.
            kernel = [min(self.stride, n) for n in x.shape[2:]]
            if self.dim == 3:
                skip = F.avg_pool3d(x, kernel, self.stride, 0, ceil_mode=True)
            else:
                skip = F.avg_pool2d(x, kernel, self.stride, 0, ceil_mode=True)
        if self.project:
            skip = self.skip_conv(skip)
            if k == stop:
                return skip, k, True
            skip = self.skip_norm(skip, TS._stat(norms, k))
            k += 1
        out = self.conv1(x)
        if k == stop:
            return out, k, True
        out = self.conv2(self.act1(self.norm1(out, TS._stat(norms, k))))
        if k + 1 == stop:
            return out, k + 1, True
        return self.act2(self.norm2(out, TS._stat(norms, k + 1)) + skip), k + 2, False


class UpStep(torch.nn.Module):

    def __init__(self, dim: int, in_channels: int, out_channels: int, norm) -> None:
        super().__init__()
        transpose = torch.nn.ConvTranspose3d if dim == 3 else torch.nn.ConvTranspose2d
        self.upsample = transpose(in_channels, out_channels, 2, 2)
        self.block = ConvNorm(dim, 2 * out_channels, out_channels, norm)


class Head(torch.nn.Module):
    """The one-hot of the label the app gives each voxel: its most probable class, ``ignored`` left out."""

    def __init__(self, dim: int, in_channels: int, num_classes: int, ignored: list[int]) -> None:
        super().__init__()
        self.conv = _conv(dim)(in_channels, num_classes, 1)
        keep = torch.ones(num_classes)
        keep[ignored] = 0
        self.register_buffer("keep", keep, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(self.conv(x), dim=1) * self.keep.view([1, -1] + [1] * (x.dim() - 2))
        return torch.zeros_like(probabilities).scatter_(1, probabilities.argmax(1, keepdim=True), 1.0)


def _matched(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """An upsampled map cut to its skip: a stride rounds an odd side up, so the way back is one voxel longer."""
    for axis in range(2, x.dim()):
        x = x.narrow(axis, 0, skip.size(axis))
    return x


class ResEncUNet(torch.nn.Module):
    """IMPACT-Seg's residual-encoder U-Net (the apps' model:ImpactSegResidualEncoderUNet) on a standardized, canonical
    input."""

    def __init__(self, dim: int, in_channels: int, features: list[int], blocks: list[int], num_classes: int, ignored: list[int]) -> None:
        super().__init__()
        norm = TS.StatNorm if dim == 3 else BatchNorm
        n = len(features)
        self.stem = ConvNorm(dim, in_channels, features[0], norm)
        self.encoder = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [
                        ResBlock(dim, features[s] if b or not s else features[s - 1], features[s], 2 if s and not b else 1, norm)
                        for b in range(blocks[s])
                    ]
                )
                for s in range(n)
            ]
        )
        self.up = torch.nn.ModuleList([UpStep(dim, features[n - 1 - d], features[n - 2 - d], norm) for d in range(n - 1)])
        self.head = Head(dim, features[0], num_classes, ignored)
        # The normalisations the first l + 1 layers go through: the stem, each block (its skip's when it projects),
        # each decoder block; the head adds none.
        counts = [1]
        for stage in self.encoder:
            counts.append(counts[-1] + sum(2 + int(block.project) for block in stage))
        counts = counts[1:]
        counts += [counts[-1] + n - 1] * 2
        self.counts = counts

    def layers(self, x: torch.Tensor, nb_layers: int, norms: list[torch.Tensor], stop: int) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """The first ``nb_layers`` layers (0: all); once the normalisation ``stop`` is reached, its input instead."""
        outputs: list[torch.Tensor] = []
        x, k, hit = self.stem(x, norms, 0, stop)
        if hit:
            return outputs, x
        skips: list[torch.Tensor] = []
        for stage in self.encoder:
            for block in stage:
                x, k, hit = block(x, norms, k, stop)
                if hit:
                    return outputs, x
            skips.append(x)
            outputs.append(x)
            if len(outputs) == nb_layers:
                return outputs, None
        level = len(skips) - 2
        for step in self.up:
            x, k, hit = step.block(torch.cat([_matched(step.upsample(x), skips[level]), skips[level]], dim=1), norms, k, stop)
            if hit:
                return outputs, x
            level -= 1
        outputs.append(x)
        if len(outputs) == nb_layers:
            return outputs, None
        outputs.append(self.head(x))
        return outputs, None


class Volume(torch.nn.Module):
    """A 3-D network as IMPACT calls it. ``norms`` and ``norm_input`` are TS.Unet's."""

    def __init__(self, net: ResEncUNet) -> None:
        super().__init__()
        self.normalize = TS.Standardize()
        self.canonical = Canonical()
        self.net = net

    def _run(self, x: torch.Tensor, nb_layers: int, stats: torch.Tensor, direction: torch.Tensor, norms: list[torch.Tensor], stop: int) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        permute_and_flip = self.canonical.get_permute_and_flip(direction)
        inverse = self.canonical.get_inverse_permute_and_flip(permute_and_flip)
        layers, captured = self.net.layers(self.canonical(self.normalize(x, stats), permute_and_flip), nb_layers, norms, stop)
        outputs = [self.canonical.inverse(layer, inverse) for layer in layers]
        if captured is None:
            return outputs, None
        return outputs, self.canonical.inverse(captured, inverse)

    def forward(self, x: torch.Tensor, nb_layers_tensor: torch.Tensor = torch.tensor([1]), stats: torch.Tensor = torch.tensor([]), direction: torch.Tensor = torch.tensor([]), norms: list[torch.Tensor] | None = None) -> list[torch.Tensor]:
        return self._run(x, int(nb_layers_tensor.item()), stats, direction, TS._given(norms), -1)[0]

    @torch.jit.export
    def norm_input(self, x: torch.Tensor, stats: torch.Tensor = torch.tensor([]), direction: torch.Tensor = torch.tensor([]), norms: list[torch.Tensor] | None = None) -> torch.Tensor:
        """The input of the normalisation numbered ``len(norms)``, the earlier ones by ``norms``, on the image's axes."""
        given = TS._given(norms)
        captured = self._run(x, 0, stats, direction, given, len(given))[1]
        assert captured is not None, "every normalisation has its statistics"
        return captured

    @torch.jit.export
    def norm_count(self, nb_layers: int) -> int:
        """The normalisations the first ``nb_layers`` layers go through (0: all of them)."""
        if 0 < nb_layers <= len(self.net.counts):
            return self.net.counts[nb_layers - 1]
        return self.net.counts[-1]


class Slices(torch.nn.Module):
    """The 2.5D network over a volume [B, 1, Z, Y, X], as the app runs it: each axial slice with its two neighbours on
    either side as its channels, reflected past the first and the last slice; its layers stacked back along Z."""

    def __init__(self, net: ResEncUNet) -> None:
        super().__init__()
        self.normalize = TS.Standardize()
        self.canonical = Canonical()
        self.net = net

    def forward(self, x: torch.Tensor, nb_layers_tensor: torch.Tensor = torch.tensor([1]), stats: torch.Tensor = torch.tensor([]), direction: torch.Tensor = torch.tensor([])) -> list[torch.Tensor]:
        permute_and_flip = self.canonical.get_permute_and_flip(direction)
        inverse = self.canonical.get_inverse_permute_and_flip(permute_and_flip)
        x = self.canonical(self.normalize(x, stats), permute_and_flip)
        batch, depth = x.size(0), x.size(2)
        stack = F.pad(x, [0, 0, 0, 0, 2, 2], mode="reflect").unfold(2, 5, 1)  # [B, 1, Z, Y, X, 5]: slices z-2 .. z+2
        stack = stack[:, 0].permute(0, 1, 4, 2, 3).reshape(batch * depth, 5, x.size(3), x.size(4))
        layers = self.net.layers(stack, int(nb_layers_tensor.item()), [], -1)[0]
        return [
            self.canonical.inverse(layer.reshape(batch, depth, layer.size(1), layer.size(2), layer.size(3)).movedim(1, 2), inverse)
            for layer in layers
        ]


NAMES = {"Conv_0": "conv", "Norm_0": "norm", "SkipConv": "skip_conv", "SkipNorm": "skip_norm", "Conv1": "conv1", "Norm1": "norm1", "Conv2": "conv2", "Norm2": "norm2"}


def new_key(key: str, last: int) -> str | None:
    """The name here of a weight of the app's network, None for a deep-supervision head left out."""
    if m := re.fullmatch(r"Stem\.(\w+)\.(\w+)", key):
        return f"stem.{NAMES[m[1]]}.{m[2]}"
    if m := re.fullmatch(r"Encoder_(\d+)_Block_(\d+)\.(\w+)\.(\w+)", key):
        return f"encoder.{m[1]}.{m[2]}.{NAMES[m[3]]}.{m[4]}"
    if m := re.fullmatch(r"Up_(\d+)\.(\w+)", key):
        return f"up.{m[1]}.upsample.{m[2]}"
    if m := re.fullmatch(r"Decoder_(\d+)\.(\w+)\.(\w+)", key):
        return f"up.{m[1]}.block.{NAMES[m[2]]}.{m[3]}"
    if m := re.fullmatch(r"Logits_(\d+)\.(\w+)", key):
        return f"head.conv.{m[2]}" if int(m[1]) == last else None
    raise KeyError(key)


FINE = [32, 64, 128, 256, 320, 512], [1, 3, 4, 6, 6, 6]
#: name: (weights in the bundle, network dimension, input channels, features and blocks per stage, classes, ignored)
MODELS = {
    "Fine3D": ("fine-3d/fold1.safetensors", 3, 1, *FINE, 132, FINE_IGNORED),
    "Fine2.5D": ("fine-2.5d/fold1.safetensors", 2, 5, *FINE, 132, FINE_IGNORED),
    "Coarse": ("coarse/coarse.safetensors", 3, 1, [16, 32, 64, 128, 192], [1, 1, 2, 2, 2], 12, []),
}


def build(name: str, bundle: Path) -> torch.nn.Module:
    """Model ``name`` in eval mode, every weight of its app's network but the deep-supervision heads loaded."""
    weights, dim, in_channels, features, blocks, num_classes, ignored = MODELS[name]
    net = ResEncUNet(dim, in_channels, features, blocks, num_classes, ignored)
    state = {}
    for key, value in load_file(str(bundle / weights)).items():
        renamed = new_key(key.split("/", 2)[2], len(features) - 2)  # Model/<network>/<parameter>
        if renamed is not None:
            state[renamed] = value
    net.load_state_dict(state)
    return (Volume(net) if dim == 3 else Slices(net)).eval()


if __name__ == "__main__":
    for name in MODELS:
        torch.jit.script(build(name, Path(sys.argv[1]))).save(f"{name}.pt")
        print(f"{name}.pt", flush=True)
