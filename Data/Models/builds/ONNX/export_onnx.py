"""Export an IMPACT TorchScript model of VBoussot/impact-torchscript-models to ONNX, at one depth.

A TorchScript model takes its depth as an argument, forward(x, nb_layers, stats, direction), and returns its first
nb_layers layers. ONNX has no runtime branch a GPU backend can take without reading the depth back to the host, so a
graph is exported per (model, depth), named after both and written beside the .pt's path, with its metadata:

    python export_onnx.py TS/M730.pt --layers 7       # -> onnx-out/TS/M730_depth7.onnx and M730_depth7.json
    python export_onnx.py Anatomix/Anatomix.pt        # -> onnx-out/Anatomix/Anatomix_depth1.onnx

Graph contract:
    inputs   x      float32 [1, 1, Z, Y, X], the spatial axes dynamic
             stats  float32 [4], the image's [min, max, mean, std]; a model that normalises by fixed statistics
                    (the TotalSegmentator CT models) ignores it, so every graph takes the same two inputs
    outputs  layer_0 ... layer_{n-1}, float32 [1, C_i, Z_i, Y_i, X_i]: the list the .pt returns for nb_layers = n

What the graph bakes:
    - the model's own normalisation, as the .pt applies it when it is handed 4 statistics;
    - no direction: the graph is the .pt called without one. A TotalSegmentator model reorients its input with
      `Canonical` from the direction it is handed (flips and an axis permutation chosen at run time); ONNX can only
      choose a permutation through an `If`, whose condition a GPU backend reads back to the host. The metadata says
      so ("orientation": "canonical"): a consumer that hands a direction applies `Canonical` to x and undoes it on
      each layer;
    - the head (TotalSegmentator's last layer) as the .pt's one-hot of the argmax, computed by comparison with the
      class indices rather than ONNX's OneHot;
    - instance norms whose mean and variance are taken one axis at a time (`_Norm`), which keeps a backend that sums
      a whole channel in one pass within 1e-5 of torch on a large image.

Metadata, <model>_depth<n>.json beside the graph:
    model, revision   the .pt it was exported from, and the repository revision it was read at
    depth             n, the nb_layers the .pt is called with
    stats             whether the normalisation reads the stats input (false: fixed statistics)
    orientation       "canonical" (the consumer applies the .pt's Canonical, see above) or "none"
    multiple          the size every spatial axis of x must divide by
    widest            (U-Nets) the channels of the widest tensor the graph computes at x's resolution, its working
                      memory's bound per pixel, a decoder's concatenation included
    layers            per output: its name, channels, stride ([Z, Y, X], x's size over the layer's), and nb_class for
                      the head's one-hot

Checks, on an input of another size than the export's: the rebuilt network against the .pt in torch (1e-5 relative L2
per layer, the input's gradient through the differentiable layers to 1e-5, the head's labels to the bit), then the
ONNX graph against the .pt with ONNX Runtime (1e-5 relative L2 per layer, labels to the bit).

Supported: the U-Nets of the builds (TS/*, MRSeg) and the models that wrap one network after their normalisation
(Anatomix). Requires torch, onnx, onnxruntime, huggingface_hub.
"""

import argparse
import json
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from torch import nn

REPO = "VBoussot/impact-torchscript-models"
# The revision KonfAI reads the models at.
REVISION = "47ffad660e67cebbac8d74aecf0fa181c5dac042"


def _copied(scripted, module):
    module.load_state_dict(scripted.state_dict())
    return module


def _conv(scripted):
    out_channels, in_channels, *kernel = scripted.weight.shape
    return _copied(scripted, nn.Conv3d(in_channels, out_channels, kernel, scripted.stride, scripted.padding))


def _transposed(scripted):
    in_channels, out_channels, *kernel = scripted.weight.shape
    return _copied(scripted, nn.ConvTranspose3d(in_channels, out_channels, kernel, scripted.stride))


class _Norm(nn.Module):
    """InstanceNorm3d, its mean and variance taken one axis at a time."""

    def __init__(self, scripted):
        super().__init__()
        self.register_buffer("weight", scripted.weight.detach().clone().view(1, -1, 1, 1, 1))
        self.register_buffer("bias", scripted.bias.detach().clone().view(1, -1, 1, 1, 1))
        self.eps = float(scripted.eps)

    def forward(self, x):
        def mean(y):
            return y.mean(4, keepdim=True).mean(3, keepdim=True).mean(2, keepdim=True)

        centered = x - mean(x)
        return centered / torch.sqrt(mean(centered * centered) + self.eps) * self.weight + self.bias


def _block(scripted):
    """A ConvBlock: twice a convolution, instance-normalised and leaky."""
    layers = []
    for i in (0, 1):
        conv, norm, activation = (getattr(scripted, f"{name}_{i}") for name in ("Conv", "Norm", "Activation"))
        layers += [_conv(conv), _Norm(norm), nn.LeakyReLU(activation.negative_slope)]
    return nn.Sequential(*layers)


def _crop(tensor, like):
    """crop_to_match: [ceil(diff / 2), size - diff // 2) along each spatial axis."""
    bounds = []
    for axis in (2, 3, 4):
        size = tensor.size(axis)
        diff = size - like.size(axis)
        bounds.append(slice((diff + 1) // 2, size - diff // 2))
    return tensor[:, :, bounds[0], bounds[1], bounds[2]]


class _Fixed(nn.Module):
    """ClipAndNormalize: the CT statistics the model stores."""

    def __init__(self, scripted):
        super().__init__()
        self.register_buffer("mean", scripted.mean.detach().clone())
        self.register_buffer("std", scripted.std.detach().clone())

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class _Standardize(nn.Module):
    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        return (x - stats[2]) / stats[3]


class _MinMax(nn.Module):
    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        return (x - stats[0]) / (stats[1] - stats[0] + 1e-6)


def _normalize(scripted):
    """The .pt's `normalize` as it runs when handed 4 statistics, by its class name."""
    kind = scripted.original_name
    if kind == "ClipAndNormalize":
        return _Fixed(scripted)
    if kind == "Standardize":
        return _Standardize()
    if kind == "Normalize":
        return _MinMax()
    raise SystemExit(f"unknown normalisation '{kind}'")


class UNet(nn.Module):
    """A U-Net of the builds (UnetCPP_*), as deep as `depth`: the encoder blocks, then the decoder's output, then the
    head's one-hot labels."""

    def __init__(self, ts, depth):
        super().__init__()
        self.normalize = _normalize(ts.normalize)
        encoder = sum(1 for name, _ in ts.named_children() if name.startswith("DownConvBlock_"))
        if not 1 <= depth <= encoder + 2:
            raise SystemExit(f"--layers must be between 1 and {encoder + 2} for this model")
        self.depth = depth
        self.down = nn.ModuleList(_block(getattr(ts, f"DownConvBlock_{i}")) for i in range(min(depth, encoder)))
        ups = range(encoder - 2, -1, -1) if depth > encoder else ()
        self.upsample = nn.ModuleList(_transposed(getattr(ts, f"Upsample_{i}")) for i in ups)
        self.up = nn.ModuleList(_block(getattr(ts, f"UpConvBlock_{i}")) for i in ups)
        self.head = _conv(ts.Head_0.Conv) if depth == encoder + 2 else None
        self.register_buffer("classes", torch.arange(int(ts.Head_0.nb_class)).view(1, -1, 1, 1, 1))

    def forward(self, x, stats):
        x = self.normalize(x, stats)
        layers = []
        for block in self.down:
            x = block(x)
            layers.append(x)
        if self.up:
            skips = layers[-2::-1]
            for upsample, block, skip in zip(self.upsample, self.up, skips):
                x = block(torch.cat([_crop(upsample(x), skip), skip], dim=1))
            layers.append(x)
        if self.head is not None:
            layers.append((self.head(x).argmax(1, keepdim=True) == self.classes).float())
        return tuple(layers)


class Wrapped(nn.Module):
    """A model that returns one network's output after its normalisation (Anatomix)."""

    def __init__(self, ts):
        super().__init__()
        self.normalize = _normalize(ts.normalize)
        self.net = ts.model

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.net(self.normalize(x, stats)),)


def rebuilt(ts, depth):
    if hasattr(ts, "DownConvBlock_0"):
        return UNet(ts, depth)
    if hasattr(ts, "model") and hasattr(ts, "normalize"):
        return torch.jit.script(Wrapped(ts))  # its network is a traced module: scripted around it
    raise SystemExit("this model is neither a U-Net of the builds nor a normalised network: no ONNX export for it")


def _inputs(shape, seed):
    x = torch.randn(1, 1, *shape, generator=torch.Generator().manual_seed(seed)) * 300 - 400
    return x, torch.stack([x.min(), x.max(), x.mean(), x.std()])


def _relative(got, expected):
    return float((got - expected).norm() / expected.norm())


def _compare(name, got, expected, labels):
    assert len(got) == len(expected), f"{len(got)} layers, the .pt returns {len(expected)}"
    for i, (a, b) in enumerate(zip(got, expected)):
        assert a.shape == b.shape, f"{name} layer {i}: {tuple(a.shape)} != {tuple(b.shape)}"
        if i == labels:
            assert torch.equal(a, b), f"{name} layer {i}: labels differ"
            print(f"  {name} layer {i} {tuple(a.shape)}: labels identical")
        else:
            error = _relative(a, b)
            print(f"  {name} layer {i} {tuple(a.shape)}: relative L2 {error:.1e}")
            assert error < 1e-5, f"{name} layer {i}: {error:.1e} off the .pt"


def _multiple(module):
    """The smallest power of two p such that the module runs on a side of 3 p: the size each axis must divide by."""
    for p in (1, 2, 4, 8, 16, 32, 64):
        try:
            with torch.no_grad():
                module(*_inputs((3 * p,) * 3, 0))
            return p
        except RuntimeError:
            continue
    raise SystemExit("the model runs on no side up to 192")


def write_metadata(ts, module, path, depth, target, head):
    multiple = _multiple(module)
    side = 64 if 64 % multiple == 0 else 4 * multiple
    widest = []

    def record(tensor):
        if tensor.shape[2:] == (side,) * 3:
            widest.append(tensor.shape[1])

    hooks = [
        hook
        for m in module.modules()
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d))
        for hook in (
            m.register_forward_pre_hook(lambda _, inputs: record(inputs[0])),
            m.register_forward_hook(lambda _, inputs, output: record(output)),
        )
    ]
    with torch.no_grad():
        layers = module(*_inputs((side,) * 3, 0))
    for hook in hooks:
        hook.remove()
    normalization = ts.normalize.original_name
    metadata = {
        "model": path,
        "revision": REVISION,
        "depth": depth,
        "stats": normalization != "ClipAndNormalize",
        "orientation": "canonical" if hasattr(ts, "canonical") else "none",
        "multiple": multiple,
        **({"widest": max(widest)} if widest else {}),
        "layers": [
            {"name": f"layer_{i}", "channels": layer.shape[1], "stride": [side // n for n in layer.shape[2:]]}
            | ({"nb_class": layer.shape[1]} if i == head else {})
            for i, layer in enumerate(layers)
        ],
    }
    target.write_text(json.dumps(metadata, indent=2) + "\n")


def export(path, depth, out, size, check):
    ts = torch.jit.load(hf_hub_download(REPO, path, revision=REVISION), map_location="cpu").eval()
    module = rebuilt(ts, depth).eval()
    head = depth - 1 if isinstance(module, UNet) and module.head is not None else None
    def reference(x, stats):  # no direction: Canonical leaves the image as it is
        return [layer.float() for layer in ts(x, torch.tensor([depth]), stats, torch.tensor([]))]

    # torch: values and the input's gradient.
    x, stats = _inputs(check, 1)
    x_ref, x_new = x.clone().requires_grad_(), x.clone().requires_grad_()
    expected, got = reference(x_ref, stats), list(module(x_new, stats))
    _compare("torch", [g.detach() for g in got], [e.detach() for e in expected], head)
    weights = [torch.randn(g.shape, generator=torch.Generator().manual_seed(i)) for i, g in enumerate(got)]
    differentiable = [i for i in range(len(got)) if i != head]
    sum((expected[i] * weights[i]).sum() for i in differentiable).backward()
    sum((got[i] * weights[i]).sum() for i in differentiable).backward()
    error = _relative(x_new.grad, x_ref.grad)
    print(f"  torch input gradient: relative L2 {error:.1e}")
    assert error < 1e-5, f"the input's gradient is {error:.1e} off the .pt's"

    target = out / Path(path).parent / f"{Path(path).stem}_depth{depth}.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    x, stats = _inputs(size, 0)
    names = [f"layer_{i}" for i in range(len(got))]
    with torch.no_grad(), tempfile.TemporaryDirectory() as scratch:
        raw = Path(scratch) / target.name
        torch.onnx.export(
            module,
            (x, stats),
            str(raw),
            dynamo=False,
            opset_version=18,
            input_names=["x", "stats"],
            output_names=names,
            dynamic_axes={"x": {2: "z", 3: "y", 4: "x"}} | {n: {2: f"{n}_z", 3: f"{n}_y", 4: f"{n}_x"} for n in names},
        )
        graph = onnx.load(str(raw))
        if [i.name for i in graph.graph.input] == ["x"]:  # a model with fixed statistics: stats is still an input
            graph.graph.input.append(onnx.helper.make_tensor_value_info("stats", onnx.TensorProto.FLOAT, [4]))
            onnx.save(graph, str(raw))
        # Constant chains (Anatomix's reflect pads) folded into constants: backends that refuse runtime pads run it.
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.optimized_model_filepath = str(target)
        ort.InferenceSession(str(raw), options, providers=["CPUExecutionProvider"])
    onnx.checker.check_model(str(target))

    x, stats = _inputs(check, 1)
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == ["x", "stats"]
    got = [torch.from_numpy(a) for a in session.run(None, {"x": x.numpy(), "stats": stats.numpy()})]
    with torch.no_grad():
        _compare("onnxruntime", got, reference(x, stats), head)
    write_metadata(ts, module, path, depth, target.with_suffix(".json"), head)
    print(f"{target} ({target.stat().st_size / 2**20:.1f} MiB)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="the .pt's path in the repository, e.g. TS/M730.pt")
    parser.add_argument("--layers", type=int, default=1, help="nb_layers, the depth the .pt is called with")
    parser.add_argument("--out", type=Path, default=Path("onnx-out"), help="the folder the .pt layout is mirrored in")
    parser.add_argument("--size", type=int, nargs=3, default=(32, 32, 32), help="the export input's Z Y X")
    parser.add_argument("--check", type=int, nargs=3, default=(48, 32, 64), help="the check input's Z Y X")
    args = parser.parse_args()
    torch.set_num_threads(8)
    export(args.model, args.layers, args.out, tuple(args.size), tuple(args.check))
