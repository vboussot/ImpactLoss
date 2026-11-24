import torch
import requests
from tqdm import tqdm
import zipfile
import shutil
from pathlib import Path
import json
import os

def _given(norms: list[torch.Tensor] | None) -> list[torch.Tensor]:
    given: list[torch.Tensor] = []
    if norms is not None:
        given = norms
    return given


def _stat(norms: list[torch.Tensor], k: int) -> torch.Tensor | None:
    if k < len(norms):
        return norms[k]
    return None


class StatNorm(torch.nn.Module):
    """InstanceNorm3d (affine) by the statistics it is handed, a [2, C] mean and variance, or else by its input's own:
    a region of an image then reads as it does in the whole image."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(num_features))
        self.bias = torch.nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, stat: torch.Tensor | None) -> torch.Tensor:
        if stat is None:
            return torch.nn.functional.instance_norm(x, weight=self.weight, bias=self.bias, eps=1e-5)
        stat = stat.to(x.dtype)
        shape = [1, x.size(1)] + [1] * (x.dim() - 2)
        scale = self.weight * torch.rsqrt(stat[1] + 1e-5)
        return (x - stat[0].view(shape)) * scale.view(shape) + self.bias.view(shape)


class ConvBlock(torch.nn.Module):
    
    def __init__(self, in_channels : int, out_channels : int, downsample: bool = False, mri: bool = False, lung: bool = False) -> None:
        super().__init__()
        if not lung:
            self.Conv_0 = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=(2 if downsample else 1) if not mri else ((1,2,2) if downsample else 1), padding=1, bias=True)
        else:
            self.Conv_0 = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=(2,1,2), padding=1, bias=True)
         
        self.Norm_0 = StatNorm(out_channels)
        self.Activation_0 = torch.nn.LeakyReLU(negative_slope=0.01)
        self.Conv_1 = torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.Norm_1 = StatNorm(out_channels)
        self.Activation_1 = torch.nn.LeakyReLU(negative_slope=0.01)
        
    def forward(self, input: torch.Tensor, norms: list[torch.Tensor], k: int, stop: int) -> tuple[torch.Tensor, int, bool]:
        """The block's output and the count of normalisations done, its two numbered k and k + 1; once ``stop`` is
        reached, the input of that normalisation instead."""
        output = self.Conv_0(input)
        if k == stop:
            return output, k, True
        output = self.Activation_0(self.Norm_0(output, _stat(norms, k)))
        output = self.Conv_1(output)
        if k + 1 == stop:
            return output, k + 1, True
        output = self.Activation_1(self.Norm_1(output, _stat(norms, k + 1)))
        return output, k + 2, False

class Head(torch.nn.Module):

    def __init__(self, in_channels: int, nb_class: int) -> None:
        super().__init__()
        self.Conv = torch.nn.Conv3d(in_channels = in_channels, out_channels = nb_class, kernel_size = 1, stride = 1, padding = 0)
        self.softmax = torch.nn.Softmax(dim=1)
        self.nb_class = nb_class

    def forward(self, input: torch.Tensor) -> torch.Tensor:
         return torch.nn.functional.one_hot(torch.argmax(self.softmax(self.Conv(input)), dim=1), self.nb_class).permute((0,4,1,2,3))


class ClipAndNormalize(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        #self.register_buffer("clip_min", torch.empty(1))
        #self.register_buffer("clip_max", torch.empty(1))
        self.register_buffer("mean", torch.empty(1))
        self.register_buffer("std", torch.empty(1))

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        #x = torch.clamp(x, self.clip_min, self.clip_max)
        return (x - self.mean) / (self.std)


class Standardize(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        if stats.numel() == 4:
            return (x-stats[2])/stats[3]
        else:
            return (x-x.mean())/x.std()
        
def crop_to_match(tensor, target_tensor):
    diff_x = tensor.size(2) - target_tensor.size(2)
    diff_y = tensor.size(3) - target_tensor.size(3)
    diff_z = tensor.size(4) - target_tensor.size(4)
    return tensor[:, :, torch.ceil(diff_x / 2) : tensor.size(2)-(diff_x // 2), torch.ceil(diff_y / 2) : tensor.size(3)-(diff_y // 2), torch.ceil(diff_z / 2) : tensor.size(4)-(diff_z // 2)]

class Canonical(torch.nn.Module):
    
    def __init__(self) -> None:
        super().__init__()

    def get_permute_and_flip(self, direction: torch.Tensor) -> tuple[bool, bool, bool, int, int, int] | None:
        if len(direction.shape) != 2 or direction.size(0) != 3 or direction.size(1) != 3:
            return None

        A = torch.diag(torch.tensor([-1., -1., 1.], dtype=torch.double, device=direction.device)) @ direction.to(torch.double).T
        
        perm_xyz: list[int] = []
        flip_xyz: list[bool] = []
        for i in range(3):
            j = int(torch.argmax(A[i].abs()).item())
            perm_xyz.append(j)
            flip_xyz.append(bool(A[i, j].item() < 0))

        axis_to_dim = [4, 3, 2]
        return flip_xyz[2], flip_xyz[1], flip_xyz[0], axis_to_dim[perm_xyz[2]], axis_to_dim[perm_xyz[1]] , axis_to_dim[perm_xyz[0]]
    
    def get_inverse_permute_and_flip(self, permute_and_flip: tuple[bool, bool, bool, int, int, int] | None) -> tuple[bool, bool, bool, int, int, int] | None:
        if permute_and_flip is None:
            return None
        flip_x, flip_y, flip_z, pD, pH, pW = permute_and_flip
        inv = [0, 0, 0]
        inv[pD - 2] = 2
        inv[pH - 2] = 3
        inv[pW - 2] = 4
        inv_pD, inv_pH, inv_pW = inv
        return (flip_x, flip_y, flip_z, inv_pD, inv_pH, inv_pW)

    def forward(self, x: torch.Tensor, permute_and_flip: tuple[bool, bool, bool, int, int, int] | None) -> torch.Tensor:
        if permute_and_flip is None:
            return x
        if permute_and_flip[0]:
            x = x.flip(2)
        if permute_and_flip[1]:
            x = x.flip(3)
        if permute_and_flip[2]:
            x = x.flip(4)
        x = x.permute([0,1, permute_and_flip[3], permute_and_flip[4], permute_and_flip[5]])
        return x
    
    def inverse(self, x: torch.Tensor, permute_and_flip: tuple[bool, bool, bool, int, int, int] | None) -> torch.Tensor:
        if permute_and_flip is None:
            return x
        x = x.permute([0,1, permute_and_flip[3], permute_and_flip[4], permute_and_flip[5]])
        if permute_and_flip[0]:
            x = x.flip(2)
        if permute_and_flip[1]:
            x = x.flip(3)
        if permute_and_flip[2]:
            x = x.flip(4)
        return x
    
class UpStep(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel: tuple[int, int, int]) -> None:
        super().__init__()
        self.Upsample = torch.nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=kernel)
        self.Block = ConvBlock(in_channels=out_channels * 2, out_channels=out_channels)


class Unet(torch.nn.Module):
    """TotalSegmentator's nnU-Net: ``depth`` encoder blocks, the decoder back to full resolution, the head. Its layers
    are the encoder blocks' outputs, the decoder's last and the head's.

    ``norms``, one [2, C] mean and variance per instance normalisation in the order they run, sets the statistics
    those normalisations use, the later ones their input's own: a region normalised by the whole image's statistics
    gives the whole image's layers there. ``norm_input`` returns the input of the next normalisation, which a pass over
    the regions sums for its statistics."""

    def __init__(self, nb_class: int, depth: int, mri: bool = False, lung: bool = False) -> None:
        super().__init__()
        if mri:
            self.normalize  = Standardize()
        else:
            self.normalize = ClipAndNormalize()
        self.canonical = Canonical()
        channels = [32, 64, 128, 256, 320, 320][:depth]
        self.down = torch.nn.ModuleList(
            [ConvBlock(1 if i == 0 else channels[i - 1], channels[i], downsample=i > 0, mri=mri and i == 5, lung=lung and i == 5) for i in range(depth)]
        )
        # The decoder, deepest first; only the step after a sixth block takes its anisotropic stride.
        kernels = [(2, 1, 2) if lung and i == 4 else (1, 2, 2) if mri and i == 4 else (2, 2, 2) for i in range(depth - 1)]
        self.up = torch.nn.ModuleList([UpStep(channels[i + 1], channels[i], kernels[i]) for i in reversed(range(depth - 1))])
        self.head = Head(32, nb_class)
        self.depth = depth

    def _run(self, x: torch.Tensor, nb_layers: int, stats: torch.Tensor, direction: torch.Tensor, norms: list[torch.Tensor], stop: int) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        x = self.normalize(x, stats)
        permute_and_flip = self.canonical.get_permute_and_flip(direction)
        inverse = self.canonical.get_inverse_permute_and_flip(permute_and_flip)
        x = self.canonical(x, permute_and_flip)
        outputs: list[torch.Tensor] = []
        skips: list[torch.Tensor] = []
        k = 0
        for block in self.down:
            x, k, hit = block(x, norms, k, stop)
            if hit:
                return outputs, self.canonical.inverse(x, inverse)
            skips.append(x)
            outputs.append(self.canonical.inverse(x, inverse))
            if len(outputs) == nb_layers:
                return outputs, None
        level = len(skips) - 2
        for step in self.up:
            x = step.Upsample(x)
            skip = skips[level]
            x, k, hit = step.Block(torch.cat([crop_to_match(x, skip), skip], dim=1), norms, k, stop)
            if hit:
                return outputs, self.canonical.inverse(x, inverse)
            level -= 1
        outputs.append(self.canonical.inverse(x, inverse))
        if len(outputs) == nb_layers:
            return outputs, None
        outputs.append(self.canonical.inverse(self.head(x), inverse))
        return outputs, None

    def forward(self, x: torch.Tensor, nb_layers_tensor: torch.Tensor = torch.tensor([0]), stats: torch.Tensor = torch.tensor([]), direction: torch.Tensor = torch.tensor([]), norms: list[torch.Tensor] | None = None) -> list[torch.Tensor]:
        return self._run(x, int(nb_layers_tensor.item()), stats, direction, _given(norms), -1)[0]

    @torch.jit.export
    def norm_input(self, x: torch.Tensor, stats: torch.Tensor = torch.tensor([]), direction: torch.Tensor = torch.tensor([]), norms: list[torch.Tensor] | None = None) -> torch.Tensor:
        """The input of the normalisation numbered ``len(norms)``, the earlier ones by ``norms``, on the image's axes."""
        given = _given(norms)
        captured = self._run(x, 0, stats, direction, given, len(given))[1]
        assert captured is not None, "every normalisation has its statistics"
        return captured

    @torch.jit.export
    def norm_count(self, nb_layers: int) -> int:
        """The normalisations the first ``nb_layers`` layers go through (0: all of them)."""
        if 0 < nb_layers <= self.depth:
            return 2 * nb_layers
        return 2 * (2 * self.depth - 1)


def UnetCPP_1(nb_class: int, mri: bool = False, lung: bool = False) -> Unet:
    return Unet(nb_class, 6, mri, lung)


def UnetCPP_2(nb_class: int, mri: bool = False) -> Unet:
    return Unet(nb_class, 5, mri)


def UnetCPP_3(nb_class: int, mri: bool = False) -> Unet:
    return Unet(nb_class, 4, mri)


def new_key(key: str, depth: int) -> str:
    """The name in ``Unet`` of a weight named as the former per-depth classes (and Destination_Unet_*.txt) name it."""
    for old, new in (("DownConvBlock_", "down."), ("Upsample_", "up.{}.Upsample"), ("UpConvBlock_", "up.{}.Block"), ("Head_0", "head")):
        if key.startswith(old):
            if old == "Head_0":
                return "head" + key[len(old):]
            index, rest = key[len(old):].split(".", 1)
            return (new + index if old == "DownConvBlock_" else new.format(depth - 2 - int(index))) + "." + rest
    return key


def download(url: str, mri: bool) -> dict[str, torch.Tensor]:
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
    path = Path(url.split("/")[-1].replace(".zip", ""))
    state_dict = torch.load(next(path.rglob("checkpoint_final.pth"), None), weights_only=False)["network_weights"]
    if not mri:
        dataset_fingerprint_path = next(path.rglob("dataset_fingerprint.json"), None)
        with open(dataset_fingerprint_path, "r") as f:
            data = json.load(f)

        ch0 = data["foreground_intensity_properties_per_channel"]["0"]

        state_dict["mean"] = torch.tensor([ch0["mean"]])
        state_dict["std"] = torch.tensor([ch0["std"]])
        state_dict["clip_min"] = torch.tensor([ch0["percentile_00_5"]])
        state_dict["clip_max"] = torch.tensor([ch0["percentile_99_5"]])
    shutil.rmtree(url.split("/")[-1].replace(".zip", ""))
    return state_dict

def convert_torchScript_full(model_name: str, model: torch.nn.Module, type: int, url: str, mri: bool):
    state_dict = download(url, mri)
    tmp = {}
    with open("Destination_Unet_{}.txt".format(type)) as f2:
        it = iter(state_dict.keys())
        for l1 in f2:
            key = next(it)
            while "decoder.seg_layers" in key:
                if type == 1:
                    if "decoder.seg_layers.4" in key :
                        break
                if type == 2:
                    if "decoder.seg_layers.3"  in key:
                        break
                if type == 3:
                    if "decoder.seg_layers.2" in key:
                        break
                key = next(it)
                    
            while "all_modules" in key or "decoder.encoder" in key:
                key = next(it)
            tmp[new_key(l1.replace("\n", ""), model.depth)] = state_dict[key]
    if not mri:
        #tmp["normalize.clip_min"] = state_dict["clip_min"]
        #tmp["normalize.clip_max"] = state_dict["clip_max"]
        tmp["normalize.mean"] = state_dict["mean"]
        tmp["normalize.std"] = state_dict["std"]
        
    model.load_state_dict(tmp)

    sm = torch.jit.script(model)
    sm.save("./{}.pt".format(model_name))

url = "https://github.com/wasserth/TotalSegmentator/releases/download/"
       
models = {
         "M258" : (UnetCPP_1(nb_class=3, lung=True), 1, url+"v2.0.0-weights/Dataset258_lung_vessels_248subj.zip", False),
         "M291" : (UnetCPP_1(nb_class=25), 1, url+"v2.0.0-weights/Dataset291_TotalSegmentator_part1_organs_1559subj.zip", False), 
         "M292" : (UnetCPP_1(nb_class=27), 1, url+"v2.0.0-weights/Dataset292_TotalSegmentator_part2_vertebrae_1532subj.zip", False),
         "M293" : (UnetCPP_1(nb_class=19), 1, url+"v2.0.0-weights/Dataset293_TotalSegmentator_part3_cardiac_1559subj.zip", False),
         "M294" : (UnetCPP_1(nb_class=24), 1, url+"v2.0.0-weights/Dataset294_TotalSegmentator_part4_muscles_1559subj.zip", False),
         "M295" : (UnetCPP_1(nb_class=27), 1, url+"v2.0.0-weights/Dataset295_TotalSegmentator_part5_ribs_1559subj.zip", False),
         "M297" : (UnetCPP_2(nb_class=118), 2, url+"v2.0.4-weights/Dataset297_TotalSegmentator_total_3mm_1559subj_v204.zip", False),
         "M298" : (UnetCPP_2(nb_class=118), 2, url+"v2.0.0-weights/Dataset298_TotalSegmentator_total_6mm_1559subj.zip", False),
         "M730" : (UnetCPP_1(nb_class=30, mri = True), 1, url+"v2.2.0-weights/Dataset730_TotalSegmentatorMRI_part1_organs_495subj.zip", True),
         "M731" : (UnetCPP_1(nb_class=28, mri = True), 1, url+"v2.2.0-weights/Dataset731_TotalSegmentatorMRI_part2_muscles_495subj.zip", True),
         "M732" : (UnetCPP_2(nb_class=57, mri = True), 2, url+"v2.2.0-weights/Dataset732_TotalSegmentatorMRI_total_3mm_495subj.zip", True),
         "M733" : (UnetCPP_3(nb_class=57, mri = True), 3, url+"v2.2.0-weights/Dataset733_TotalSegmentatorMRI_total_6mm_495subj.zip", True),
         "M850" : (UnetCPP_1(nb_class=30, mri = True), 1, url+"v2.5.0-weights/Dataset850_TotalSegMRI_part1_organs_1088subj.zip", True),
         "M851" : (UnetCPP_1(nb_class=22, mri = True), 1, url+"v2.5.0-weights/Dataset851_TotalSegMRI_part2_muscles_1088subj.zip", True),
         "M852" : (UnetCPP_2(nb_class=51, mri = True), 2, url+"v2.5.0-weights/Dataset852_TotalSegMRI_total_3mm_1088subj.zip", True),
         "M853" : (UnetCPP_3(nb_class=51, mri = True), 3, url+"v2.5.0-weights/Dataset853_TotalSegMRI_total_6mm_1088subj.zip", True)
         }

if __name__ == "__main__":
    for name, model in models.items():
        convert_torchScript_full(name, model[0], model[1], model[2], model[3])


