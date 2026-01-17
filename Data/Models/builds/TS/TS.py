import torch
import requests
from tqdm import tqdm
import zipfile
import shutil
from pathlib import Path
import json
import os

class ConvBlock(torch.nn.Module):
    
    def __init__(self, in_channels : int, out_channels : int, downsample: bool = False, mri: bool = False, lung: bool = False) -> None:
        super().__init__()
        if not lung:
            self.Conv_0 = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=(2 if downsample else 1) if not mri else ((1,2,2) if downsample else 1), padding=1, bias=True)
        else:
            self.Conv_0 = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=(2,1,2), padding=1, bias=True)
         
        self.Norm_0 = torch.nn.InstanceNorm3d(num_features=out_channels, affine=True)
        self.Activation_0 = torch.nn.LeakyReLU(negative_slope=0.01)
        self.Conv_1 = torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.Norm_1 = torch.nn.InstanceNorm3d(num_features=out_channels, affine=True)
        self.Activation_1 = torch.nn.LeakyReLU(negative_slope=0.01)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.Conv_0(input)
        output = self.Norm_0(output)
        output = self.Activation_0(output)
        output = self.Conv_1(output)
        output = self.Norm_1(output)
        output = self.Activation_1(output)
        return output

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
        self.register_buffer("clip_min", torch.empty(1))
        self.register_buffer("clip_max", torch.empty(1))
        self.register_buffer("mean", torch.empty(1))
        self.register_buffer("std", torch.empty(1))

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, self.clip_min, self.clip_max)
        return (x - self.mean) / (self.std)


class Standardize(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        if stats.shape[0] == 4:
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
        if direction.dim() != 2 or direction.shape[0] != 3 or direction.shape[1] != 3: 
            return None
        
        A = torch.diag(torch.tensor([-1., -1., 1.], dtype=torch.double)) @ direction.to(torch.double).T
        
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
    
class UnetCPP_1(torch.nn.Module):

    def __init__(self, nb_class: int, mri: bool = False, lung: bool = False) -> None:
        super().__init__()
        if mri:
            self.normalize  = Standardize()
        else:
            self.normalize = ClipAndNormalize()
        self.canonical = Canonical()

        self.DownConvBlock_0 = ConvBlock(in_channels=1, out_channels=32)
        self.DownConvBlock_1 = ConvBlock(in_channels=32, out_channels=64, downsample=True)
        self.DownConvBlock_2 = ConvBlock(in_channels=64, out_channels=128, downsample=True)
        self.DownConvBlock_3 = ConvBlock(in_channels=128, out_channels=256, downsample=True)
        self.DownConvBlock_4 = ConvBlock(in_channels=256, out_channels=320, downsample=True)
        self.DownConvBlock_5 = ConvBlock(in_channels=320, out_channels=320, downsample=True, mri=mri, lung = lung)

        if lung:
            self.Upsample_4 = torch.nn.ConvTranspose3d(in_channels=320, out_channels=320, kernel_size=(2,1,2), stride=(2,1,2))
        else:
            self.Upsample_4 = torch.nn.ConvTranspose3d(in_channels=320, out_channels=320, kernel_size=(1,2,2) if mri else 2, stride=(1,2,2) if mri else 2)        
        
        self.UpConvBlock_4 = ConvBlock(in_channels=320*2, out_channels=320)
        
        self.Upsample_3 = torch.nn.ConvTranspose3d(in_channels=320, out_channels=256, kernel_size=2, stride=2)
        self.UpConvBlock_3 = ConvBlock(in_channels=256*2, out_channels=256)
        
        self.Upsample_2 = torch.nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.UpConvBlock_2 = ConvBlock(in_channels=128*2, out_channels=128)
        
        self.Upsample_1 = torch.nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.UpConvBlock_1 = ConvBlock(in_channels=64*2, out_channels=64)
        
        self.Upsample_0 = torch.nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2)        
        self.UpConvBlock_0 = ConvBlock(in_channels=32*2, out_channels=32)
        self.Head_0 = Head(32, nb_class)

    def forward(self, x: torch.Tensor, nb_layers_tensor: torch.Tensor = torch.tensor([8]), stats: torch.Tensor = torch.tensor([]), direction: torch.Tensor = torch.tensor([])) -> list[torch.Tensor]:
        x = self.normalize(x, stats)
        nb_layers = nb_layers_tensor.item()

        permute_and_flip = self.canonical.get_permute_and_flip(direction)
        inverse_permute_and_flip = self.canonical.get_inverse_permute_and_flip(permute_and_flip)

        x = self.canonical(x, permute_and_flip)

        output_0 = self.DownConvBlock_0(x)
        if nb_layers == 1:
            return [self.canonical.inverse(output_0, inverse_permute_and_flip)]
        output_1 = self.DownConvBlock_1(output_0)
        if nb_layers == 2:
            return [self.canonical.inverse(output_0, inverse_permute_and_flip), self.canonical.inverse(output_1, inverse_permute_and_flip)]
        output_2 = self.DownConvBlock_2(output_1)
        if nb_layers == 3:
            return [self.canonical.inverse(output_0, inverse_permute_and_flip), self.canonical.inverse(output_1, inverse_permute_and_flip), self.canonical.inverse(output_2, inverse_permute_and_flip)]
        output_3 = self.DownConvBlock_3(output_2)
        if nb_layers == 4:
            return [self.canonical.inverse(output_0, inverse_permute_and_flip), self.canonical.inverse(output_1, inverse_permute_and_flip), self.canonical.inverse(output_2, inverse_permute_and_flip), self.canonical.inverse(output_3, inverse_permute_and_flip)]
        output_4 = self.DownConvBlock_4(output_3)
        if nb_layers == 5:
            return [self.canonical.inverse(output_0, inverse_permute_and_flip), self.canonical.inverse(output_1, inverse_permute_and_flip), self.canonical.inverse(output_2, inverse_permute_and_flip), self.canonical.inverse(output_3, inverse_permute_and_flip), self.canonical.inverse(output_4, inverse_permute_and_flip)]
        output_5 = self.DownConvBlock_5(output_4)
        if nb_layers == 6:
            return [self.canonical.inverse(output_0, inverse_permute_and_flip), self.canonical.inverse(output_1, inverse_permute_and_flip), self.canonical.inverse(output_2, inverse_permute_and_flip), self.canonical.inverse(output_3, inverse_permute_and_flip), self.canonical.inverse(output_4, inverse_permute_and_flip), self.canonical.inverse(output_5, inverse_permute_and_flip)]
        
        output = self.Upsample_4(output_5)
        output = self.UpConvBlock_4(torch.cat([crop_to_match(output, output_4), output_4], dim=1))
        
        output = self.Upsample_3(output)
        output = self.UpConvBlock_3(torch.cat([crop_to_match(output, output_3), output_3], dim=1))

        output = self.Upsample_2(output)
        output = self.UpConvBlock_2(torch.cat([crop_to_match(output, output_2), output_2], dim=1))

        output = self.Upsample_1(output)
        output = self.UpConvBlock_1(torch.cat([crop_to_match(output, output_1), output_1], dim=1))

        output = self.Upsample_0(output)
        output = self.UpConvBlock_0(torch.cat([crop_to_match(output, output_0), output_0], dim=1))

        if nb_layers == 7:            
            return [self.canonical.inverse(output_0, inverse_permute_and_flip), self.canonical.inverse(output_1, inverse_permute_and_flip), self.canonical.inverse(output_2, inverse_permute_and_flip), self.canonical.inverse(output_3, inverse_permute_and_flip), self.canonical.inverse(output_4, inverse_permute_and_flip), self.canonical.inverse(output_5, inverse_permute_and_flip), self.canonical.inverse(output, inverse_permute_and_flip)]    
        return [self.canonical.inverse(output_0, inverse_permute_and_flip), self.canonical.inverse(output_1, inverse_permute_and_flip), self.canonical.inverse(output_2, inverse_permute_and_flip), self.canonical.inverse(output_3, inverse_permute_and_flip), self.canonical.inverse(output_4, inverse_permute_and_flip), self.canonical.inverse(output_5, inverse_permute_and_flip), self.canonical.inverse(output, inverse_permute_and_flip), self.canonical.inverse(self.Head_0(output), inverse_permute_and_flip)]

class UnetCPP_2(torch.nn.Module):

    def __init__(self, nb_class: int, mri: bool = False) -> None:
        super().__init__()
        if mri:
            self.normalize  = Standardize()
        else:
            self.normalize = ClipAndNormalize()
        self.canonical = Canonical()

        self.DownConvBlock_0 = ConvBlock(in_channels=1, out_channels=32)
        self.DownConvBlock_1 = ConvBlock(in_channels=32, out_channels=64, downsample=True)
        self.DownConvBlock_2 = ConvBlock(in_channels=64, out_channels=128, downsample=True)
        self.DownConvBlock_3 = ConvBlock(in_channels=128, out_channels=256, downsample=True)
        self.DownConvBlock_4 = ConvBlock(in_channels=256, out_channels=320, downsample=True)
        
        self.Upsample_3 = torch.nn.ConvTranspose3d(in_channels=320, out_channels=256, kernel_size=2, stride=2)
        self.UpConvBlock_3 = ConvBlock(in_channels=256*2, out_channels=256)

        self.Upsample_2 = torch.nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.UpConvBlock_2 = ConvBlock(in_channels=128*2, out_channels=128)

        self.Upsample_1 = torch.nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.UpConvBlock_1 = ConvBlock(in_channels=64*2, out_channels=64)

        self.Upsample_0 = torch.nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2)        
        self.UpConvBlock_0 = ConvBlock(in_channels=32*2, out_channels=32)
        self.Head_0 = Head(32, nb_class)

    def forward(self, x: torch.Tensor, nb_layers_tensor: torch.Tensor = torch.tensor([7]), stats: torch.Tensor = torch.tensor([]), direction: torch.Tensor = torch.tensor([])) -> list[torch.Tensor]:
        x = self.normalize(x, stats)
        nb_layers = nb_layers_tensor.item()
        permute_and_flip = self.canonical.get_permute_and_flip(direction)
        inverse_permute_and_flip = self.canonical.get_inverse_permute_and_flip(permute_and_flip)

        x = self.canonical(x, permute_and_flip)

        output_0 = self.DownConvBlock_0(x)
        if nb_layers == 1:
            return [self.canonical.inverse(output_0, inverse_permute_and_flip)]
        output_1 = self.DownConvBlock_1(output_0)
        if nb_layers == 2:
            return [self.canonical.inverse(output_0, inverse_permute_and_flip), self.canonical.inverse(output_1, inverse_permute_and_flip)]
        output_2 = self.DownConvBlock_2(output_1)
        if nb_layers == 3:
            return [self.canonical.inverse(output_0, inverse_permute_and_flip), self.canonical.inverse(output_1, inverse_permute_and_flip), self.canonical.inverse(output_2, inverse_permute_and_flip)]
        output_3 = self.DownConvBlock_3(output_2)
        if nb_layers == 4:
            return [self.canonical.inverse(output_0, inverse_permute_and_flip), self.canonical.inverse(output_1, inverse_permute_and_flip), self.canonical.inverse(output_2, inverse_permute_and_flip), self.canonical.inverse(output_3, inverse_permute_and_flip)]
        output_4 = self.DownConvBlock_4(output_3)
        if nb_layers == 5:
            return [self.canonical.inverse(output_0, inverse_permute_and_flip), self.canonical.inverse(output_1, inverse_permute_and_flip), self.canonical.inverse(output_2, inverse_permute_and_flip), self.canonical.inverse(output_3, inverse_permute_and_flip), self.canonical.inverse(output_4, inverse_permute_and_flip)]
        
        output = self.Upsample_3(output_4)
        output = self.UpConvBlock_3(torch.cat([crop_to_match(output, output_3), output_3], dim=1))

        output = self.Upsample_2(output)
        output = self.UpConvBlock_2(torch.cat([crop_to_match(output, output_2), output_2], dim=1))

        output = self.Upsample_1(output)
        output = self.UpConvBlock_1(torch.cat([crop_to_match(output, output_1), output_1], dim=1))

        output = self.Upsample_0(output)
        output = self.UpConvBlock_0(torch.cat([crop_to_match(output, output_0), output_0], dim=1))

        if nb_layers == 6:            
            return [self.canonical.inverse(output_0, inverse_permute_and_flip), self.canonical.inverse(output_1, inverse_permute_and_flip), self.canonical.inverse(output_2, inverse_permute_and_flip), self.canonical.inverse(output_3, inverse_permute_and_flip), self.canonical.inverse(output_4, inverse_permute_and_flip), self.canonical.inverse(output, inverse_permute_and_flip)]    
        return [self.canonical.inverse(output_0, inverse_permute_and_flip), self.canonical.inverse(output_1, inverse_permute_and_flip), self.canonical.inverse(output_2, inverse_permute_and_flip), self.canonical.inverse(output_3, inverse_permute_and_flip), self.canonical.inverse(output_4, inverse_permute_and_flip), self.canonical.inverse(output, inverse_permute_and_flip), self.canonical.inverse(self.Head_0(output), inverse_permute_and_flip)]


class UnetCPP_3(torch.nn.Module):

    def __init__(self, nb_class: int, mri: bool = False) -> None:
        super().__init__()
        if mri:
            self.normalize  = Standardize()
        else:
            self.normalize = ClipAndNormalize()
        self.canonical = Canonical()
        self.DownConvBlock_0 = ConvBlock(in_channels=1, out_channels=32)
        self.DownConvBlock_1 = ConvBlock(in_channels=32, out_channels=64, downsample=True)
        self.DownConvBlock_2 = ConvBlock(in_channels=64, out_channels=128, downsample=True)
        self.DownConvBlock_3 = ConvBlock(in_channels=128, out_channels=256, downsample=True)
        
        self.Upsample_2 = torch.nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.UpConvBlock_2 = ConvBlock(in_channels=128*2, out_channels=128)

        
        self.Upsample_1 = torch.nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.UpConvBlock_1 = ConvBlock(in_channels=64*2, out_channels=64)

        self.Upsample_0 = torch.nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2)        
        self.UpConvBlock_0 = ConvBlock(in_channels=32*2, out_channels=32)
        self.Head_0 = Head(32, nb_class)

    def forward(self, x: torch.Tensor, nb_layers_tensor: torch.Tensor = torch.tensor([6]), stats: torch.Tensor = torch.tensor([]), direction: torch.Tensor = torch.tensor([])) -> list[torch.Tensor]:
        x = self.normalize(x, stats)
        nb_layers = nb_layers_tensor.item()

        permute_and_flip = self.canonical.get_permute_and_flip(direction)
        inverse_permute_and_flip = self.canonical.get_inverse_permute_and_flip(permute_and_flip)

        x = self.canonical(x, permute_and_flip)

        output_0 = self.DownConvBlock_0(x)
        if nb_layers == 1:
            return [self.canonical.inverse(output_0, inverse_permute_and_flip)]
        output_1 = self.DownConvBlock_1(output_0)
        if nb_layers == 2:
            return [self.canonical.inverse(output_0, inverse_permute_and_flip), self.canonical.inverse(output_1, inverse_permute_and_flip)]
        output_2 = self.DownConvBlock_2(output_1)
        if nb_layers == 3:
            return [self.canonical.inverse(output_0, inverse_permute_and_flip), self.canonical.inverse(output_1, inverse_permute_and_flip), self.canonical.inverse(output_2, inverse_permute_and_flip)]
        output_3 = self.DownConvBlock_3(output_2)
        if nb_layers == 4:
            return [self.canonical.inverse(output_0, inverse_permute_and_flip), self.canonical.inverse(output_1, inverse_permute_and_flip), self.canonical.inverse(output_2, inverse_permute_and_flip), self.canonical.inverse(output_3, inverse_permute_and_flip)]
        
        output = self.Upsample_2(output_3)
        output = self.UpConvBlock_2(torch.cat([crop_to_match(output, output_2), output_2], dim=1))

        output = self.Upsample_1(output)
        output = self.UpConvBlock_1(torch.cat([crop_to_match(output, output_1), output_1], dim=1))

        output = self.Upsample_0(output)
        output = self.UpConvBlock_0(torch.cat([crop_to_match(output, output_0), output_0], dim=1))

        if nb_layers == 5:            
            return [self.canonical.inverse(output_0, inverse_permute_and_flip), self.canonical.inverse(output_1, inverse_permute_and_flip), self.canonical.inverse(output_2, inverse_permute_and_flip), self.canonical.inverse(output_3, inverse_permute_and_flip), self.canonical.inverse(output, inverse_permute_and_flip)]    
        return [self.canonical.inverse(output_0, inverse_permute_and_flip), self.canonical.inverse(output_1, inverse_permute_and_flip), self.canonical.inverse(output_2, inverse_permute_and_flip), self.canonical.inverse(output_3, inverse_permute_and_flip), self.canonical.inverse(output, inverse_permute_and_flip), self.canonical.inverse(self.Head_0(output), inverse_permute_and_flip)]
    
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
            tmp[l1.replace("\n", "")] = state_dict[key]
    if not mri:
        tmp["normalize.clip_min"] = state_dict["clip_min"]
        tmp["normalize.clip_max"] = state_dict["clip_max"]
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


