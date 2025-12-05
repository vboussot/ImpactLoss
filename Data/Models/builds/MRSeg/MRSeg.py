import torch
import requests
from tqdm import tqdm
import zipfile
import shutil
from pathlib import Path
import os

class ConvBlock(torch.nn.Module):
    
    def __init__(self, in_channels : int, out_channels : int, downsample: bool = False, last: bool = False) -> None:
        super().__init__()
        self.Conv_0 = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=(2 if downsample else 1) if not last else ((1,2,2) if downsample else 1), padding=1, bias=True) 
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


def crop_to_match(tensor, target_tensor):
    diff_x = tensor.size(2) - target_tensor.size(2)
    diff_y = tensor.size(3) - target_tensor.size(3)
    diff_z = tensor.size(4) - target_tensor.size(4)
    return tensor[:, :, torch.ceil(diff_x / 2) : tensor.size(2)-(diff_x // 2), torch.ceil(diff_y / 2) : tensor.size(3)-(diff_y // 2), torch.ceil(diff_z / 2) : tensor.size(4)-(diff_z // 2)]

class UnetCPP_1(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.DownConvBlock_0 = ConvBlock(in_channels=1, out_channels=32)

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        output0 = self.DownConvBlock_0(input)
        return [output0]

class UnetCPP_2(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.DownConvBlock_0 = ConvBlock(in_channels=1, out_channels=32)
        self.DownConvBlock_1 = ConvBlock(in_channels=32, out_channels=64, downsample=True)

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        output0 = self.DownConvBlock_0(input)
        output1 = self.DownConvBlock_1(output0)

        return [output0, output1]

class UnetCPP_3(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.DownConvBlock_0 = ConvBlock(in_channels=1, out_channels=32)
        self.DownConvBlock_1 = ConvBlock(in_channels=32, out_channels=64, downsample=True)
        self.DownConvBlock_2 = ConvBlock(in_channels=64, out_channels=128, downsample=True)

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        output0 = self.DownConvBlock_0(input)
        output1 = self.DownConvBlock_1(output0)
        output2 = self.DownConvBlock_2(output1)

        return [output0, output1, output2]

class UnetCPP_4(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.DownConvBlock_0 = ConvBlock(in_channels=1, out_channels=32)
        self.DownConvBlock_1 = ConvBlock(in_channels=32, out_channels=64, downsample=True)
        self.DownConvBlock_2 = ConvBlock(in_channels=64, out_channels=128, downsample=True)
        self.DownConvBlock_3 = ConvBlock(in_channels=128, out_channels=256, downsample=True)

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        output0 = self.DownConvBlock_0(input)
        output1 = self.DownConvBlock_1(output0)
        output2 = self.DownConvBlock_2(output1)
        output3 = self.DownConvBlock_3(output2)

        return [output0, output1, output2, output3]

class UnetCPP_5(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.DownConvBlock_0 = ConvBlock(in_channels=1, out_channels=32)
        self.DownConvBlock_1 = ConvBlock(in_channels=32, out_channels=64, downsample=True)
        self.DownConvBlock_2 = ConvBlock(in_channels=64, out_channels=128, downsample=True)
        self.DownConvBlock_3 = ConvBlock(in_channels=128, out_channels=256, downsample=True)
        self.DownConvBlock_4 = ConvBlock(in_channels=256, out_channels=320, downsample=True)

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        output0 = self.DownConvBlock_0(input)
        output1 = self.DownConvBlock_1(output0)
        output2 = self.DownConvBlock_2(output1)
        output3 = self.DownConvBlock_3(output2)
        output4 = self.DownConvBlock_4(output3)

        return [output0, output1, output2, output3, output4]
    
class UnetCPP_6(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.DownConvBlock_0 = ConvBlock(in_channels=1, out_channels=32)
        self.DownConvBlock_1 = ConvBlock(in_channels=32, out_channels=64, downsample=True)
        self.DownConvBlock_2 = ConvBlock(in_channels=64, out_channels=128, downsample=True)
        self.DownConvBlock_3 = ConvBlock(in_channels=128, out_channels=256, downsample=True)
        self.DownConvBlock_4 = ConvBlock(in_channels=256, out_channels=320, downsample=True)
        self.DownConvBlock_5 = ConvBlock(in_channels=320, out_channels=320, downsample=True, last=True)

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        output0 = self.DownConvBlock_0(input)
        output1 = self.DownConvBlock_1(output0)
        output2 = self.DownConvBlock_2(output1)
        output3 = self.DownConvBlock_3(output2)
        output4 = self.DownConvBlock_4(output3)
        output5 = self.DownConvBlock_5(output4)
        return [output0, output1, output2, output3, output4, output5]

class UnetCPP_7(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.DownConvBlock_0 = ConvBlock(in_channels=1, out_channels=32)
        self.DownConvBlock_1 = ConvBlock(in_channels=32, out_channels=64, downsample=True)
        self.DownConvBlock_2 = ConvBlock(in_channels=64, out_channels=128, downsample=True)
        self.DownConvBlock_3 = ConvBlock(in_channels=128, out_channels=256, downsample=True)
        self.DownConvBlock_4 = ConvBlock(in_channels=256, out_channels=320, downsample=True)
        self.DownConvBlock_5 = ConvBlock(in_channels=320, out_channels=320, downsample=True, last=True)

        self.Upsample_4 = torch.nn.ConvTranspose3d(in_channels=320, out_channels=320, kernel_size=(1,2,2), stride=(1,2,2))        
        self.UpConvBlock_4 = ConvBlock(in_channels=320*2, out_channels=320)
        
        self.Upsample_3 = torch.nn.ConvTranspose3d(in_channels=320, out_channels=256, kernel_size=2, stride=2)
        self.UpConvBlock_3 = ConvBlock(in_channels=256*2, out_channels=256)
        
        self.Upsample_2 = torch.nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.UpConvBlock_2 = ConvBlock(in_channels=128*2, out_channels=128)
        
        self.Upsample_1 = torch.nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.UpConvBlock_1 = ConvBlock(in_channels=64*2, out_channels=64)
        
        self.Upsample_0 = torch.nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2)        
        self.UpConvBlock_0 = ConvBlock(in_channels=32*2, out_channels=32)

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        output0 = self.DownConvBlock_0(input)
        output1 = self.DownConvBlock_1(output0)
        output2 = self.DownConvBlock_2(output1)
        output3 = self.DownConvBlock_3(output2)
        output4 = self.DownConvBlock_4(output3)
        output5 = self.DownConvBlock_5(output4)
        
        output = self.Upsample_4(output5)
        output = self.UpConvBlock_4(torch.cat([crop_to_match(output, output4), output4], dim=1))
        
        output = self.Upsample_3(output)
        output = self.UpConvBlock_3(torch.cat([crop_to_match(output, output3), output3], dim=1))

        output = self.Upsample_2(output)
        output = self.UpConvBlock_2(torch.cat([crop_to_match(output, output2), output2], dim=1))

        output = self.Upsample_1(output)
        output = self.UpConvBlock_1(torch.cat([crop_to_match(output, output1), output1], dim=1))

        output = self.Upsample_0(output)
        output = self.UpConvBlock_0(torch.cat([crop_to_match(output, output0), output0], dim=1))

        return [output0, output1, output2, output3, output4, output5, output]
    
class UnetCPP_8(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.DownConvBlock_0 = ConvBlock(in_channels=1, out_channels=32)
        self.DownConvBlock_1 = ConvBlock(in_channels=32, out_channels=64, downsample=True)
        self.DownConvBlock_2 = ConvBlock(in_channels=64, out_channels=128, downsample=True)
        self.DownConvBlock_3 = ConvBlock(in_channels=128, out_channels=256, downsample=True)
        self.DownConvBlock_4 = ConvBlock(in_channels=256, out_channels=320, downsample=True)
        self.DownConvBlock_5 = ConvBlock(in_channels=320, out_channels=320, downsample=True, last=True)


        self.Upsample_4 = torch.nn.ConvTranspose3d(in_channels=320, out_channels=320, kernel_size=(1,2,2), stride=(1,2,2))        
        
        self.UpConvBlock_4 = ConvBlock(in_channels=320*2, out_channels=320)
        
        self.Upsample_3 = torch.nn.ConvTranspose3d(in_channels=320, out_channels=256, kernel_size=2, stride=2)
        self.UpConvBlock_3 = ConvBlock(in_channels=256*2, out_channels=256)
        
        self.Upsample_2 = torch.nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.UpConvBlock_2 = ConvBlock(in_channels=128*2, out_channels=128)
        
        self.Upsample_1 = torch.nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.UpConvBlock_1 = ConvBlock(in_channels=64*2, out_channels=64)
        
        self.Upsample_0 = torch.nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2)        
        self.UpConvBlock_0 = ConvBlock(in_channels=32*2, out_channels=32)
        self.Head_0 = Head(32, 41)

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        output0 = self.DownConvBlock_0(input)
        output1 = self.DownConvBlock_1(output0)
        output2 = self.DownConvBlock_2(output1)
        output3 = self.DownConvBlock_3(output2)
        output4 = self.DownConvBlock_4(output3)
        output5 = self.DownConvBlock_5(output4)
        
        output = self.Upsample_4(output5)
        output = self.UpConvBlock_4(torch.cat([crop_to_match(output, output4), output4], dim=1))
        
        output = self.Upsample_3(output)
        output = self.UpConvBlock_3(torch.cat([crop_to_match(output, output3), output3], dim=1))

        output = self.Upsample_2(output)
        output = self.UpConvBlock_2(torch.cat([crop_to_match(output, output2), output2], dim=1))

        output = self.Upsample_1(output)
        output = self.UpConvBlock_1(torch.cat([crop_to_match(output, output1), output1], dim=1))

        output = self.Upsample_0(output)
        output = self.UpConvBlock_0(torch.cat([crop_to_match(output, output0), output0], dim=1))

        return [output0, output1, output2, output3, output4, output5, output, self.Head_0(output)]


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

def convert_torchScript_full(model_name: str, model: torch.nn.Module, type: int, url: str):
    state_dict = download(url)
    tmp = {}

    model_state_dict_key = model.state_dict().keys()
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
            if l1.replace("\n", "") not in model_state_dict_key:
                break
            tmp[l1.replace("\n", "")] = state_dict[key]

    model.load_state_dict(tmp)

    sm = torch.jit.script(model)
    sm.save(model_name+".pt")

url = "https://github.com/hhaentze/MRSegmentator/releases/download/"
       
models = {"MRSeg" : ([UnetCPP_1(), UnetCPP_2(), UnetCPP_3(), UnetCPP_4(), UnetCPP_5(), UnetCPP_6(), UnetCPP_7(), UnetCPP_8()], url+"v1.2.0/weights.zip")}

if __name__ == "__main__":
    for name, setup in models.items():
        models, weight_url = setup
        for l, model in enumerate(models):
            convert_torchScript_full(f"{name}_{l+1}_Layers", model, 1, weight_url)