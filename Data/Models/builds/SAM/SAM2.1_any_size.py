"""Re-export the SAM 2.1 IMPACT models (Hiera trunk) so they take a 2D slice of any size.

    python SAM2.1.py && python SAM2.1_any_size.py   ->  SAM2.1_Tiny.pt, SAM2.1_Small.pt (in the current directory),
                                                       rebuilt from the files SAM2.1.py writes there

forward(x [B, 3, H, W], nb_layers [1], stats [4] = [min, max, mean, std], direction) -> List[Tensor [B, C, h, w]]
(the 4 stage outputs of the Hiera trunk, strides 4, 8, 16, 32, the first nb_layers of them), as before.

The old files were scripted from facebookresearch/sam2 (Hiera._get_pos_embed, MultiScaleBlock). Two things tied them
to a few sizes:
  1. _get_pos_embed tiles the 8 x 8 window position embedding h // 8 x w // 8 times over the h x w patch grid
     (h = ceil(H / 4)): the sum fails unless h and w are multiples of 8 (H, W in 29..32, 61..64, ...).
  2. The shortcut of a stage's first block is a 2 x 2 max pool with ceil_mode False: a 1 x 1 grid (H <= 16 at the
     last stage) pools to 0 x 0.
Here the window embedding is tiled ceil(h / 8) x ceil(w / 8) times and cropped to h x w (the windows are anchored at
the top-left, as window_partition pads at the bottom-right, so every token keeps the embedding of its place in its
window), and the shortcut pools with ceil_mode True, which only differs on odd grids, where the old model failed.
Everything else is the old scripted modules, unchanged and called as the old forward called them: the patch embed,
the norms, attention, MLP and projections of every block, and the intensity normalisation (StandardizeImageNet).
So at every size the old model accepted, the output is the same computation.
"""
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F



def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [0, 0, 0, pad_w, 0, pad_h])
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C), (Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]) -> torch.Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


class Block(torch.nn.Module):
    """sam2 MultiScaleBlock.forward over the old block's scripted children; only the shortcut pool is ceil_mode."""

    def __init__(self, old: torch.nn.Module) -> None:
        super().__init__()
        self.window_size = int(old.window_size)
        self.pooled = old.dim != old.dim_out
        self.q_stride = int(old.q_stride[0]) if self.pooled else 0
        self.norm1, self.attn, self.norm2, self.mlp = old.norm1, old.attn, old.norm2, old.mlp
        self.proj = old.proj if self.pooled else torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.pooled:
            shortcut = self.proj(x).permute(0, 3, 1, 2)
            shortcut = F.max_pool2d(shortcut, self.q_stride, self.q_stride, ceil_mode=True).permute(0, 2, 3, 1)
        window_size = self.window_size
        H, W = x.shape[1], x.shape[2]
        pad_hw = (0, 0)
        if window_size > 0:
            x, pad_hw = window_partition(x, window_size)
        x = self.attn(x)
        if self.pooled:
            window_size = self.window_size // self.q_stride
            H, W = shortcut.shape[1], shortcut.shape[2]
            pad_hw = (H + (window_size - H % window_size) % window_size, W + (window_size - W % window_size) % window_size)
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class SAM(torch.nn.Module):
    def __init__(self, old: torch.nn.Module) -> None:
        super().__init__()
        hiera = old.model
        self.standardizeImageNet = old.standardizeImageNet
        self.patch_embed = hiera.patch_embed
        self.pos_embed = hiera.pos_embed
        self.pos_embed_window = hiera.pos_embed_window
        blocks = list(hiera.blocks.children())
        ends = list(hiera.stage_ends)
        self.outputs = [i == ends[-1] or (i in ends and bool(hiera.return_interm_layers)) for i in range(len(blocks))]
        self.blocks = torch.nn.ModuleList([Block(b) for b in blocks])

    def _get_pos_embed(self, h: int, w: int) -> torch.Tensor:
        pos_embed = F.interpolate(self.pos_embed, size=[h, w], mode="bicubic")
        wh, ww = self.pos_embed_window.shape[2], self.pos_embed_window.shape[3]
        window = self.pos_embed_window.tile([1, 1, (h + wh - 1) // wh, (w + ww - 1) // ww])[:, :, :h, :w]
        return (pos_embed + window).permute(0, 2, 3, 1)

    def forward(
        self,
        x: torch.Tensor,
        nb_layers_tensor: torch.Tensor = torch.tensor([1]),
        stats: torch.Tensor = torch.tensor([]),
        direction: torch.Tensor = torch.tensor([]),
    ) -> List[torch.Tensor]:
        x = self.standardizeImageNet(x, stats)
        nb_layers = int(nb_layers_tensor.item())
        x = self.patch_embed(x)
        x = x + self._get_pos_embed(x.shape[1], x.shape[2])
        outputs: List[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.outputs[i]:
                outputs.append(x.permute(0, 3, 1, 2))
                if len(outputs) == nb_layers:  # a return, not a break: TorchScript unrolls a ModuleList loop
                    return outputs
        return outputs


def export(name: str) -> None:
    old = torch.jit.load(f"{name}.pt").eval()
    torch.jit.script(SAM(old).eval()).save(f"{name}.pt")
    print(f"{name}.pt written")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["SAM2.1_Tiny", "SAM2.1_Small"]:
        export(name)
