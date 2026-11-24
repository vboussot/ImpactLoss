"""Re-export the published MRSeg model from its own weights with the current MRSeg.py (TS.Unet's ``norms`` and
``norm_input``). Every layer of the new export is checked against the published one, bit for bit, on a random volume,
without a direction and with one, which both ignore.

    python reexport.py PUBLISHED_MRSEG_PT OUT_PT
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import MRSeg  # noqa: E402
import TS  # noqa: E402  (put on the path by MRSeg)

published, out = Path(sys.argv[1]), Path(sys.argv[2])
torch.manual_seed(0)
x = torch.randn(1, 1, 64, 48, 40) * 300
directions = [torch.tensor([]), torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])]
old = torch.jit.load(str(published), map_location="cpu").eval()
model = MRSeg.UnetCPP(41)
model.load_state_dict({TS.new_key(key, model.depth): value for key, value in old.state_dict().items()})
new = torch.jit.script(model.eval())
with torch.no_grad():
    for direction in directions:
        for nb_layers in range(0, 2 * model.depth):
            a = old(x, torch.tensor([nb_layers]), torch.tensor([]), direction)
            b = new(x, torch.tensor([nb_layers]), torch.tensor([]), direction)
            assert len(a) == len(b) and all(torch.equal(u, v) for u, v in zip(a, b)), nb_layers
new.save(str(out))
print(f"{published.name}: every layer identical to the published export", flush=True)
