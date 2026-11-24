"""Re-export the published TS models from their own weights with the current TS.py, which adds ``norms`` and
``norm_input`` (statistics of the instance normalisations taken from outside). Every layer of the new export is
checked against the published one, bit for bit, on a random volume, without a direction and with one that permutes
and flips the axes.

    python reexport.py PUBLISHED_TS_DIR OUT_DIR
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import TS  # noqa: E402  (the export module, imported as a module so that TorchScript can compile its functions)

argv = sys.argv

published, out = Path(argv[1]), Path(argv[2])
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
x = torch.randn(1, 1, 64, 48, 40) * 300
directions = [torch.tensor([]), torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])]
for name, (model, _, _, _) in TS.models.items():
    path = published / f"{name}.pt"
    if not path.exists():
        continue
    old = torch.jit.load(str(path), map_location="cpu").eval()
    model.load_state_dict({TS.new_key(key, model.depth): value for key, value in old.state_dict().items()})
    new = torch.jit.script(model.eval())
    with torch.no_grad():
        for direction in directions:
            for nb_layers in range(1, 2 * model.depth):
                a = old(x, torch.tensor([nb_layers]), torch.tensor([]), direction)
                b = new(x, torch.tensor([nb_layers]), torch.tensor([]), direction)
                assert len(a) == len(b) and all(torch.equal(u, v) for u, v in zip(a, b)), (name, nb_layers)
    new.save(str(out / f"{name}.pt"))
    print(f"{name}: depth {model.depth}, every layer identical to the published export", flush=True)
