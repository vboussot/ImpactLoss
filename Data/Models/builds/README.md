### 🔧 Model Generation from Source

This folder contains the scripts used to generate the TorchScript-exported models from their original sources.

These scripts reproduce the models provided at:
🔗 https://huggingface.co/VBoussot/impact-torchscript-models

Use them if you want to regenerate the models yourself from the official sources.

### ONNX exports

`ONNX/export_onnx.py` exports a TorchScript model of the repository to ONNX at one depth, for runtimes that do not
read TorchScript. A `.pt` takes its depth as an argument (`nb_layers`); an ONNX graph cannot branch on it without
reading it back to the host, so there is one graph per model and depth, named `<model>_depth<n>.onnx` and stored
next to its `.pt` with a `.json` of metadata:

```bash
python ONNX/export_onnx.py TS/M730.pt --layers 7   # -> onnx-out/TS/M730_depth7.onnx and .json
```

The script's docstring gives the graph contract (inputs `x` and `stats`, one output per layer the `.pt` returns at
that depth, called with no direction), the metadata, and the checks it runs against the `.pt` before writing the file.
The graphs it produced are hosted beside the `.pt` files, at the same Hugging Face repository.
