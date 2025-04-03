## 🧠 Pretrained Models

The `/Data/Models/` directory must contain the **TorchScript-exported feature extractors** used by the IMPACT similarity metric.

---

### 📥 Download the Available TorchScript-Exported Feature Extractors

We provide a collection of TorchScript-exported feature extractors derived from publicly available pretrained segmentation networks, including:

- **TotalSegmentator**
- **Segment Anything (SAM2.1, MedSAM)**
- **DINOv2**
- **Anatomix**
- **MIND** (handcrafted features)

Each model is available in multiple versions, corresponding to different feature extraction layers (e.g., `_1_Layers.pt`, `_2_Layers.pt`, etc.), allowing fine-grained control over the level of semantic abstraction used.

You can selectively download individual models directly from Hugging Face:

🔗 [https://huggingface.co/VBoussot/impact-torchscript-models](https://huggingface.co/VBoussot/impact-torchscript-models)

Or you can download all models at once using the script included in this folder:

```bash
sh download_models.sh
```

This will download and organize all models into subfolders (`TS/`, `SAM2.1/`, `MIND/`, etc.).

---

### 🧾 Example Usage in Elastix

Once the models are in place, you can reference them in your Elastix parameter map like this:

```txt
(ModelsPath "/Data/Models/SAM2.1/Tiny_1_Layers.pt")
```

## 🧪 Create Your Own Pretrained Model

You can also define and export your own feature extractor, as long as it complies with the expected **TorchScript format**.

The **only constraint** is:

- **Input**: a 4D (for 2D models) or 5D (for 3D models) `torch.Tensor`  
- **Output**: a **list of feature maps**, i.e. one `torch.Tensor` per extracted layer  
  → e.g., `return [layer_0, layer_1]`

Any architecture is allowed, CNNs, transformers, or hybrid models, as long as this input/output format is respected.
⚠️ Note: To use a model in Jacobian mode, it must be fully differentiable to support gradient backpropagation.

### 🧑‍💻 Example

If you have a trained model, you can convert it to TorchScript format like this:

```python
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = torch.nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.relu_0 = torch.nn.ReLU()
        self.conv_1 = torch.nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.relu_1 = torch.nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        layer_0 = self.relu_0(self.conv_0(x))
        layer_1 = self.relu_1(self.conv_1(layer_0))
        return [layer_0, layer_1]

# Load pretrained weights
model = Model()  # Example of a 3D model with 2 extracted layers
model.load_state_dict(torch.load("Model_weights.pt"), strict=True)

# Create TorchScript model
example_input = torch.zeros((1, 1, 5, 5, 5))  # Input size matching the receptive field of the model
scripted_model = torch.jit.trace(model, example_input)
scripted_model.save("Data/Models/CustomModel/Custom_Model_Torchscript.pt")
```

Once saved, your model can be used with IMPACT by referencing it in your Elastix configuration:

```txt
(ModelsPath "Data/Models/CustomModel/Custom_Model_Torchscript.pt")
```

✅ Don’t forget to set `PatchSize`, `Dimension`, `NumberOfChannels`, and `LayersMask` based on the characteristics of your custom model in your `ParameterMap.txt`.