## 🧠 Models

The `/Data/Models/` directory must contain the TorchScript-exported feature extractors used by the IMPACT metric.

These models are derived from publicly available pretrained networks such as TotalSegmentator, SAM2.1, DINOv2, Anatomix, or MIND. Each model is available in multiple versions corresponding to different feature extraction layers (e.g., `_1_Layers.pt`, `_2_Layers.pt`, etc.).

You can download compatible models directly from the Hugging Face repository:

🔗 https://huggingface.co/VBoussot/impact-torchscript-models

To automate this, use the following script provided in the `/Data/Models` directory:

```bash
sh download_models.sh
```

All models are exported in **TorchScript format** (`.pt`). You can reference them in your Elastix parameter map as follows:

```txt
(ModelsPath "/Data/Models/SAM2.1/Tiny_1_Layers.pt")
```
