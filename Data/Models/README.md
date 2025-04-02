## 🧠 Download Models

The `/Data/Models/` directory must contain the **TorchScript-exported feature extractors** used by the IMPACT similarity metric.

These models are derived from publicly available pretrained networks such as:

- **TotalSegmentator**
- **Segment Anything (SAM2.1, MedSAM)**
- **DINOv2**
- **Anatomix**
- **MIND** (handcrafted features)

Each model is available in multiple versions, corresponding to different feature extraction layers (e.g., `_1_Layers.pt`, `_2_Layers.pt`, etc.), allowing fine-grained control over the level of semantic abstraction used.

---

### 📥 Download All Models

You can download all compatible models directly from Hugging Face:

🔗 [https://huggingface.co/VBoussot/impact-torchscript-models](https://huggingface.co/VBoussot/impact-torchscript-models)

To automate the process, use the following script provided in this folder:

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

For multi-model setups or layer-specific configuration, refer to the main documentation or the `ParameterMaps/` examples.
