# 🐳 Docker for Elastix + IMPACT Metric

This Docker environment builds a version of **Elastix** with support for the **IMPACT similarity metric**, enabling semantic image registration based on TorchScript-extracted features.

---

## ⚙️ Build the Docker Image

From the `docker/` directory, run:

```bash
docker build -t elastix_impact .
```

## 🚀 Run the Container

To execute a registration, run:

```bash
docker run --rm --gpus all \
  -v "./Data:/Data" \
  -v './Out:/Out" \
  elastix_impact
```

## 📁 Required Folder Structure

The container expects a specific folder structure inside the `/Data` directory:

- `/Data/Models/` must contain the TorchScript-exported models used by the IMPACT metric (e.g., TotalSegmentator, SAM2.1, etc.).
- `/Data/FixedImage.mha` is the fixed image used for registration.
- `/Data/MovingImage.mha` is the moving image that will be aligned.
- `/Data/ParameterMap.txt` is the Elastix parameter file, which must include the configuration for the Impact metric.

The following files are optional but can be included to enhance registration:

- `/Data/Fixed_mask.mha` and `/Data/Moving_mask.mha` can be used to restrict the metric computation to specific regions.
- `/Data/Fixed_landmarks.txt` and `/Data/Moving_landmarks.txt` can be used for evaluation (TRE).

All output files (transforms, logs) will be written to the `/Out` directory mounted into the container.

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

## 🧠 Models

The `./Data/Models/` directory must contain the TorchScript-exported feature extractors used by the IMPACT metric.
See [`Data/Models/README.md`](Data/Models/README.md) for full details.