# 🔬 IMPACT: A Generic Semantic Loss for Multimodal Image Registration


[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/SuperElastix/elastix/raw/main/LICENSE)
[![Models](https://img.shields.io/badge/models-huggingface-orange)](https://huggingface.co/VBoussot/impact-torchscript-models)
[![Docker](https://img.shields.io/badge/docker-ready-blueviolet)](https://hub.docker.com/repository/docker/vboussot/elastix_impact)

**IMPACT** is a novel, task-agnostic similarity metric designed for **multimodal medical image registration**. Instead of relying on intensity based metric, handcrafted descriptors or training task-specific models, IMPACT reuses powerful segmentation foundation models (e.g., TotalSegmentator, SAM) as generic feature extractors. These deep features are used to define a semantic similarity loss, optimized directly in registration frameworks like Elastix or VoxelMorph.

> 🔗 IMPACT: A Generic Semantic Loss for Multimodal Image Registration 
> Valentin Boussot, Cédric Hémon, Jean-Claude Nunes, Jason Downling, Simon Rouzé, Caroline Lafond, Anaïs Barateau, Jean-Louis Dillenseger
> [arXiv:2503.24121](https://arxiv.org/abs/2503.24121) – _Under review_ 

---

## ✨ Key Features

- **Generic, Training-free**  
  No need for task-specific training, IMPACT reuses powerful representations from large-scale pretrained segmentation models.

- **Flexible model integration**  
  Compatible with TorchScript 2D/3D models (e.g., TotalSegmentator, SAM2.1, MIND), supporting multi-layer fusion and multi-resolution setups.

- **Jacobian vs Static optimization modes**  
  Choose between fully differentiable Jacobian mode (for downsampling models) and fast inference-only Static mode, depending on model type and computation time constraints.

- **Robust across modalities**  
  Handles complex multimodal scenarios (CT/CBCT, MR/CT) using a unified semantic loss that bypasses raw intensity mismatches.

- **Benchmark-proven**
  Ranked in the top participants of multiple Learn2Reg challenges, showing state-of-the-art performance across diverse tasks (thorax, abdomen, pelvis, CT/CBCT/MRI).

- **Seamless integration with Elastix**  
  Natively implemented as a standard Elastix metric, IMPACT inherits all the strengths of classical registration:  
  multi-resolution strategies, mask support, transformation priors, precise interpolation control, and full reproducibility.  
  It also handles **images of different sizes, resolutions, and fields of view**, 
  making it ideal for real-world clinical datasets with heterogeneous inputs.

- **Efficient runtime**  
  - ~150 seconds (Static mode)  
  - ~300 seconds (Jacobian mode)

- **Docker-ready for quick deployment**  
  Run out of the box with a single Docker command, no need to install dependencies manually.

---

## 🏆 Challenge Results

IMPACT has demonstrated strong generalization performance across multiple tasks without training.
🔗 [Learn2Reg Challenge](https://learn2reg.grand-challenge.org/)

| Challenge       | Task                           | Rank      
|----------------|--------------------------------|-----------
| **Learn2Reg 2021** | CT Lung Registration            | 🥉 3rd     
| **Learn2Reg 2023** | Thorax CBCT                    | 🥉 Top-6
| **Learn2Reg 2023** | Abdomen MR→CT  | 🥈 2nd 

---

> 🔍 **IMPACT redefines image registration as a semantic alignment task**, leveraging deep anatomical priors from pretrained models rather than relying on low-level pixel similarities. Beyond its practical effectiveness, IMPACT is also a modular research platform, enabling systematic experimentation across pretrained models, feature layers, patch sizes, loss functions, and sampling strategies.

## 🚀 Quick Start with Docker

You can quickly test the IMPACT metric using the provided Docker environment:

```bash
git clone https://github.com/vboussot/ImpactLoss.git
cd ImpactLoss
```

Build the Docker image
```bash
docker build -t elastix_impact Docker
```

Then, run Elastix with your own data:

```bash
docker run --rm --gpus all \
  -v "./Data:/Data" \
  -v "./Out:/Out" \
  elastix_impact
```

Make sure that the `Data/` folder contains:
- `FixedImage.mha`, `MovingImage.mha` your input images to be registered.
- `ParameterMap.txt` using Impact configuration. 👉 See [`ParameterMaps/README.md`](ParameterMaps/README.md) for detailed configuration examples.
- A `Data/Models/` directory with TorchScript models. 👉 See [`Data/Models/README.md`](Data/Models/README.md) for model download instructions.

See [`Docker/README.md`](Docker/README.md) for full details and usage examples.

---

## 🛠️ Manual Build Instructions (without Docker)

Build Elastix with IMPACT support directly on your machine.

### 📦 (Optional) Get LibTorch (if not already installed)

Download and extract the **C++ distribution** of LibTorch (with or without CUDA) from the official website:

🔗 https://pytorch.org/

### 📦 (Optional) Build ITK (if not already installed):

```bash
git clone https://github.com/InsightSoftwareConsortium/ITK.git
mkdir ITK-build ITK-install
cd ITK-build
cmake -DCMAKE_INSTALL_PREFIX=../ITK-install ../ITK
make install
cd ..
```

### 🧱 Build Instructions

1. Clone the [ImpactElastix](https://github.com/vboussot/ImpactElastix) repository:

```bash
git clone https://github.com/vboussot/ImpactElastix.git
```

2. Create build and install directories:

```bash
mkdir ImpactElastix-build ImpactElastix-install
cd ImpactElastix-build
```

3. Configure the build with CMake:

```bash
cmake -DTorch_DIR=../libtorch/share/cmake/Torch/ \
      -DITK_DIR=../ITK-install/lib/cmake/ITK-6.0/ \
      -DCMAKE_INSTALL_PREFIX=../ImpactElastix-install \
      ../ImpactElastix
```

- `Torch_DIR`: path to the **CMake config directory of LibTorch** (usually inside `libtorch/share/cmake/Torch/`)
- `ITK_DIR`: path to the **CMake config directory of ITK**, typically inside your ITK install folder (e.g., `ITK-install/lib/cmake/ITK-*`)

4. Build and install Elastix with IMPACT:

```bash
make install
```

The final binaries will be available in:

```bash
../ImpactElastix-install/bin/elastix
```

## ⚙️ Run Elastix

To use IMPACT, start by downloading the pretrained TorchScript models.  
👉 See [`Data/Models/README.md`](Data/Models/README.md) for download instructions.

Elastix is executed as usual, using a parameter map configured to use the IMPACT metric.  
👉 Refer to [`ParameterMaps/README.md`](ParameterMaps/README.md) for detailed configuration examples.

⚠️ **Preprocessing Recommendation**  
Input images must be **preprocessed consistently with the training of the selected model**.  
For TotalSegmentator-based models, images should be in **canonical orientation**.

Apply the appropriate preprocessing depending on the model:

- **ImageNet-based models** (e.g., SAM2.1, DINOv2):  
  - Normalize intensities to [0, 1]  
  - Then standardize with mean `0.485` and standard deviation `0.229` 

- **MRI models** (e.g., TS/M730–M733):  
  - Standardize intensities to zero mean and unit variance  

- **CT models** (e.g., all other TotalSegmentator variants, MIND):  
  - Clip intensities to `[-1024, 276]` HU  
  - Then normalize by centering at `-370 HU` and scaling by `436.6`

A complete example of how to run registration with IMPACT is provided in:  
👉 [`run_impact_example.py`](run_impact_example.py)
