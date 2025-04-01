# 🔬 IMPACT: A Generic Semantic Loss for Multimodal Image Registration

This repository provides the official implementation of **IMPACT** (Image Metric with Pretrained model-Agnostic Comparison for Transmodality registration), a generic similarity metric for multimodal medical image registration based on pretrained semantic features.

The method is described in the following paper:

> IMPACT: A Generic Semantic Loss for Multimodal Medical Image Registration [arXiv:2503.24121](https://arxiv.org/abs/2503.24121) – _Under review_  
> Valentin Boussot, Cédric Hémon, Jean-Claude Nunes, Jason Downling, Simon Rouzé, Caroline Lafond, Anaïs Barateau, Jean-Louis Dillenseger

---

## ✨ Key Features

- **Generic, training-free semantic similarity**  
  IMPACT leverages semantic features from large-scale pretrained segmentation models (e.g., TotalSegmentator, SAM) without any task-specific training.

- **TorchScript-based runtime feature extraction**  
  Compatible with any TorchScript model, enabling flexible integration of 2D/3D architectures and seamless GPU acceleration.

- **Multi-layer semantic fusion**  
  Supports combining features across multiple model layers with flexible configuration.

- **Jacobian and Static optimization modes**  
  Choose between gradient-based (Jacobian) or precomputed feature mode (Static) depending on speed/memory trade-offs.

- **Patch-based sampling with resolution-aware alignment**  
  Uses patch sampling strategies and resolution-specific settings (voxel size, patch size, etc.) for efficient and precise alignment.

- **Docker-ready for quick deployment**  
  A containerized version is available for instant testing, eliminating dependency hassles.

- **Benchmark-proven**  
  Ranked in the top participants of multiple Learn2Reg challenges, showing state-of-the-art performance across diverse tasks (thorax, abdomen, pelvis, CT/CBCT/MRI).

- **Efficient compute time**  
  Registration typically takes **~300 seconds per image pair** in *Jacobian* mode, and around **~150 seconds** in *Static* mode on GPU.

- **Weakly-supervised mask support**  
  Natively supports mask-based supervision for focusing on regions of interest during registration (e.g., lungs, organs).
---

## 🚀 Quick Start with Docker

You can quickly test the IMPACT metric using the provided Docker environment:

```bash
git clone https://github.com/vboussot/ImpactLoss.git
cd ImpactLoss/docker
```

Build the Docker image
```bash
docker build -t elastix_impact .
```

Then, run Elastix with your own data:

```bash
docker run --rm --gpus all \
  -v "./Data:/Data" \
  -v "./Out:/Out" \
  elastix_impact
```

Make sure that the `/Data` folder contains:
- `FixedImage.mha`, `MovingImage.mha`
- `ParameterMap.txt` using Impact configuration
- A `/Data/Models/` directory with TorchScript models

See [`docker/README.md`](docker/README.md) for full details and usage examples.

---

## 🛠️ Manual Build Instructions (without Docker)

To compile Elastix with the IMPACT metric on your own system, you will need:

- A CUDA-compatible GPU (for LibTorch GPU backend)
- CMake ≥ 3.18
- GCC ≥ 10 (recommended)

### ✅ Required dependencies

Install the following packages using your system’s package manager:

- `git`, `cmake`, `gcc`, `g++`, `make`, `cuda`

### 📦 (Optional) Get LibTorch (if not already installed):

Download and extract the C++ distribution of LibTorch:

```bash
wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu124.zip
unzip libtorch*.zip
```

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

5. Build and install Elastix with IMPACT:

```bash
make install
```

The final binaries will be available in:

```bash
../ImpactElastix-install/bin/elastix
```

## ⚙️ Run Elastix

Make sure that the `/Data` folder contains:
- `FixedImage.mha`, `MovingImage.mha`
- `ParameterMap.txt` using Impact configuration
- A `/Data/Models/` directory with TorchScript models. See [`Data/Models/README.md`](Data/Models/README.md) for full details.

A complete example of how to run registration with IMPACT is provided in the script [`impact_example.py`](impact_example.py).

## ⚙️ IMPACT Parameter Map Reference

This section describes the configuration parameters required to use the `IMPACT` in Elastix.


### 🔗 Example Minimal Configuration

```txt
(ModelsPath "/Data/Models/TS/M291_1_Layers.pt")
(Dimension "3")
(NumberOfChannels "1")
(PatchSize "5*5*5")
(VoxelSize "1.5*1.5*1.5")
(LayersMask "1")
(SubsetFeatures "32")
(LayersWeight "1")
(Mode "Jacobian")
(GPU 0)
(PCA "0")
(Distance "MSE")
(FeaturesMapUpdateInterval -1)

### 📘 Parameter Descriptions

- **`ModelsPath`**  
  Path to TorchScript models used for feature extraction.  
  Example:  
  ```txt
  (ModelsPath "/Models/model2.pt")
  ```

- **`Dimension`**  
  Dimensionality of the model input images (typically `"2"` or `"3"`).    
  ```txt
  (Dimension "3")
  ```

- **`NumberOfChannels`**  
  Number of channels in the model input image (e.g., `1` for grayscale, `3` for RGB).  
  ```txt
  (NumberOfChannels "1")
  ```

- **`PatchSize`**  
  Local patch size (in voxels) for feature sampling. Format: `X*Y*Z`.   
  ```txt
  (PatchSize "5*5*5")
  ```

- **`VoxelSize`**  
  Physical spacing of the voxels (in mm).  
  Defines the resolution of features extracted from the model.  
  ```txt
  (VoxelSize "1.5*1.5*1.5")
  ```

- **`LayersMask`**  
  Binary string that selects which output layers to use.  
  For example, `"1"` selects the first layer, `"00000001"` selects the last layer.  
  ```txt
  (LayersMask "1")
  ```

- **`SubsetFeatures`**  
  Number of feature channels to randomly sample per voxel.  
  Helps reduce dimensionality and memory usage.  
  ```txt
  (SubsetFeatures "32")
  ```

- **`LayersWeight`**  
  Weight applied to each model or layer in the final loss.  
  Allows tuning the relative importance.  
  ```txt
  (LayersWeight "1")
  ```

- **`Mode`**  
  Execution mode of the metric:  
  - `"Static"` = features are precomputed once  
  - `"Jacobian"` = features are recomputed and gradients are backpropagated  
  ```txt
  (Mode "Jacobian")
  ```

- **`GPU`**  
  Index of the GPU to use (e.g., `0` for the first GPU).  
  Set to `-1` to force CPU mode.  
  ```txt
  (GPU 0)
  ```

- **`PCA`**  
  Number of principal components to retain (for feature compression).  
  Set to `0` to disable PCA.  
  ```txt
  (PCA "0")
  ```

- **`Distance`**  
  Distance function used to compare feature vectors.  
  Supported values: `L1`, `L2`, `Cosine`, `L1Cosine`, `Dice`, `NCC`.  
  ```txt
  (Distance "L2")
  ```

- **`FeaturesMapUpdateInterval`**  
  In `"Static"` mode, controls how often features are recomputed.  
  Set to `-1` to never update (fully fixed features).  
  ```txt
  (FeaturesMapUpdateInterval -1)
  ```

- **`WriteFeatureMaps`**  
  Enables saving the input images and the corresponding output feature maps to disk when using `"Static"` mode.  
  Useful for inspection, debugging, or visualizing which semantic features are extracted at each level.  
  The feature maps are saved in the output directory with the following naming conventions:

  - Input images:  
    ```
    Fixed_<N>_<M>.mha
    Moving_<N>_<M>.mha
    ```
    where `<N>` is the resolution level and `<M>` is the model index.

  - Feature maps:  
    ```
    FeatureMap/Fixed_<N>_<R1>_<R2>_<R3>.mha
    FeatureMap/Moving_<N>_<R1>_<R2>_<R3>.mha
    ```
    where `<R1>, <R2>, <R3>` are the voxel sizes used.

  Example:  
  ```txt
  (WriteFeatureMaps "/Data/Features")
  ```

  Default: `"false"`


### 🔧 Advanced Use: Multi-model and Multi-resolution

IMPACT supports the use of multiple pretrained models in parallel, and allows full control over their configuration at each resolution level.

---

#### 🧠 Multi-model Setup

You can provide multiple TorchScript models using space-separated paths:

```txt
(ModelsPath "/Models/M850_8_Layers.pt /Models/MIND/R1D2.pt")
```

Each model must have corresponding values for all key parameters, such as:

```txt
(Dimension "3 3")
(NumberOfChannels "1 1")
(PatchSize "5*5*5 7*7*7")
(VoxelSize "1.5*1.5*1.5 6*6*6")
(LayersMask "00000001 1")
(SubsetFeatures "64 16")
(LayersWeight "1.0 0.5")
(Distance "Dice L2")
(PCA "0 0")
```

---

#### 🌀 Multi-resolution Configuration

All parameters support Elastix's **multi-resolution** syntax.  
You can define a different value per resolution level (outer quotes per level):

```txt
(VoxelSize "6*6*6" "3*3*3" "1.5*1.5*1.5" "1*1*1")
```

This defines a 4-resolution setup where:
- First model uses decreasing resolution across levels

The same syntax can be used for:
- `VoxelSize`
- `LayersMask`
- `SubsetFeatures`
- `LayersWeight`
- `PCA`
- `Distance`

---

#### 🔀 Fixed and Moving-Specific Models

You can also assign **different models** to the fixed and moving images:

```txt
(FixedModelsPath "/Models/TS/M850_8_Layers.pt")
(MovingModelsPath "/Models/MIND/R1D2.pt")
```

In that case, use the `Fixed*` and `Moving*` versions of all model-related parameters:

```txt
(FixedDimension "3")
(FixedVoxelSize "1.5*1.5*1.5")
(MovingDimension "3")
(MovingVoxelSize "6*6*6")
```

All other parameters (`PatchSize`, `Distance`, `LayersMask`, etc.) should follow the same rule.

---

This flexibility allows you to combine low- and high-level features, handcrafted and learned representations, and modality-specific models — all in the same registration.
