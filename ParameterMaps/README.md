## 🔧 IMPACT Configuration

This section describes how to configure the IMPACT metric in Elastix.

Several example parameter maps are provided in the [`ParameterMaps/`](../ParameterMaps) directory to help you get started.

---

### 🔗 Minimal Example

```txt
(ImpactModelsPath "/Data/Models/TS/M291_1_Layers.pt")
(ImpactDimension "3")
(ImpactNumberOfChannels "1")
(ImpactPatchSize "5*5*5")
(ImpactVoxelSize "1.5*1.5*1.5")
(ImpactLayersMask "1")
(ImpactSubsetFeatures "32")
(ImpactLayersWeight "1")
(ImpactMode "Jacobian")
(ImpactGPU 0)
(ImpactPCA "0")
(ImpactDistance "L2")
(ImpactFeaturesMapUpdateInterval -1)
(ImpactWriteFeatureMaps "false")
```

---

### 📘 Parameter Descriptions

- **`ImpactModelsPath`**  
  Path to TorchScript model used for feature extraction.

- **`ImpactDimension`**  
  Dimensionality of the input expected by the TorchScript model.
  Typically:
  - `2` 2D models (e.g., SAM2.1, DINOv2) 
  - `3` 3D models (e.g., TotalSegmentator, Anatomix, MIND).


- **`ImpactNumberOfChannels`**  
   Number of channels expected by the TorchScript model.
   Typically:
    - `1` for grayscale medical images (e.g., **TotalSegmentator**, **Anatomix**, **MIND**)
    - `3` for RGB-based models (e.g., **SAM2.1**, **DINOv2**)

- **`ImpactPatchSize`**  
  Size of the local patch (in voxels) used for feature extraction. 
  This value controls the spatial context seen by the feature extractor around each sampled point.
  
  Format:
  - `X*Y*Z` for 3D models  
  - `X*Y` for 2D models

  The **minimum valid patch size** corresponds to the **field of view (FOV)** of the selected model and layer.  
  This is the smallest spatial context required for the model to produce valid features. 
  ➤ The FOV values for each model/layer are documented in the model repository:
    🔗 [https://huggingface.co/VBoussot/impact-torchscript-models](https://huggingface.co/VBoussot/impact-torchscript-models)

    ⚠️ **Warning:** Setting a `PatchSize` larger than the model’s field of view won’t cause an error,  
    but it will slow down computation without improving registration quality.  
    Use the smallest valid size for best performance.

- **`ImpactVoxelSize`**  
  Physical spacing of the voxels (in millimeters).  
  This defines the **resolution of the input image at which features are extracted** by the model.

  Format:
  - `X*Y*Z` for 3D models  
  - `X*Y` for 2D models

  ⚠️ **Warning:** Do **not rely on Elastix’s default image pyramid resampling**,  
  as it may silently change the resolution of your images at each level,  
  leading to inconsistencies with the `ImpactVoxelSize` explicitly defined in your configuration.

  You should instead **explicitly define the `ImpactVoxelSize`** and **disable automatic resampling** in the image pyramid.

  #### 🔧 Recommended configuration to disable resampling:
  ```txt
  (FixedImagePyramid "FixedGenericImagePyramid")
  (MovingImagePyramid "MovingGenericImagePyramid")
  (FixedImagePyramidRescaleSchedule 1 1 1 1 1 1)     # one `1` per (resolution level × dimension)
  (MovingImagePyramidRescaleSchedule 1 1 1 1 1 1)
  ```

  ✅ This ensures that the `ImpactVoxelSize` you provide is preserved exactly across all pyramid levels,  
  so features are extracted at the **correct physical scale**.

- **`ImpactLayersMask`**  
  Binary string used to select which output layers of the model to include in the similarity computation.

  Each character corresponds to one layer:
  - `1` = use this layer
  - `0` = ignore this layer

  The number of bits must exactly match the number of output layers available in the model.

  For example:
  - `"00000001"` selects only the last layer in an 8-layer model
  - `"11100000"` selects the first three layers
  - `"1"` selects the only layer (for single-layer models like MIND)

- **`ImpactMode`**  
  Defines how features are computed during registration:

  - `"Static"`: features are computed **once per image and per resolution level** (pure inference).  
    The image is divided into overlapping patches defined by `PatchSize`, and features are extracted per patch.  
    To apply the model to the **entire image without patching**, use:
    ```txt
    (PatchSize "0*0*0")
    ```
    This avoids edge artifacts, especially with fully convolutional networks.

    ✅ Fast and memory-efficient  
    ❌ Not differentiable, no gradient propagation through the feature extractor  
    ⚠️ **Important limitations with downsampling models:**  

      This mode is not recommended for models that significantly reduce spatial resolution  
      (e.g., transformers like SAM, DINOv2, or deep CNNs with large strides), because:

      - The output features **lose local alignment** with input voxels,  
        making similarity computations spatially inconsistent.

      - Features are **frozen and no longer respond** to geometric transformations applied during registration,  
        breaking the natural **spatial sensitivity** of deep features.

      As a result, comparing static feature maps can lead to **inaccurate similarity estimation**  
      and **suboptimal convergence** during optimization.<br><br>

  - `"Jacobian"`: features are computed from randomly extracted patches at each iteration.
    Gradients are backpropagated through the TorchScript model.

    ✅ Precise, fully differentiable  
    ❌ Slower

- **`ImpactSubsetFeatures`**  
  Number of feature channels randomly selected per features voxel at each iteration.
  Reduces both computational cost and memory usage during registration.


- **`ImpactLayersWeight`**  
  Relative importance of each layer in the final similarity score.  


- **`ImpactGPU`**  
  GPU device index. Use `-1` to force CPU execution.  


- **`ImpactPCA`**  
  Number of principal components to retain for dimensionality reduction. Set to `0` to disable.  


- **`ImpactDistance`**  
  Distance metric used to compare feature vectors. Supported values:  
  `L1`, `L2`, `Cosine`, `L1Cosine`, `Dice`, `NCC`, `DotProduct`.  


- **`ImpactFeaturesMapUpdateInterval`**  
  In **`Static`** mode, this parameter controls how often the feature maps are recomputed during optimization.  
  - Set to `-1` to compute features **once per resolution level** (recommended for efficiency).  
  - Set to a positive integer (e.g., `10`) to recompute every _N_ iterations, which can improve alignment if deformation changes rapidly.

- **`ImpactWriteFeatureMaps`**  
  Enables saving the input images and extracted feature maps to disk (only in `"Static"` mode).  
  Useful for inspection, debugging, or visualizing the semantic features used during registration.

  - If set to `"false"` → writing is disabled  
  - If set to a path (e.g., `"/Data/Features"`) → feature maps will be saved to that directory
  
  The output includes:
  
    - Input images per resolution and model:
      ```txt
      Fixed_<res>_<X>_<Y>_<Z>.mha
      Moving_<res>_<X>_<Y>_<Z>.mha
      ```

    - Extracted feature maps:
      ```txt
      Fixed_<res>_<model>.mha
      Moving_<res>_<model>.mha
      ```

  where `<res>` is the resolution level, and `<X>`, `<Y>`, `<Z>` are the voxel sizes used for patch extraction.
---

### 🔧 Advanced Use: Multi-model and Multi-resolution

IMPACT supports parallel use of multiple models and per-resolution customization.

#### 🧠 Multi-model Setup

Use space-separated lists for multiple models:

```txt
(ImpactModelsPath "/Models/M850_8_Layers.pt /Models/MIND/R1D2.pt")
(ImpactDimension "3 3")
(ImpactNumberOfChannels "1 1")
(ImpactPatchSize "5*5*5 7*7*7")
(ImpactVoxelSize "1.5*1.5*1.5 6*6*6")
(ImpactLayersMask "00000001 1")
(ImpactSubsetFeatures "64 16")
(ImpactLayersWeight "1.0 0.5")
(ImpactDistance "Dice L2")
(ImpactPCA "0 0")
```

 ⚠️ **Note:** This syntax is not supported by parameters:

 - `ImpactMode`  
 - `ImpactGPU`  
 - `ImpactFeaturesMapUpdateInterval`  
 - `ImpactWriteFeatureMaps`

---

#### 🌀 Multi-resolution Configuration

You can define different settings per resolution level (outer quotes = one level):

```txt
(ImpactVoxelSize "6*6*6" "3*3*3" "1.5*1.5*1.5" "1*1*1")
```

This syntax is supported by all parameters.

⚠️ Note: If a parameter is not explicitly specified for a given resolution level, the value from the first level is automatically reused for the remaining levels.

This allows you to simplify the configuration when the same value applies across multiple resolutions.

---

#### 🔀 Fixed and Moving-Specific Models

You can assign **different models** to the fixed and moving images using:

```txt
(FixedModelsPath "/Models/TS/M850_8_Layers.pt")
(MovingModelsPath "/Models/MIND/R1D2.pt")
```

In this case, all model-specific parameters must be defined **independently** for each image using the `ImpactFixed*` and `ImpactMoving*` prefixes:

- `ImpactFixedModelsPath`  
- `ImpactFixedDimension`  
- `ImpactFixedNumberOfChannels`  
- `ImpactFixedPatchSize`  
- `ImpactFixedVoxelSize`  
- `ImpactFixedLayersMask`

This enables **asymmetric model configurations**, where each image can use a model adapted to its modality or anatomical content. 

---

This flexible structure allows combining high-level and low-level features, handcrafted and deep features, and modality-specific models, all within a single registration pipeline.
It also serves as a modular platform for research and experimentation, enabling systematic comparisons across models, layers, patch sizes, and similarity strategies.
