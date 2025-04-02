# 🐳 Docker for Elastix + IMPACT Metric

This Docker environment builds a version of **Elastix** with support for the **IMPACT similarity metric**.

---

## ⚙️ Build the Docker Image

From the root of the repository:

```bash
docker build -t elastix_impact docker/
```

---

## 🚀 Run the Container

To execute a registration:

```bash
docker run --rm --gpus all \
  -v "./Data:/Data" \
  -v "./Out:/Out" \
  elastix_impact
```

---

## 📁 Required Folder Structure

The container expects the following files inside the `Data/` directory:

- `Data/Models/`  
  Must contain the TorchScript-exported models used by the IMPACT metric (e.g., TotalSegmentator, SAM2.1, etc.).  
  👉 See [`Data/Models/README.md`](../Data/Models/README.md) for model download instructions.

- `Data/FixedImage.mha` and `Data/MovingImage.mha`
  The images used for registration.

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

- `Data/ParameterMap.txt`  
  Elastix parameter file with the configuration for Impact.  
  👉 See [`ParameterMaps/README.md`](../ParameterMaps/README.md) for detailed configuration examples.

Optional files (for mask-based registration or evaluation):

- `Data/Fixed_mask.mha` and `Data/Moving_mask.mha` — binary masks to restrict metric computation.
- `Data/Fixed_landmarks.txt` and `Data/Moving_landmarks.txt` — optional landmarks for TRE evaluation.

Output results (transforms, logs) will be written to the mounted `Out/` directory.

---

## 📜 Example

To see a complete example of how to run a registration using this Docker setup, check:

👉 [`docker/run_docker_impact_example.py`](run_docker_impact_example.py)

---

### 💡 Using GPU:

If you have a compatible GPU and the necessary toolkit (e.g., NVIDIA Docker support), it will be available inside the Docker container. You can select which GPU to use by setting the GPU parameter in the ParameterMap file ((GPU 0), (GPU 1), or (GPU -1) to disable GPU usage). This enables the registration process to run on the GPU, significantly speeding up computation.

Without GPU / No Toolkit:
If you don't have a GPU or the required toolkit, simply remove the --gpus all flag from the command. In this case, set (GPU -1) in the Elastix parameter map to disable GPU usage and run the process on the CPU.
