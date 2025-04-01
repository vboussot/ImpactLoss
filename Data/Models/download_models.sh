#!/bin/bash

# ======================================
# IMPACT TorchScript Model Downloader
# ======================================

BASE_URL="https://huggingface.co/VBoussot/impact-torchscript-models/resolve/main"
DEST_DIR="."

declare -A MODELS

# Anatomix
MODELS["Anatomix"]="Anatomix.pt"

# DinoV2
MODELS["DinoV2"]="Small_1_Layers.pt"

# MIND
MODELS["MIND"]="R1D1.pt R1D2.pt R2D1.pt R2D2.pt"

# SAM2.1
MODELS["SAM2.1"]="Tiny_1_Layers.pt Tiny_2_Layers.pt Tiny_3_Layers.pt Tiny_4_Layers.pt Tiny_5_Layers.pt"

# TotalSegmentator (TS)
MODELS["TS"]="$(for m in M258 M291 M292 M293 M294 M295 M297 M298 M730 M731 M732 M733; do
  for l in {1..8}; do
    echo -n "${m}_${l}_Layers.pt "
  done
done)"

# =============================================

# Create folders
echo "Creating destination directories under $DEST_DIR"
for folder in "${!MODELS[@]}"; do
  mkdir -p "$DEST_DIR/$folder"
done

# Download function
download() {
  local folder="$1"
  local file="$2"
  local url="$BASE_URL/$folder/$file"
  local out="$DEST_DIR/$folder/$file"
  echo "→ $folder/$file"
  wget -q --show-progress -O "$out" "$url"
}

# Perform downloads
echo "Downloading models..."
for folder in "${!MODELS[@]}"; do
  for file in ${MODELS[$folder]}; do
    download "$folder" "$file"
  done
done

echo "All models downloaded successfully into $DEST_DIR."