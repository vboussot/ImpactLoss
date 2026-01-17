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
MODELS["DinoV2"]="Small.pt"

# MIND
MODELS["MIND"]="R1D1_2D.pt R1D2_2D.pt R2D1_2D.pt R2D2_2D.pt R1D1_3D.pt R1D2_3D.pt R2D1_3D.pt R2D2_3D.pt"

# SAM2.1
MODELS["SAM2.1"]="SAM2.1_Tiny.pt SAM2.1_Small.pt"

# TotalSegmentator (TS)
MODELS["TS"]="M258.pt M291.pt M292.pt M293.pt M294.pt M295.pt M297.pt M298.pt M730.pt M731.pt M732.pt M733.pt M850.pt M851.pt M852.pt M853.pt"

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