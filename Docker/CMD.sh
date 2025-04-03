#!/bin/bash
set -e

CMD="/lib/elastix-install/bin/elastix"

if [ -f "/Data/Fixed_image.mha" ]; then
    FIXED="/Data/Fixed_image.mha"
elif [ -f "/Data/Fixed_image.nii.gz" ]; then
    FIXED="/Data/Fixed_image.nii.gz"
else
    echo "Fixed image not found (.mha or .nii.gz)"
    exit 1
fi

if [ -f "/Data/Moving_image.mha" ]; then
    MOVING="/Data/Moving_image.mha"
elif [ -f "/Data/Moving_image.nii.gz" ]; then
    MOVING="/Data/Moving_image.nii.gz"
else
    echo "Moving image not found (.mha or .nii.gz)"
    exit 1
fi

CMD+=" -f $FIXED -m $MOVING -p /Data/ParameterMap.txt -out /Out/ -threads 12"

if [ -f "/Data/Fixed_mask.mha" ]; then
    CMD+=" -fMask /Data/Fixed_mask.mha"
elif [ -f "/Data/Fixed_mask.nii.gz" ]; then
    CMD+=" -fMask /Data/Fixed_mask.nii.gz"
fi

if [ -f "/Data/Moving_mask.mha" ]; then
    CMD+=" -mMask /Data/Moving_mask.mha"
elif [ -f "/Data/Moving_mask.nii.gz" ]; then
    CMD+=" -mMask /Data/Moving_mask.nii.gz"
fi

if [ -f "/Data/Fixed_landmarks.txt" ] && [ -f "/Data/Moving_landmarks.txt" ]; then
    CMD+=" -fp /Data/Fixed_landmarks.txt -mp /Data/Moving_landmarks.txt"
fi

$CMD