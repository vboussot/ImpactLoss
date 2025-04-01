#!/bin/bash
set -e

CMD="/lib/elastix-install/bin/elastix"

CMD+=" -f /Data/Fixed_image.mha -m /Data/Moving_image.mha -p /Data/ParameterMap.txt -out /Out/ -threads 12"

if [ -f "/Data/Fixed_mask.mha" ]; then
    CMD+=" -fMask /Data/Fixed_mask.mha"
fi

if [ -f "/Data/Moving_mask.mha" ]; then
    CMD+=" -fMask /Data/Moving_mask.mha"
fi

if [ -f "/Data/Fixed_landmarks.txt" ] && [ -f "/Data/Moving_landmarks.txt" ]; then
    CMD+=" -fp /Data/Fixed_landmarks.txt -mp /Data/Moving_landmarks.txt"
fi

$CMD