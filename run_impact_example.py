import SimpleITK as sitk
import numpy as np
import os
import time

def clip_and_standardize_MRI(image: sitk.Image) -> sitk.Image:    
    data = sitk.GetArrayFromImage(image)
    data = (data-data.mean())/data.std()
    result = sitk.GetImageFromArray(data)
    result.CopyInformation(image)
    return result

def clip_and_standardize_CT(image: sitk.Image) -> sitk.Image:    
    data = sitk.GetArrayFromImage(image)
    data[data < -1024] = -1024
    data[data > 276.0] = 276
    data = (data-(-370.00039267657144))/436.5998675471528
    result = sitk.GetImageFromArray(data)
    result.CopyInformation(image)
    return result

def clip_and_standardize_ImageNet(image: sitk.Image) -> sitk.Image:
    data = sitk.GetArrayFromImage(image)
    data = (data-np.min(data))/(np.max(data)-np.min(data))
    data = (data-0.485)/0.229
    result = sitk.GetImageFromArray(data)
    result.CopyInformation(image)
    return result

def writeLandmarks(landmarks: np.ndarray, filename: str):
    with open(filename, "w") as f:
        f.write("point\n") 
        f.write(f"{len(landmarks)}\n")
        for point in landmarks:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

if __name__ == "__main__":
    data_path = "."
    out_path = "."
    fixed_image = sitk.ReadImage("...")
    moving_image = sitk.ReadImage("...")

    sitk.WriteImage(clip_and_standardize_ImageNet(fixed_image), "{}Fixed_image.mha".format(data_path))
    sitk.WriteImage(clip_and_standardize_ImageNet(moving_image), "{}Moving_image.mha".format(data_path))

    os.system("cp ./ParameterMaps/ParameterMap_SAM_2_Layers_Jacobian.txt ParameterMap.txt".format(data_path))
    start_time = time.time()
    os.system("elastix-install/bin/elastix -f {}/Fixed_image.mha -m {}/Moving_image.mha -p {}/ParameterMap.txt -out {} -threads 12".format(data_path,data_path,data_path,out_path))
    stop_time = time.time()
    print("Time : {}".format(stop_time-start_time))

    os.system("cp {}Transform.itk.txt ./Transform.itk.txt".format(out_path))
    