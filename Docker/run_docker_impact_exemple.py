import SimpleITK as sitk
import numpy as np
import os
import time

def standardize_MRI(image: sitk.Image) -> sitk.Image:    
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

def standardize_ImageNet(image: sitk.Image) -> sitk.Image:
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

    # Define absolute paths for the data and output directories
    data_path = os.path.abspath("../Data/")  # Directory for input data
    out_path = os.path.abspath("../Out/")    # Directory for output results

    # Load the fixed and moving images from specified paths
    fixed_image = sitk.ReadImage("...")  # Replace with the path to the fixed image
    # fixed_mask = sitk.ReadImage("...")  # Uncomment if you have a fixed mask
    # fixed_landmarks = writeLandmarks(np.zeros((100, 3)), "{}Fixed_landmarks.mha".format(data_path))  # Uncomment if you have fixed landmarks

    moving_image = sitk.ReadImage("...")  # Replace with the path to the moving image
    # moving_mask = sitk.ReadImage("...")  # Uncomment if you have a moving mask
    # moving_landmarks = writeLandmarks(np.zeros((100, 3)), "{}Moving_landmarks.mha".format(data_path))  # Uncomment if you have moving landmarks

    # Normalize or standardize the images based on the model type and write them to the data directory
    # For SAM2.1, use clip_and_standardize_ImageNet, 
    # For TS/M730, TS/M731, TS/M732, TS/M733, use clip_and_standardize_MRI, 
    # For other models, use clip_and_standardize_CT
    sitk.WriteImage(standardize_ImageNet(fixed_image), "{}/Fixed_image.mha".format(data_path))  # Standardize fixed image
    sitk.WriteImage(standardize_ImageNet(moving_image), "{}/Moving_image.mha".format(data_path))  # Standardize moving image

    # Copy the example parameter map configuration to the data directory
    os.system("cp ../ParameterMaps/ParameterMap_TS_2_Layers_Jacobian.txt {}/ParameterMap.txt".format(data_path))

    start_time = time.time()

    # Run the Docker container with GPU support, mounting the necessary directories for input data and output
    os.system("docker run --rm --gpus all -v \"{}:/Data\" -v \"{}:/Out\" elastix_impact".format(data_path, out_path))

    stop_time = time.time()
    print("Time : {}".format(stop_time - start_time))  # Output the total time taken for the Docker run

    # Copy the resulting transformation file from the output directory to the current directory
    os.system("cp {}/TransformParameters.0.txt ./TransformParameters.0.txt".format(out_path))
    if os.path.exists("{}/TransformParameters.0.itk.txt".format(out_path)):
        os.system("cp {}/TransformParameters.0.itk.txt ./Transform.itk.txt".format(out_path))