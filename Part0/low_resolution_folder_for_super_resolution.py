import os
import cv2
import numpy as np

def apply_low_resolution(image):
    """
    Applies a series of processes to the input image to create a low-resolution version.
    Returns the resized high-resolution image and the low-resolution image.
    """
    # === Adjust: Resize to match the dataset's target size
    # Step 1: Resize to HR target size
    hr_resized = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)

    # Step 2: Downsample to target LR size using bicubic interpolation
    lr_resized = cv2.resize(hr_resized, (32, 32), interpolation=cv2.INTER_CUBIC)

    return hr_resized, lr_resized

# === Adjust: Folder Paths
base_path = "dataset/gt"  # Path to source high-resolution images
lr_base_path = "dataset/lq"  # Path to save the degraded low-resolution images

# Iterate through train and test folders
for dataset in ["train", "test"]:
    dataset_path = os.path.join(base_path, dataset)
    lr_dataset_path = os.path.join(lr_base_path, dataset)

    # Create LR train/test directory if it doesn't exist
    os.makedirs(lr_dataset_path, exist_ok=True)

    for image_name in os.listdir(dataset_path):
        if image_name.lower().endswith(('.jpeg', '.jpg', '.png')):
            image_path = os.path.join(dataset_path, image_name)
            image = cv2.imread(image_path)

            if image is not None:
                hr_image, degraded_image = apply_low_resolution(image)

                hr_output_image_path = os.path.join(dataset_path, image_name)
                if cv2.imwrite(hr_output_image_path, hr_image):
                    print(f"Saved degraded GT resized image: {hr_output_image_path}")

                lr_output_image_path = os.path.join(lr_dataset_path, image_name)
                
                if cv2.imwrite(lr_output_image_path, degraded_image):
                    print(f"Saved degraded LR image: {lr_output_image_path}")
                else:
                    print(f"Failed to save degraded LR image: {lr_output_image_path}")
            else:
                print(f"Failed to load image at {image_path}")

