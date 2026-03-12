import numpy as np
from scipy.spatial import distance as dist
import os
import cv2
import imutils
import shutil
from tqdm import tqdm 

def findEdges(image):
    # find edges in image
    gray = cv2.GaussianBlur(image, (1, 1), 0)
    edged = cv2.Canny(gray, 100, 400)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    return edged

def getImgContours(edged):
    # find contours in the edge map
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    return contours

def getBoxes(contours, orig):
    # get the boxes
    boxes = []
    centers = []
    for contour in contours:
        box = cv2.minAreaRect(contour)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        (tl, tr, br, bl) = box
        if (dist.euclidean(tl, bl)) > 0 and (dist.euclidean(tl, tr)) > 0:
            boxes.append(box)
    return boxes

def detect_cells(src_path, image_size=(128, 128)):

    # get pathname of each image
    img_path = os.path.join(src_path)

    # Open 
    original = cv2.imread(img_path)
    image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # add padding to the image to better detect cell at the edge
    image = cv2.copyMakeBorder(image,10,10,10,10,cv2.BORDER_CONSTANT,value=[198, 203, 208])
    
    #thresholding the image to get the target cell
    image1 = cv2.inRange(image,(80, 80, 180),(180, 170, 245))
    
    # openning errosion then dilation
    kernel = np.ones((3, 3), np.uint8)
    kernel1 = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(image1, kernel, iterations=2)
    image1 = cv2.dilate(img_erosion, kernel1, iterations=5)
    
    #detecting the blood cell
    edgedImage = findEdges(image1)
    edgedContours = getImgContours(edgedImage)
    edgedBoxes =  getBoxes(edgedContours, image.copy())
    if len(edgedBoxes)==0:
        return None
    # get the large box and get its cordinate
    last = edgedBoxes[-1]
    max_x = int(max(last[:,0]))
    min_x = int( min(last[:,0]))
    max_y = int(max(last[:,1]))
    min_y = int(min(last[:,1]))
    
    # draw the contour and fill it 
    mask = np.zeros_like(image)
    cv2.drawContours(mask, edgedContours, len(edgedContours)-1, (255,255,255), -1) 
    
    # any pixel but the pixels inside the contour is zero
    image[mask==0] = 0
    
    # extract th blood cell
    image = image[min_y:max_y, min_x:max_x]

    if (np.size(image)==0):
        return None
    # resize th image
    image = cv2.resize(image, image_size)

    return image 


def create_dataset_from_txt(txt_train, txt_test, original_dir, train_dataset_dir, test_dataset_dir):
    original_train_dir = os.path.join(original_dir, 'train')
    original_test_dir = os.path.join(original_dir, 'test')

    # Read train and test image paths from text files
    with open(txt_train, 'r') as f:
        train_images = [line.strip() for line in f.readlines()]
    with open(txt_test, 'r') as f:
        test_images = [line.strip() for line in f.readlines()]

    # Create train dataset
    for img_path in train_images:
        img_basename = os.path.basename(img_path)
        src = os.path.join(original_train_dir, img_basename)
        image = detect_cells(src)
        dst = os.path.join(train_dataset_dir, img_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if image is not None:
            cv2.imwrite(dst, image)

    # Create test dataset
    for img_path in test_images:
        img_basename = os.path.basename(img_path)
        src = os.path.join(original_test_dir, img_basename)
        image = detect_cells(src)
        dst = os.path.join(test_dataset_dir, img_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if image is not None:
            cv2.imwrite(dst, image)
            

def create_test_dataset_from_txt(txt_test, original_dir, test_dataset_dir):
    # Read train and test image paths from text files
    with open(txt_test, 'r') as f:
        test_images = [line.strip() for line in f.readlines()]

    # Create test dataset
    for img_path in test_images:
        img_basename = os.path.basename(img_path)
        src = os.path.join(original_dir, img_basename)
        dst = os.path.join(test_dataset_dir, img_path)

        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

# === Adjust: Folder Paths
# Paths for the dataset and output directories
original_dir = 'dataset/SR_20k/output'  # Path to original images
train_dataset_dir = 'dataset/CLS_20k/train'  # Path to save train dataset
test_dataset_dir = 'dataset/CLS_20k/test'  # Path to save test dataset

# Generate train and test datasets
# create_dataset_from_txt('../CS_dataset/CS_classification_dataset/CS_classification_data_info/train_images.txt',
#                                '../CS_dataset/CS_classification_dataset/CS_classification_data_info/test_images.txt',
#                                original_dir, train_dataset_dir, test_dataset_dir)
create_dataset_from_txt('dataset/meta_info/gt_train_images.txt', 'dataset/meta_info/gt_test_images.txt',
                               original_dir, train_dataset_dir, test_dataset_dir)