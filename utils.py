# -*- coding: utf-8 -*-
from PIL import Image
import os
import numpy as np
from skimage.io import imread
from tqdm import tqdm
from skimage.transform import resize

# Original image size and desired image size for processing
OGIMG_SIZE = 1500
IMG_SIZE = 224
# Overlap between adjacent image patches
OVERLAP = 14

def imgstitch(img_path, overlap=100):
    """
    Efficiently stitches predicted image patches into the final output image, considering overlap regions.

    Parameters:
        img_path (str): Path to the directory containing test image patches.
        overlap (int, optional): The amount of overlap between adjacent patches. Defaults to 100.

    Returns:
        None. The final stitched image is saved as "output.png" within the same directory.
    """
    # Get a list of image files sorted by their numeric index
    _, _, img_files = next(os.walk(img_path))
    img_files = sorted(img_files, key=lambda x: int(os.path.splitext(x)[0]))

    # Get the width and height of a single image patch
    patch_width, patch_height = Image.open(os.path.join(img_path, img_files[0])).size

    # Calculate the final stitched image dimensions considering overlap
    num_patches = len(img_files)
    rows = (num_patches // (patch_width // overlap) + 1)  # Number of rows to accommodate all patches
    stitched_width = num_patches * (patch_width - overlap) + patch_width
    stitched_height = rows * (patch_height - overlap) + patch_height

    # Create a new empty image for the final stitched result
    full_img = Image.new('RGB', (stitched_width, stitched_height))

    # Loop through each image file and stitch them together
    x, y = 0, 0  # Current position for pasting in the full image
    for n, id_ in enumerate(img_files):
        patch = Image.open(os.path.join(img_path, id_))

        # Check if we need to move to a new row in the stitched image
        if x + patch_width > stitched_width:
            x = 0
            y += patch_height - overlap

        # Paste the current patch onto the full image, considering overlap
        full_img.paste(patch, (x, y))

        # Update position for the next patch
        x += patch_width - overlap

    # Save the final stitched image
    full_img.save(os.path.join(img_path, 'output') + '.png', 'PNG')


def get_image_data_from_X_test(X_test, image_names):
    """
    Extracts the image data from the X_test array for a list of image names.

    Args:
        X_test (numpy.ndarray): Array containing image data for testing.
        image_names (list): List of image names or identifiers.

    Returns:
        numpy.ndarray: The batch of extracted image data with shape (None, height, width, channels).
    """
    image_batch = []
    for image_name in image_names:
        if image_name in X_test:
            image_data = X_test[image_name]
            image_batch.append(image_data)
        else:
            raise ValueError(f"Image '{image_name}' not found in X_test")

    # Convert the list of image data to a numpy array and stack them along the batch axis
    return np.stack(image_batch, axis=0)


    
def DatasetLoad(train_dataset, test_dataset, val_dataset):
    """
    Loads training, testing, and validation datasets.

    Args:
        train_dataset (str): Sampled training images directory.
        test_dataset (str): Sampled test images directory.
        val_dataset (str): Sampled validation images directory.

    Returns:
        Tuple: Tuple containing training, testing, and validation datasets.
            X_train (numpy.ndarray): Training dataset for features.
            Y_train (numpy.ndarray): Training dataset for labels.
            X_test (numpy.ndarray): Test dataset for feature predictions.
            Y_test (numpy.ndarray): Test dataset for label predictions.
            X_val (numpy.ndarray): Validation dataset for feature validation.
            Y_val (numpy.ndarray): Validation dataset for label validation.
    """
    # TRAINING DATASET
    train_files = os.listdir(os.path.join(train_dataset, 'image'))
    training_imgs = len(train_files)
    train_ids = list(range(1, training_imgs + 1))
    
    X_train = np.zeros((len(train_ids), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_SIZE, IMG_SIZE, 1), dtype=bool)
    
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        img = imread(train_dataset + '/image/' + str(id_) + '.png')
        img_resized = resize(img, (IMG_SIZE, IMG_SIZE, 3), mode='constant', preserve_range=True)
        X_train[n] = img_resized
        mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=bool)
        for mask_file in next(os.walk(train_dataset  + '/mask/')):
            mask_ = imread(train_dataset + '/mask/' + str(id_) + '.png')
            mask_ = np.expand_dims(mask_, axis=-1)
            mask_resized = resize(mask_, (IMG_SIZE, IMG_SIZE, 1), mode='constant', preserve_range=True)
            mask = np.maximum(mask, mask_resized)
        
        Y_train[n] = mask
    
    # VALIDATION DATASET
    val_files = os.listdir(os.path.join(val_dataset, 'image'))
    val_imgs = len(val_files)
    val_ids = list(range(1, val_imgs + 1))
    
    X_val = np.zeros((len(val_ids), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    Y_val = np.zeros((len(val_ids), IMG_SIZE, IMG_SIZE, 1), dtype=bool)
    
    for n, id_ in tqdm(enumerate(val_ids), total=len(val_ids)):
        img = imread(val_dataset + '/image/' + str(id_) + '.png')
        img_resized = resize(img, (IMG_SIZE, IMG_SIZE, 3), mode='constant', preserve_range=True)
        X_val[n] = img_resized
        mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=bool)
        for mask_file in next(os.walk(val_dataset  + '/mask/')):
            mask_ = imread(val_dataset + '/mask/' + str(id_) + '.png')
            mask_ = np.expand_dims(mask_, axis=-1)
            mask_resized = resize(mask_, (IMG_SIZE, IMG_SIZE, 1), mode='constant', preserve_range=True)
            mask = np.maximum(mask, mask_resized)
        
        Y_val[n] = mask
        
    # TESTING DATASET
    test_files = os.listdir(test_dataset)
    test_imgs = len(test_files)
    test_ids = list(range(1, test_imgs + 1))

    X_test = np.zeros((len(test_ids), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    Y_test = np.zeros((len(test_ids), IMG_SIZE, IMG_SIZE, 1), dtype=np.bool_)

    mask_files = os.listdir(os.path.join(test_dataset, 'mask'))
    for mask_file in mask_files:
        mask_ = imread(os.path.join(test_dataset, 'mask', mask_file))
        mask_ = np.expand_dims(mask_, axis=-1)
        mask_resized = resize(mask_, (IMG_SIZE, IMG_SIZE, 1), mode='constant', preserve_range=True)
        mask = np.maximum(mask, mask_resized)

    return X_train, Y_train, X_test, Y_test, X_val, Y_val
