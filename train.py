# -*- coding: utf-8 -*-
import os

# Get the absolute and directory path of the current file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

# Import necessary modules and functions
import model_resunet
import numpy as np
import tensorflow as tf
from math import floor
from tqdm import tqdm
from skimage.io import imread
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam

import random
import scipy.misc
from PIL import Image
import shutil
from utils import imgstitch, DatasetLoad, get_image_data_from_X_test

# Define a custom F1Score metric class
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall = tf.keras.metrics.Recall(thresholds=threshold)
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.bool)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * (p * r) / (p + r + tf.keras.backend.epsilon())

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

# Function for learning rate decay. The learning rate will reduce by a factor of 0.1 every 10 epochs.
def schedlr(epoch, lr):
    new_lr = 0.001 * (0.1)**(floor(epoch/10))
    return new_lr

# Define hyperparameters
IMG_SIZE = 224
BATCH = 8
EPOCHS = 20

# Define dataset paths
train_dataset = r'dataset/samples_train'
test_dataset = r'dataset/testing'
val_dataset = r'dataset/validation'

# Make a list of the test folders to be used when predicting the model
test_fol = os.listdir(test_dataset)

# Load the datasets
#X_train, Y_train, X_test, Y_test, X_val, Y_val = DatasetLoad(train_dataset, test_dataset, val_dataset)

print("Contents of image_files before loading datasets:", os.listdir(os.path.join(test_dataset, 'image')))
image_files = os.listdir(os.path.join(test_dataset, 'image'))
print("Contents of image_files after loading datasets:", image_files)
# Define optimizer
sgd_optimizer = Adam()

# Define metrics for evaluation
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
f1 = F1Score()

# Instantiate the model
model = model_resunet.ResUNet((IMG_SIZE, IMG_SIZE, 3))
model.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy', precision, recall, f1])
model.summary()

# Callback to save model checkpoints
checkpoint_path = os.path.join(dname, 'models', 'resunet.{epoch:02d}.hdf5')
checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False)
# Define the parent folder containing the testing data
testing_folder = 'dataset/testing'

# If previous results exist, delete them so the results won't be mixed up
results_dir = os.path.join('results')
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)

# Make a new results directory
os.makedirs(results_dir, exist_ok=True)

# Get the list of image files in the "images" folder
image_files = os.listdir(os.path.join(testing_folder, 'image'))

# Loop through the image files
for image_file in image_files:
    # Create a subdirectory within results for each image
    image_name = os.path.splitext(image_file)[0]
    sub_dir = os.path.join(results_dir, image_name)
    os.makedirs(sub_dir, exist_ok=True)

    # Generate the image index from the image name
    image_index = int(image_name) - 1  # Since the image names start from 1

    # Get image data for prediction from X_test
    X_test, _, _, _, _, _ = DatasetLoad(train_dataset, test_dataset, val_dataset)
    image_data = get_image_data_from_X_test(X_test, image_index)

   # Add batch dimension to image data
    image_data_batch = np.expand_dims(image_data, axis=0)

    # Make predictions on the image data
    pred_test = model.predict(image_data, verbose=1)
    

    # Assuming the model predicts class probabilities, threshold the predictions to create binary masks
    pred_test_mask = (pred_test > 0.4).astype(np.uint8)

    # Save predicted masks
    for mask_index, mask in enumerate(pred_test_mask):
        output_mask = np.squeeze(mask * 255)
        save_img = Image.fromarray(output_mask, 'L')
        save_path = os.path.join(sub_dir, f"{mask_index}.png")
        save_img.save(save_path.replace('\\', '/'), 'PNG')

    # Call imgstitch function to stitch masks
    imgstitch(sub_dir)