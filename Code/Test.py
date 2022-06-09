#%%
import cv2
from cv2 import INTER_AREA
import torch
import matplotlib.pyplot as pl
import os
import numpy
from PIL import Image
import h5py
from skimage.feature import hog
# assign directory


param = h5py.File('../Images/DatasetAlcohol.h5', 'r')
X = param['X_TrainSet'][:]

fd, hog_image = hog(X[18], orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, channel_axis=-1)
pl.imshow(hog_image)