# First model, a simple stack of 3 convolution layers with a ReLU activation and followed by max-pooling layers

#KERAS
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sklearn

import data_generator
from keras.models import Sequential

from PIL import Image
from numpy import *

# SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Parameters
params = {'dim': (1363200, 1),
          'batch_size': 64,
          'n_classes': 6,
          'n_channels': 1,
          'shuffle': True}

# Directories
ok_directory = 'C:/Users/Tony/Downloads/Dataset2/OK/'
nok_directory = 'C:/Users/Tony/Downloads/Dataset2/NOK/'

labels = data_generator.DataGenerator.build_label_list(ok_directory=ok_directory, nok_directory=nok_directory)

partition = data_generator.DataGenerator.build_partition(validation_amount=0.3, labels=labels)

# Generators
training_generator = data_generator.DataGenerator(partition['train'], labels, **params)
validation_generator = data_generator.DataGenerator(partition['validation'], labels, **params)