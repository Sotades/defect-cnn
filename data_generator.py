import numpy as np
import keras
import cv2
import os.path as osp
import os
import random
import math

class DataGenerator(keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32, 32, 32), n_channels=1,
                 n_classes=10, shuffle=True):
        # Initialization
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Generates data containing batch_size samples.
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Read file
            img = np.array(cv2.imread(input_path + '/' + ID + 'JPG'))
            img.flatten()


            # Store sample
            X[i, ] = img

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    @staticmethod
    def build_label_list(ok_directory, nok_directory):
        # Return labels
        # Generate list of files to process
        labels = {}
        files = [(ok_directory + f) for f in os.listdir(ok_directory) if osp.isfile(osp.join(ok_directory, f))]
        for filename in files:
            id = filename[:-4]
            labels[id] = "0"

        files = [(nok_directory + f) for f in os.listdir(nok_directory) if osp.isfile(osp.join(nok_directory, f))]
        for filename in files:
            id = filename[:-4]
            labels[id] = "1"

        return labels

    @staticmethod
    def build_partition(validation_amount, labels):
        partition = {}
        id = []
        for label in labels.items():
            id.append(label)

            random.shuffle(id)

        total_ids_num = len(id)

        train_ids_num = math.floor((1 - validation_amount) * total_ids_num)

        validate_ids_num = total_ids_num - train_ids_num

        train_ids = id[0:train_ids_num]
        validate_ids = id[(train_ids_num + 1):]

        partition['train'] = train_ids
        partition['validation'] = validate_ids

        return partition
