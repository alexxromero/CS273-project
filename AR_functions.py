from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow
print("Num GPUs Available: ", len(tensorflow.config.experimental.list_physical_devices('GPU')))

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D
from tensorflow.keras.optimizers import SGD

tensorflow.set_random_seed(123)
np.random.seed(123)

def load_dataset():
    num_classes=10

    # training set
    cifar10_dir = "./cifar-10-batches-py/"
    x_train = np.zeros((5, 10000, 32, 32, 3))
    y_train = np.zeros((5, 10000, 10))
    for i in range(1, 6):
        batch_file = cifar10_dir+"data_batch_"+str(i)
        with open(batch_file, 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
        x, y = batch[b'data'], batch[b'labels']
        x = x.reshape((np.shape(x)[0], 3, 32, 32)).transpose(0, 2, 3, 1).astype('float32')
        y = to_categorical(y, num_classes)
        x_train[i-1] = x
        y_train[i-1] = y
    x_train = np.vstack((x_train))
    y_train = np.vstack((y_train))

    # testing set
    test_file = "./cifar-10-batches-py/test_batch"
    x_test = np.zeros((10000, 32, 32, 3))
    y_test = np.zeros((10000, 10))
    with open(test_file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    x, y = batch[b'data'], batch[b'labels']
    x = x.reshape((np.shape(x)[0], 3, 32, 32)).transpose(0, 2, 3, 1).astype('float32')
    y = to_categorical(y, num_classes)
    x_test = x
    y_test = y

    return (x_train, y_train), (x_test, y_test)


def preprocessImages(x_train, x_test):
    norm = np.max(x_train) # 225
    x_train = x_train/norm
    x_test = x_test/norm

    return x_train, x_test


def loss_acc_plots(history):
    fig = plt.figure(figsize=(6, 8))
    # plot loss
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='green', label='validation')
    plt.legend()
    # plot accuracy
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['acc'], color='blue', label='train')
    plt.plot(history.history['val_acc'], color='green', label='validation')
    plt.legend()
    plt.show()


def evaluate_model(model, x_test, y_test):

    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print('> %.3f' % (acc * 100.0))
