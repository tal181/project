from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
import numpy as np
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D , UpSampling2D ,Conv2DTranspose

def createModel(img_size):
    input_shape = (img_size, img_size, 1)

    model = Sequential()
    model.add(Conv2D(img_size, (3, 3), input_shape=input_shape))  # First convolution Layer
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(img_size, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Second Convolution Layer

    model.add(Conv2D(img_size *2, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # Flatten the layers
    model.add(Dense(img_size *8))  # FC layer with 256 neurons
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(3))  # As classes are 3
    model.add(Activation('softmax'))

    return model

    # Cu Layers

def createModel2(img_size):
    input_shape = (img_size, img_size, 1)

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(16, 16), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(4, 4), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(
        Conv2DTranspose(128, (24, 24), strides=(2, 2), activation='relu', padding='same', kernel_initializer='uniform'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(
        Conv2DTranspose(64, (12, 12), strides=(2, 2), activation='relu', padding='same', kernel_initializer='uniform'))
    model.add(UpSampling2D(size=(2, 2)))

    # Cs Layers
    model.add(Conv2D(256, kernel_size=(12, 12), activation='relu'))

    model.add(Conv2D(256, kernel_size=(12, 12), activation='relu'))

    model.add(Conv2D(256, kernel_size=(10, 10), activation='relu'))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(4096, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(2383, activation='relu'))

    model.add(Dense(3, activation='softmax'))

    return model