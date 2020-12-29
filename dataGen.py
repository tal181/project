import h5py
import matplotlib.pyplot as plt
import cv2
import numpy as np
from itertools import chain, islice
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Reshape, Flatten, LSTM, Dense, \
    TimeDistributed, InputLayer, Conv2D, MaxPool2D
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import backend
from sklearn import preprocessing
import keras
import collections
from sklearn.model_selection import KFold


def get_points(pts, index):
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[:, :, index].T[0]
    rect[1] = pts[:, :, index].T[1]
    rect[3] = pts[:, :, index].T[2]
    rect[2] = pts[:, :, index].T[3]
    return rect


def split(word):
    return [char for char in word]


def data_generator(data, img, times):
    arr = img_to_array(img[1])  # shape (28, 28, 1)
    arr = arr.reshape((1,) + arr.shape)  # shape (1, 28, 28, 1)
    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    # prefix = im.split('_', 1)[0]
    for batch in datagen.flow(arr, batch_size=64):
        # save_to_dir = 'font_recognition_train_set/extra',
        # save_prefix = prefix,
        # save_format = 'jpg'
        data.append((img[0].decode('UTF-8'), np.squeeze(batch, axis=0), img[2]))
        i += 1
        if i > times:
            break  # otherwise the generator would loop indefinitely
    return data


def get_map_of_duplication(list_per_font):
    return collections.Counter([z for (x, y, z) in list_per_font])


def count_for_data_generator(data):
    # split data per font
    list_per_font = collections.defaultdict(list)
    list_of_duplication_per_font = []
    for sub in data:
        list_per_font[sub[0].decode('UTF-8')].append(sub)

    # count each char for each for
    for font in FONTS:
        list_of_duplication_per_font.append(get_map_of_duplication(
            list_per_font[font]))

    return list_per_font, list_of_duplication_per_font


SIZE, ZERO, DATA_STR = 28, 0, 'data'
FONTS = ['Skylark', 'Sweet Puppy', 'Ubuntu Mono']
file_name = 'font_recognition_train_set/SynthText.h5'
db = h5py.File(file_name, 'r')
im_names = list(db[DATA_STR].keys())
data = list(tuple())

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest')

for im in im_names:
    img = db[DATA_STR][im][:]
    font = db[DATA_STR][im].attrs['font']
    txt = db[DATA_STR][im].attrs['txt']
    charBB = db[DATA_STR][im].attrs['charBB']
    wordBB = db[DATA_STR][im].attrs['wordBB']

    txt_split = []
    for word in txt:
        txt_split.append(split(word.decode('UTF-8')))
    txt_split = [val for sublist in txt_split for val in sublist]

    for i in range(ZERO, charBB.shape[2]):
        rect = get_points(charBB, i)
        dst = np.array([[ZERO, ZERO], [SIZE, ZERO], [ZERO, SIZE], [SIZE, SIZE]],
                       dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (SIZE, SIZE))

        data.append((font[i], warped, txt_split[i]))  # sharpened

list_per_font, list_of_duplication_per_font = count_for_data_generator(data)

for line, font in zip(list_of_duplication_per_font, FONTS):
    max_value = max(line.values())
    for img, pair in zip(list_per_font[font], zip(line.keys(),
                                                  line.values())):
        if max_value > pair[1]:
            data_generator(data, img, max_value - pair[1])  # max_value - pair[1]
            line[pair[0]] += (max_value - pair[1])  # max_value - pair[1]

gray_data = list(tuple())
for tuple_info in data:
    img_gray = cv2.cvtColor(tuple_info[1], cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img_gray, kernel, iterations=1)
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(erosion, -1, kernel)
    gray_data.append((tuple_info[0], sharpened, tuple_info[2]))

X, y = np.asarray(list(chain.from_iterable(islice(item, 1, 2) for item in
                                           gray_data))), \
       np.asarray(list(chain.from_iterable(islice(item, 0, 1) for item in
                                           gray_data)))
a = np.array(X).reshape(len(X), 28, 28, 1)
X_train, X_test, y_train, y_test = train_test_split(a, y,
                                                    test_size=0.2,
                                                    random_state=42)

lb = preprocessing.LabelBinarizer()
y_train, y_test = lb.fit_transform(y_train), lb.fit_transform(y_test)
#########################################################################
from model import createModel,createModel2

img_size = 28
model = createModel(img_size, 0)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# split data
train_files, test_files, train_cat, test_cat = train_test_split(X_train, y_train,
                                                                test_size=(0.8),
                                                                random_state=42)
# train data
history = model.fit(train_files,
                    train_cat,
                    verbose=1,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test, y_test)
                    )

# Loss Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)

# Accuracy Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)
a
# fc_layer_size, num_folds, fold_no = 256, 2, 1
# # Define per-fold score containers
# acc_per_fold, loss_per_fold = [], []
#
# # Define the K-fold Cross Validator
# kfold = KFold(n_splits=num_folds, shuffle=True)
#
# for train, test in kfold.split(X_train, y_train):
#     conv_inputs = keras.Input(shape=(SIZE, SIZE, 1), name='ani_image')
#     conv_layer = Conv2D(fc_layer_size, kernel_size=3,
#                         activation='relu')(conv_inputs)
#     conv_layer = MaxPool2D(pool_size=(2,2))(conv_layer)
#
#     conv_layer = Conv2D(fc_layer_size / 2, kernel_size=3,
#                         activation='relu')(conv_layer)
#     conv_layer = MaxPool2D(pool_size=(2,2))(conv_layer)
#     # turn image to vector.
#     conv_x = Flatten(name = 'flattened_features')(conv_layer)
#
#     conv_x = Dense(fc_layer_size, activation='relu', name='first_layer')(conv_x)
#     conv_x = Dense(fc_layer_size / 2, activation='relu',
#                    name='second_layer')(conv_x)
#     conv_outputs = Dense(3, activation='softmax', name='class')(conv_x)
#     conv_model = keras.Model(inputs=conv_inputs, outputs=conv_outputs)
#
#
#     conv_model.compile(optimizer=keras.optimizers.Adam(lr=1e-6),  # Optimizer
#                   # Loss function to minimize
#                   loss='categorical_crossentropy',
#                   # List of metrics to monitor
#                   metrics=['accuracy'])
#
#     print(f'Training for fold {fold_no} ...')
#
#     # conv_model.summary()
#     history = conv_model.fit(X_train[train], y_train[train], epochs=64,
#                              batch_size=64,
#               validation_data=(X_train[test], y_train[test]))
#
#     # Generate generalization metrics
#     scores = conv_model.evaluate(X_test, y_test, verbose=0)
#     acc_per_fold.append(scores[1] * 100)
#     loss_per_fold.append(scores[0])
#
#     # Increase fold number
#     fold_no += 1
#
#
# # == Provide average scores ==
# print('-----------------------------------------------------------------------')
# print('Score per fold')
# for i in range(0, len(acc_per_fold)):
#   print('---------------------------------------------------------------------')
#   print(f'> Fold:{i+1} - Loss:{loss_per_fold[i]} - Accuracy:{acc_per_fold[i]}%')
# print('-----------------------------------------------------------------------')
# print('Average scores for all folds:')
# print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
# print(f'> Loss: {np.mean(loss_per_fold)}')
# print('-----------------------------------------------------------------------')
