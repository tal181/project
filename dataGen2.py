import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
import collections

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest')


def data_generator(data, img, times):
    arr = img_to_array(img[2])  # shape (32, 32, 1)
    arr = arr.reshape((1,) + arr.shape)  # shape (1, 32, 32, 1)
    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    # prefix = im.split('_', 1)[0]
    for batch in datagen.flow(arr, batch_size=64):
        # save_to_dir = 'font_recognition_train_set/extra',
        # save_prefix = prefix,
        # save_format = 'jpg'
        data.append((img[0], img[1], np.squeeze(batch, axis=0), img[3]))
        i += 1
        if i > times:
            break  # otherwise the generator would loop indefinitely
    return data


def get_map_of_duplication(list_per_font):
    return collections.Counter([d for (a, b, c, d) in list_per_font])


def count_for_data_generator(data, labelsMap):
    # split data per font
    list_per_font = collections.defaultdict(list)
    list_of_duplication_per_font = []
    for sub in data:
        list_per_font[sub[1]].append(sub)

    # count each char for each for
    for font in list(labelsMap.keys()):
        list_of_duplication_per_font.append(get_map_of_duplication(
            list_per_font[font]))

    return list_per_font, list_of_duplication_per_font


def generate(data, labelsMap, gen_factor):
    #list_per_font, list_of_duplication_per_font = count_for_data_generator(data, labelsMap)

    images = []
    for i in range(len(data)) :
        img = data[i]
        data_generator(images, img, gen_factor)

    data = data + images
    index = 0
    gray_data = list(tuple())
    for tuple_info in data:
        img_gray = cv2.cvtColor(tuple_info[2], cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(img_gray, kernel, iterations=1)
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        # applying the sharpening kernel to the input image & displaying it.
        sharpened = cv2.filter2D(erosion, -1, kernel)
        gray_data.append((tuple_info[0], tuple_info[1], sharpened, tuple_info[3]))
        print("index is " + str(index) + " from " + str(len(data)))
        index = index + 1

    return gray_data
