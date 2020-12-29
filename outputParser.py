import h5py
import numpy as np
import cv2
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

import csv


def write(fineName, data):
    with open(fineName, mode='w') as validation_file:
        fieldnames = ['', 'image', 'char', 'Skylark', 'Sweet Puppy', 'Ubuntu Mono']  # utf8 ?
        validation_writer = csv.DictWriter(validation_file, fieldnames=fieldnames)
        validation_writer.writeheader()
        for index in range(len(data)):
            classifier = data[index][2][0]
            print("classifier " + str(classifier))
            validation_writer.writerow(
                {'': index, 'image': data[index][0], 'char': data[index][1], 'Skylark': classifier[0],
                 'Sweet Puppy': classifier[1], 'Ubuntu Mono': classifier[2]})


# data = [("ant+hill_10.jpg_0", "word1", "pic", [[0, 0, 1]], "X"),
#         ("ant+2222222.jpg_0", "word2", "pic2", [[0, 1, 0]], "Y")]
#
# write("validation.csv", data)
