import h5py
import numpy as np
import cv2
from sklearn.preprocessing import label_binarize,binarize
import matplotlib.pyplot as plt


# def showImages(images,labels):
#     plt.figure()
#     fig, axes = plt.subplots(nrows=1, ncols=len(images), sharex=True, sharey=True, squeeze=False)
#     for k in range(len(images)):
#         im = images[k]
#         label = labels[k]
#         x = int(k / len(images))
#         y = int(k % len(images))
#
#         axes[x][y].imshow(im, cmap='gray', interpolation='nearest')
#         axes[x][y].set_title(label)
#     plt.show()
def readFromPath(fileName, subSet=False):
    db = h5py.File(fileName, 'r')
    if subSet:
        im_names = list(db['data'].keys())[:200]
    else:
        im_names = list(db['data'].keys())

    return (db, im_names)

import itertools
def readData(db, im_names, labelsMap, imgSize):
    chars = []
    for i in range(len(im_names)):
        # images
        image_path = im_names[i]

        img = db['data'][image_path]
        imgFont = img.attrs['font']
        imgTxt = img.attrs['txt']

        charBB = img.attrs['charBB']
        wordBB = img.attrs['wordBB']

        org_img = np.array(db['data'][image_path])

        offset = 0
        iterIndex = 0
        for img_txt_ind in range(len(imgTxt)):
            leftIndex = offset
            rightIndex = offset + len(imgTxt[img_txt_ind]) - 1

            print("IMG index is " + str(i) + " from " + str(len(im_names)))
            wordTxt = imgTxt[iterIndex]
            wordChars = createWordChars(image_path, org_img, wordTxt, leftIndex, rightIndex, charBB, imgFont, imgSize,
                              labelsMap)
            chars = chars + wordChars
            iterIndex = iterIndex + 1
            offset = offset + len(imgTxt[img_txt_ind])

    return chars


def createWordChars(image_path, org_img, wordTxt, leftIndex, rightIndex, charBB, imgFont, imgSize, labelsMap):

    index = 0
    chars = []
    for b_inx in range(leftIndex, rightIndex + 1):
        labels = []
        bb = charBB[:, :, b_inx]
        sec = np.float32([bb.T[0], bb.T[1], bb.T[3], bb.T[2]])
        target = np.float32([[0, 0], [imgSize, 0], [0, imgSize], [imgSize, imgSize]])
        M = cv2.getPerspectiveTransform(sec, target)
        dst = cv2.warpPerspective(org_img, M, (imgSize, imgSize))

        grayImgChar = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

        labelTxt = imgFont[b_inx].decode('UTF-8')
        labels.append(labelsMap[labelTxt])
        y_bin = label_binarize(labels, classes=[i for i in range(0, 3)])

        charTxt = wordTxt.decode('UTF-8')[index]
        print("img is " + str(image_path) + ", word is " + str(wordTxt.decode("utf-8")), "char is " + charTxt)
        char = (image_path, wordTxt.decode("utf-8"), grayImgChar, y_bin, charTxt, labelTxt)
        chars.append(char)
        index = index +1
    return chars
