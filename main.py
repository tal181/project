import matplotlib.pyplot as plt
from model import createModel, createModel2
from inputParser import readData, readFromPath
from outputParser import write
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import label_binarize
fileName = "font_recognition_train_set/SynthText.h5"

# parms
percent = 0.8
img_size = 32
epochs = 1
batch_size = 32
filter_size = 64
subSet = True

# labels
labelsMap = {"Skylark": 0, "Sweet Puppy": 1, "Ubuntu Mono": 2}
# ############# read data

(db, im_names) = readFromPath(fileName, subSet)
ans = readData(db, im_names, labelsMap, img_size)

Y_data = []
for item in ans:
    Y_data.append(item[3])

# X_data = np.array(X_data).reshape(len(X_data), img_size, img_size, 1)
# split data
train_files, test_files, train_cat, test_cat = train_test_split(np.array(ans), np.array(Y_data),
                                                                test_size=(1 - percent),
                                                                random_state=42)

# ############# create model
model = createModel(img_size, filter_size)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# train data
X_data = []
for item in train_files:
    X_data.append(item[2])
X_data = np.array(X_data).reshape(len(X_data), img_size, img_size, 1)

X_data_test = []
for item in test_files:
    X_data_test.append(item[2])
X_data_test = np.array(X_data_test).reshape(len(X_data_test), img_size, img_size, 1)

history = model.fit(X_data,
          train_cat,
          verbose=1,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(X_data_test, test_cat)
          )


test_results = []
for index in range(len(test_files)):
    temp = np.array(test_files[index][2]).reshape(1, 32, 32, 1)
    predict = model.predict(temp)
    class_predict = model.predict_classes(temp)

    bin_class = label_binarize(class_predict, classes=[i for i in range(0, 3)])

    plt.imshow(np.array(test_files[index][2]))
    print("img is " + str(test_files[index][0]))
    print("class_predict is " + str(bin_class)) # this is the class
    print("real is " + str(test_cat[index]))
    print("predict is " + str(predict))
    print("charTxt is " + str(test_files[index][4]))

    print("--------------------------------")
    test_results.append((test_files[index][0],test_files[index][4],bin_class))
# write output
write("validation.csv",test_results)

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

