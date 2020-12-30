import matplotlib.pyplot as plt
from inputParser import readData, readFromPath
from outputParser import write
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import label_binarize
from dataGen import generate
from itertools import chain, islice
from sklearn import preprocessing
from model import createModel

# parms
fileName = "font_recognition_train_set/SynthText.h5"
percent = 0.8
img_size = 32
epochs = 50
batch_size = 32
filter_size = 64
subSet = True
labelsMap = {"Skylark": 0, "Sweet Puppy": 1, "Ubuntu Mono": 2}

# ############# read data
(db, im_names) = readFromPath(fileName, subSet)
data = readData(db, im_names, labelsMap, img_size)

gray_data = generate(data, labelsMap)

X, y = np.asarray(list(chain.from_iterable(islice(item, 2, 3) for item in
                                           gray_data))), \
       np.asarray(list(chain.from_iterable(islice(item, 1, 2) for item in
                                           gray_data)))

a = np.array(X).reshape(len(X), img_size, img_size, 1)
X_train, X_test, y_train, y_test = train_test_split(a, y,
                                                    test_size=0.2,
                                                    random_state=42)

lb = preprocessing.LabelBinarizer()
y_train, y_test = lb.fit_transform(y_train), lb.fit_transform(y_test)


#########################################################################
model = createModel(img_size)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# train data
history = model.fit(X_train,
                    y_train,
                    verbose=1,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test, y_test)
                    )


test_results = []
for index in range(len(X_test)):
    temp = np.array(X_test[index][2]).reshape(1, 32, 32, 1)
    predict = model.predict(temp)
    class_predict = model.predict_classes(temp)

    bin_class = label_binarize(class_predict, classes=[i for i in range(0, 3)])

    plt.imshow(np.array(X_test[index][2]))
    print("img is " + str(X_test[index][0]))
    print("class_predict is " + str(bin_class)) # this is the class
    print("real is " + str(y_test[index]))
    print("predict is " + str(predict))
    print("charTxt is " + str(X_test[index][4]))

    print("--------------------------------")
    test_results.append((X_test[index][0],X_test[index][4],bin_class))
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

