import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import pandas as pd
import glob
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

#Neural networks defining
def FCNN(unit):
    FCNN = tf.keras.models.Sequential()
    FCNN.add(tf.keras.layers.Dense(units = unit, activation='relu'))
    FCNN.add(tf.keras.layers.Dense(units = unit, activation='relu'))
    FCNN.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))
    FCNN.compile(optimizer='adam', loss='CategoricalCrossentropy', metrics=['accuracy'])
    return FCNN

def CNN(unit,unit2):
    CNN = tf.keras.models.Sequential()
    CNN.add = tf.keras.layers.InputLayer(input_shape=(32, 32, 3))
    CNN.add(tf.keras.layers.Conv1D(unit, kernel_size= 3 ,activation='relu'))
    CNN.add(tf.keras.layers.Conv1D(unit2, kernel_size= 3, activation='relu'))
    CNN.add(tf.keras.layers.Flatten())
    CNN.add(tf.keras.layers.Dense(1, activation='softmax'))
    CNN.compile(optimizer='adam', loss='CategoricalCrossentropy', metrics=['accuracy'])
    return CNN
#Create lists to store features and labels
data = []
labels = []
new_size = (32,32)

#Loops Through every image in given dictionary
for root, subFolders, files in os.walk("Training/"):
    for file in files:
        file_name = root + "/" + file 
        image = cv.imread(file_name)#Reads image
        
        resize_image = cv.resize(image, new_size)
        #Enhance - Runs functions
        data.append(resize_image)

        # Extract label from filename
        label = file_name.split("/")[-2] 
        labels.append(label)

X = np.array(data)
y = np.array(labels)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, y_train.shape)

X_train = np.array(X_train) / 255
X_test = np.array(X_test) / 255

X_train.reshape(-1, 32, 32, 1)
y_train = np.array(y_train)

X_test.reshape(-1, 32, 32, 1)
y_test = np.array(y_test)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

"""
model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(32,32,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()

model.compile(optimizer = 'adam' , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

history = model.fit(X_train,y_train, epochs = 500 , validation_data = (X_test, y_test))
history.evaluate(X_test, y_test)

"""
#-------------------------------------Hyper Parameter tuning----------------------------------------------------------
param_SVC= {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['linear','rbf']}
param_KNN = {'n_neighbors':[1,2,3,4], 'leaf_size':[100, 200, 300, 400]}
param_FCNN = {'epochs':[9, 12, 15], 'unit':[32, 64]}
FCNN_Model = KerasClassifier(FCNN)
param_CNN = {'epochs':[9, 12, 15], 'unit':[32, 64], 'unit2':[32, 64]}
CNN_Model = KerasClassifier(CNN)


TUNE_SVC = GridSearchCV(estimator=SVC(), param_grid=param_SVC, cv=10, scoring='accuracy')
TUNE_SVC.fit(X_train,y_train)
print("HyperParameter results")
print(TUNE_SVC.cv_results_)
print("")
print("Best HyperParameters:")
print(TUNE_SVC.best_estimator_)

print("Summary of results")
df = pd.DataFrame({'parammeters': TUNE_SVC.cv_results_["params"], 'Mean_Accuracy': TUNE_SVC.cv_results_["mean_test_score"]})
print(df)

"""
TUNE_KNN = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_KNN, cv=10, scoring='accuracy')
TUNE_KNN.fit(X_train,y_train)
print("HyperParameter results")
print(TUNE_KNN.cv_results_)
print("")
print("Best HyperParameters:")
print(TUNE_KNN.best_estimator_)


print("Summary of results")
df = pd.DataFrame({'parammeters': TUNE_KNN.cv_results_["params"], 'Mean_Accuracy': TUNE_KNN.cv_results_["mean_test_score"]})
print(df)


TUNE_FCNN = GridSearchCV(estimator=FCNN_Model, param_grid=param_FCNN, cv=3, scoring='accuracy')
gs = TUNE_FCNN.fit(X_train,y_train)
print("HyperParameter results")
print(gs.cv_results_)
print("")
print("Best HyperParameters:")
t = gs.best_estimator_
print(t)


print("Summary of results")
df = pd.DataFrame({'parammeters': gs.cv_results_["params"], 'Mean_Accuracy': gs.cv_results_["mean_test_score"]})
print(df)


TUNE_CNN = GridSearchCV(estimator=CNN_Model, param_grid=param_CNN, cv=3, scoring='accuracy')
gs = TUNE_CNN.fit(X_train,y_train)
print("HyperParameter results")
print(gs.cv_results_)
print("")
print("Best HyperParameters:")
t = gs.best_estimator_
print(t)


print("Summary of results")
df = pd.DataFrame({'parammeters': gs.cv_results_["params"], 'Mean_Accuracy': gs.cv_results_["mean_test_score"]})
print(df)

#---------------------------HyperParameter End------------------------------
#---------------------------Models Creation------------------------------

clf = SVC()
clf.fit(X_train,y_train)
print(clf.score(X_train,y_train))

m = KerasClassifier(FCNN)
m.fit(X_train, y_train, epochs = 3)
results = m.evaluate(X_test, y_test, batch_size=128)
print(results)

pred = clf.predict(X_test)

matrix = confusion_matrix(y_test, pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix = matrix)

print("Pred = ", accuracy_score(y_test, pred))
cm_display.plot()
plt.show()
"""
