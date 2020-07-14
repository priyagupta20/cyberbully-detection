# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:36:20 2019

@author: Priya_Gupta
"""

import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np  
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D

X_data = []
Y_data = []
files = glob.glob("data/trainimages/violence/*.png")
for myFile in files:
    X_data.append (myFile)
    Y_data.append("Violence")
files = glob.glob("data/trainimages/nonviolence/*.png")
for myFile in files:
    X_data.append (myFile)
    Y_data.append("NonViolence")
    
train_data = pd.DataFrame()
train_data['image'] = X_data
train_data['class'] = Y_data

# converting the dataframe into csv file 
train_data.to_csv('train.csv',header=True, index=False)

train = pd.read_csv('train.csv')
train_image = []

# for loop to read and store frames
for i in tqdm(range(train.shape[0])):
    # loading the image and keeping the target size as (224,224,3)
    img = image.load_img(train['image'][i], target_size=(28,28,3))
    # converting it to array
    img = image.img_to_array(img)
    # normalizing the pixel value
    img = img/255
    # appending the image to the train_image list
    train_image.append(img)
    
# converting the list to numpy array
X = np.array(train_image)

# shape of the array
print(X.shape)

# separating the target
y = train['class']

# creating the training and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify = y)
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


model = Sequential()
model.add(Dense(1024, activation='relu',input_shape=(28,28,3)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=50)
model.save('classifier_2.h5')