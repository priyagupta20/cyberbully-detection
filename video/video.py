import pandas as pd
from keras.preprocessing import image   
import numpy as np  
import os
import subprocess
from sklearn.model_selection import train_test_split
import glob
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, LSTM, Conv2D, MaxPooling2D, TimeDistributed
import matplotlib.pyplot as plt

class FFMPEGFrames:
    def __init__(self, output):
        self.output = output

    def extract_frames(self, input, fps, n):

        if not os.path.exists(self.output):
            os.makedirs(self.output)

        query = "ffmpeg -i " + input + " -vf fps=" + str(fps) + " " + self.output+ "/" + str(n) + ".png"
        response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()


frames_not_converted = False
if(frames_not_converted):
    fps=1
    f_train_v = FFMPEGFrames("data/trainimages/violence/")
    f_train_nv = FFMPEGFrames("data/trainimages/nonviolence/")
    f_test_v = FFMPEGFrames("data/testimages/violence/")
    f_test_nv = FFMPEGFrames("data/testimages/nonviolence/")
    
    files = glob.glob("data/train/Violence/*.avi")
    i=0
    for f in files:
        f_train_v.extract_frames(f,fps,i)
        i+=1
    
    files = glob.glob("data/train/NonViolence/*.avi")
    i=0
    for f in files:
        f_train_nv.extract_frames(f,fps,i)
        i+=1


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
    # loading the image and keeping the target size as (28,28,3)
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
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, 
                                                    stratify = y)
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(X_train.shape)
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print(type(self.y_val), self.y_val)
            print("--------")
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
            
RocAuc = RocAucEvaluation(validation_data=(X_test, y_test), interval=1)

filters=10
filtersize=(5,5)
model = Sequential() 
model.add(Conv2D(32, (2, 2), input_shape=(28,28,3))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Conv2D(32, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Conv2D(64, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(60, return_sequences=True))
model.add(Flatten()) 
model.add(Dense(64)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(2)) 
model.add(Activation('sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test),callbacks=[RocAuc], 
          batch_size=50)
model.summary()
model.save('classifier_3.h5')