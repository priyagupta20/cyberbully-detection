from numpy import expand_dims
from keras.models import load_model
import glob
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix
# load the model
model = load_model('model.h5')
print("loaded model from disk")
X_data = []
Y_data = []
files = glob.glob("nude_test/*.jpg")
for myFile in files:
    image = Image.open(myFile)
    img_resized = image.resize((28,28)).convert("L")
    arr = np.array(img_resized)
    X_data.append (arr)
    Y_data.append(0)
files = glob.glob("violence_test/*.jpg")
for myFile in files:
    image = Image.open(myFile)
    img_resized = image.resize((28,28)).convert("L")
    arr = np.array(img_resized)
    X_data.append (arr)
    Y_data.append(1)
X_data = np.array(X_data)    
Y_data = np.array(Y_data)
X = expand_dims(X_data, axis=-1)
X = X.astype('float32')
X = (X - 127.5) / 127.5
y_pred = model.predict(X,verbose=1)
y_pred = y_pred>0.5
Y_data = pd.get_dummies(Y_data)

cm = multilabel_confusion_matrix(Y_data, y_pred)
print(cm)
TP = 0
TN = 0
FN = 0
FP = 0
for i in range(0,2):
    TP += cm[i][0][0]
    TN += cm[i][1][1]
    FP += cm[i][1][0]
    FN += cm[i][0][1]
cm = np.array([[TP, FP],[FN, TN]])
row_labels = ['Actual P', 'Actual N'] 
print()
print("-----------------CONFUSION MATRIX----------------------")
print()
print ("          Predicted P   Predcited N")
for row_label, row in zip(row_labels, cm):
    print ('%s [%s]' % (row_label, ' '.join('%9s' % i for i in row)))
print()
print("--------------------------------------------------------")
print()    
precision = TP/(TP+FP) 
recall = TP/(TP+FN)
f1 = (2*precision*recall)/(precision+recall)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1)
