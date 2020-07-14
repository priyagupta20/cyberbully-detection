from keras.models import load_model
import glob
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np  
import os
import subprocess
from tqdm import tqdm
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score, roc_curve
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
    f_test_v = FFMPEGFrames("data/testimages/violence/")
    f_test_nv = FFMPEGFrames("data/testimages/nonviolence/")
    
    
    files = glob.glob("data/test/Violence/*.avi")
    i=0
    for f in files:
        f_test_v.extract_frames(f,fps,i)
        i+=1
        
    files = glob.glob("data/test/NonViolence/*.avi")
    i=0
    for f in files:
        f_test_nv.extract_frames(f,fps,i)
        i+=1
        
model=load_model("classifier_3.h5")
X_data = []
Y_data = []
files = glob.glob("data/testimages/violence/*.png")
for myFile in files:
    X_data.append (myFile)
    Y_data.append("Violence")
files = glob.glob("data/testimages/nonviolence/*.png")
for myFile in files:
    X_data.append (myFile)
    Y_data.append("NonViolence")
    
test_data = pd.DataFrame()
test_data['image'] = X_data
test_data['class'] = Y_data

# converting the dataframe into csv file 
test_data.to_csv('test.csv',header=True, index=False)

test = pd.read_csv('test.csv')
test_image = []

# for loop to read and store frames
for i in tqdm(range(test.shape[0])):
    # loading the image and keeping the target size as (224,224,3)
    img = image.load_img(test['image'][i], target_size=(28,28,3))
    # converting it to array
    img = image.img_to_array(img)
    # normalizing the pixel value
    img = img/255
    # appending the image to the train_image list
    test_image.append(img)
    
# converting the list to numpy array
x = np.array(test_image)
y = test['class']


y = pd.get_dummies(y)


test_loss, test_acc = model.evaluate(x,y, verbose=1)
print(test_acc)

y_pred = model.predict(x,verbose=1)
print(roc_auc_score(y['NonViolence'].to_list(), y_pred[:,0]))
fpr, tpr, _= roc_curve(y['NonViolence'].to_list(), y_pred[:,0])
plt.figure()
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1])
plt.xlim([0, 1.0])
plt.ylim([0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for class Non-Violence')
plt.legend(loc="lower right")
plt.show()

fpr, tpr, _= roc_curve(y['Violence'].to_list(), y_pred[:,1])
plt.figure()
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1])
plt.xlim([0, 1.0])
plt.ylim([0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for class Violence')
plt.legend(loc="lower right")
plt.show()

y_pred = y_pred>0.5
cm = multilabel_confusion_matrix(y, y_pred)
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