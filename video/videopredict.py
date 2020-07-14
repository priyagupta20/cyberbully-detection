import glob
import os
import subprocess
from keras.models import load_model
import pandas as pd
from tqdm import tqdm
import numpy as np
from keras.preprocessing import image   # for preprocessing the images
import shutil
class FFMPEGFrames:
    def __init__(self, output):
        self.output = output

    def extract_frames(self, input, fps):

        if not os.path.exists(self.output):
            os.makedirs(self.output)

        query = "ffmpeg -i " + input + " -vf fps=" + str(fps) + " " + self.output+ "/output%0d.png"
        response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
        
        
vid = input("Enter the name of video:")
fps = 1

f = FFMPEGFrames("predict/")
f.extract_frames(vid, fps)

X_data = []
files = glob.glob("predict/*.png")
for myFile in files:
    X_data.append (myFile)
    

predict_image = []
for i in tqdm(range(len(X_data))):
    img = image.load_img(X_data[i], target_size=(28,28,3))
    img = image.img_to_array(img)
    # normalizing the pixel value
    img = img/255
    predict_image.append(img)
    
# converting the list to numpy array
X = np.array(predict_image)
    
model=load_model("classifier_3.h5")
y = model.predict(X)

v=0
nv=0
for i in range(0,len(y)):
    nv+= y[i][0]
    v+=y[i][1]

nv/=len(y)
v/=len(y)

print("Violence:",v)
print("NonViolence:",nv)

shutil.rmtree("predict")