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

image = Image.open('sample4.jpg')
# convert image to numpy array
img_resized = image.resize((28,28)).convert("L")
arr = np.array(img_resized)
data = np.array([arr])

data = expand_dims(data,axis=-1)
data = data.astype('float32')
data = (data - 127.5) / 127.5

print("prediction=",model.predict(data))