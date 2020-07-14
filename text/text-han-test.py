from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras import backend as K
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, roc_curve, roc_auc_score
import pandas as pd
def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        if mask is not None:
            a *= K.cast(mask, K.floatx())
            
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
    

json_file = open('han_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json,custom_objects={'AttentionWithContext':AttentionWithContext})
# load weights into new model
loaded_model.load_weights("han_model.h5")
print("Loaded model from disk");

y_test = pd.read_csv("ytest.csv")
y_pred = pd.read_csv("ypred.csv")
y_test = y_test.to_numpy()
y_pred = y_pred.to_numpy()
print(type(y_pred[:,0]))
print(y_test)
print(y_pred)

fpr = [0,0,0,0,0,0]
tpr = [0,0,0,0,0,0]
for i in range(6):
    fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_pred[ :,i])
y_pred = y_pred>0.5

cm = multilabel_confusion_matrix(y_test, y_pred)
TP = 0
TN = 0
FN = 0
FP = 0
for i in range(0,6):
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
print()
  

# Plot of a ROC curve for a specific class0
for i in range(6):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve')
    plt.plot([0, 1], [0, 1])
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic ')
    plt.legend(loc="lower right")
    plt.xticks([])
    plt.yticks([])
    plt.show()
