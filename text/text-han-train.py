import numpy as np 
import pandas as pd 
np.random.seed(42)
from keras import backend as K
from keras.layers import Dense,Input, LSTM, Bidirectional, Embedding, TimeDistributed, SpatialDropout1D
from keras.preprocessing import text, sequence
from keras import initializers, regularizers, constraints, optimizers
from keras.callbacks import Callback, EarlyStopping
from keras.models import Model, model_from_json
from keras.optimizers import Optimizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, multilabel_confusion_matrix
import nltk
import re
import pickle
import csv
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from keras.engine.topology import Layer
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

print('imports done')

train=pd.read_csv("train.csv")
print(train.shape)
train["comment_text"].fillna("fillna")
X = train["comment_text"].str.lower()
y = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X = list(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print('split done')

def remove_noise(input_text):
    text = re.sub('\(talk\)(.*)\(utc\)','',input_text)
    text = text.split()
    text = [re.sub('[\d]+','',x) for x in text]
    return ' '.join(text)

for i in range(len(X_train)):
    X_train[i] = remove_noise(X_train[i])
for i in range(len(X_test)):
    X_test[i] = remove_noise(X_test[i])
      
print('noise removed')   
    
def replace_word(X):
    repl = {
         ":d": " good ",":dd": " good ",":p": " good ","yay!": " good ","yay": " good ",
         "yaay": " good ","yaaay": " good ","yaaaay": " good ", "yaaaaay": " good ",":/": " bad ",
         ":')": " sad ","&lt;3": " heart ","8)": " smile ", ":-)": " smile ", ":)": " smile ", ";)": " smile ","(-:": " smile ",
        "(:": " smile ",":&gt;": " angry ", ":')": " sad ",":-(": " sad ",":(": " sad ",":s": " sad ", ":-s": " sad ",r"\br\b": "are",r"\bu\b": "you",r"\bhaha\b": "ha",r"\bhahaha\b": "ha",
        r"\bdon't\b": "do not", r"\bdoesn't\b": "does not",r"\bdidn't\b": "did not",r"\bhasn't\b": "has not",
        r"\bhaven't\b": "have not",r"\bhadn't\b": "had not", r"\bwon't\b": "will not",
        r"\bwouldn't\b": "would not",r"\bcan't\b": "can not",r"\bcannot\b": "can not",r"\bi'm\b": "i am",
        "m": "am","r": "are","u": "you","haha": "ha","hahaha": "ha","don't": "do not","doesn't": "does not",
        "didn't": "did not", "hasn't": "has not","haven't": "have not","hadn't": "had not","won't": "will not",
        "wouldn't": "would not","can't": "can not", "cannot": "can not","i'm": "i am","m": "am",
        "i'll" : "i will","its" : "it is","it's" : "it is","'s" : " is","that's" : "that is", "weren't" : "were not"
    }
    keys = repl.keys()
    new_X = []
    for i in X:
        arr = str(i).split()
        xx = ""
        for j in arr:
            j = str(j).lower()
            if j[:4] == 'http' or j[:3] == 'www':
                continue
            if j in keys:
                j = repl[j]
            xx += j + " "
        new_X.append(xx)
    return new_X

X_train = replace_word(X_train)
X_test = replace_word(X_test)

print('words replaced');

max_features=190609
max_senten_len=30
max_senten_num=10
embed_size=300

def filt_sent(X,max_senten_num):
    X_sent = []
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for paragraph in X:
        raw = sent_tokenizer.tokenize(paragraph)
        filt = []
        min_sent_len = 5 if len(raw) <= 10 else 10
        for sentence in raw:
            if len(sentence.split()) >= min_sent_len and len(filt) < max_senten_num:
                filt.append(sentence)
        while len(filt) < max_senten_num:
            filt.append('nosentence')
        X_sent.append(filt)
    return X_sent

X_train_sent = filt_sent(X_train ,max_senten_num)
X_test_sent = filt_sent(X_test, max_senten_num)
print('filtering done')
print("after filt:", len(X_train_sent))

tok=text.Tokenizer(num_words=max_features,lower=True)
tok.fit_on_texts(list(X_train)+list(X_test))
for i in range(len(X_train_sent)):
        X_train_sent[i] = tok.texts_to_sequences(X_train_sent[i])
        X_train_sent[i] = sequence.pad_sequences(X_train_sent[i],maxlen=max_senten_len)
for i in range(len(X_test_sent)):
        X_test_sent[i] = tok.texts_to_sequences(X_test_sent[i])
        X_test_sent[i] = sequence.pad_sequences(X_test_sent[i],maxlen=max_senten_len)
print('tokenize done')

EMBEDDING_FILE = 'glove840b300dtxt/glove.840B.300d.txt'
embeddings_index = {}
with open(EMBEDDING_FILE,encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('embed done')
        
word_index = tok.word_index
#prepare embedding matrix
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

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

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
            

embedding_layer = Embedding(max_features,
                            embed_size,
                            input_length=max_senten_len,
                            weights=[embedding_matrix])

class COCOB(Optimizer):
    """Coin Betting Optimizer from the paper:
        https://arxiv.org/pdf/1705.07795.pdf
    """
    def __init__(self, alpha=100, **kwargs):
            super(COCOB, self).__init__(**kwargs)
            self._alpha = alpha
            with K.name_scope(self.__class__.__name__):
                self.iterations = K.variable(0, dtype='int64', name='iterations')
    def get_updates(self, params, loss, contraints=None):
            self.updates = [K.update_add(self.iterations, 1)]
            grads = self.get_gradients(loss, params)
            shapes = [K.int_shape(p) for p in params]
            L = [K.variable(np.full(fill_value=1e-8, shape=shape)) for shape in shapes]
            reward = [K.zeros(shape) for shape in shapes]
            tilde_w = [K.zeros(shape) for shape in shapes]
            gradients_sum = [K.zeros(shape) for shape in shapes]
            gradients_norm_sum = [K.zeros(shape) for shape in shapes]
            for p, g, li, ri, twi, gsi, gns in zip(params, grads, L, reward,
                                                 tilde_w,gradients_sum,
                                                   gradients_norm_sum):
                grad_sum_update = gsi + g
                grad_norm_sum_update = gns + K.abs(g)
                l_update = K.maximum(li, K.abs(g))
                reward_update = K.maximum(ri - g * twi, 0)
                new_w = - grad_sum_update / (l_update * (K.maximum(grad_norm_sum_update + l_update, self._alpha * l_update))) * (reward_update + l_update)
                param_update = p - twi + new_w
                tilde_w_update = new_w
                self.updates.append(K.update(gsi, grad_sum_update))
                self.updates.append(K.update(gns, grad_norm_sum_update))
                self.updates.append(K.update(li, l_update))
                self.updates.append(K.update(ri, reward_update))
                self.updates.append(K.update(p, param_update))
                self.updates.append(K.update(twi, tilde_w_update))
            return self.updates
    def get_config(self):
            config = {'alpha': float(K.get_value(self._alpha)) }
            base_config = super(COCOB, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

word_input = Input(shape=(max_senten_len,), dtype='int32')
word = embedding_layer(word_input)
word = SpatialDropout1D(0.2)(word)
word = Bidirectional(LSTM(128, return_sequences=True))(word)
word_out = AttentionWithContext()(word)
wordEncoder = Model(word_input, word_out)

sente_input = Input(shape=(max_senten_num, max_senten_len), dtype='int32')
sente = TimeDistributed(wordEncoder)(sente_input)
sente = SpatialDropout1D(0.2)(sente)
sente = Bidirectional(LSTM(128, return_sequences=True))(sente)
sente = AttentionWithContext()(sente)
preds = Dense(6, activation='sigmoid')(sente)
model = Model(sente_input, preds)
#opt = optimizers.Adam(clipnorm=5.0)

model.compile(loss='binary_crossentropy',
              optimizer=COCOB(),
              metrics=['acc'])

model.summary()

X_train_sent = np.asarray(X_train_sent)
X_test_sent = np.asarray(X_test_sent)

batch_size = 256

X_tra, X_val, y_tra, y_val = train_test_split(X_train_sent, y_train, train_size=0.95, random_state=233)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
history = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=1, 
          validation_data=(X_val, y_val),callbacks = [RocAuc,early], verbose=1)
model_json = model.to_json()
with open("han_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("han_model.h5")
print("Saved model to disk")
with open('tokenizer.pickle','wb') as handle:
    pickle.dump(tok,handle,protocol=pickle.HIGHEST_PROTOCOL)

print()
scores = model.evaluate(X_test_sent, y_test, verbose=1)
y_pred = model.predict(X_test_sent, verbose=1)
with open('ytest.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(y_test)
csvFile.close()
with open('ypred.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(y_pred)
csvFile.close()
   
print('Test loss: ', scores[0])
print('Test accuracy: ', scores[1])

json_file = open('han_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json,custom_objects={'AttentionWithContext':AttentionWithContext})
# load weights into new model
loaded_model.load_weights("han_model.h5")
print("Loaded model from disk");
y_pred = loaded_model.predict(X_val,verbose=1)
fpr=[0,0,0,0,0,0]
tpr=[0,0,0,0,0,0]
for i in range(6):
    fpr[i], tpr[i], _ = roc_curve(y_val[:,i], y_pred[ :,i])
    

plt.figure()
plt.plot(fpr[0], tpr[0], label='ROC curve')
plt.plot([0, 1], [0, 1])
plt.xlim([0, 1.0])
plt.ylim([0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve: Class - TOXIC')
plt.legend(loc="lower right")
plt.show()

plt.figure()
plt.plot(fpr[1], tpr[1], label='ROC curve')
plt.plot([0, 1], [0, 1])
plt.xlim([0, 1.0])
plt.ylim([0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve: Class - SEVERE TOXIC')
plt.legend(loc="lower right")
plt.show()

plt.figure()
plt.plot(fpr[2], tpr[2], label='ROC curve')
plt.plot([0, 1], [0, 1])
plt.xlim([0, 1.0])
plt.ylim([0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve: Class - OBSCENE')
plt.legend(loc="lower right")
plt.show()

plt.figure()
plt.plot(fpr[3], tpr[3], label='ROC curve')
plt.plot([0, 1], [0, 1])
plt.xlim([0, 1.0])
plt.ylim([0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve: Class - THREAT')
plt.legend(loc="lower right")
plt.show()

plt.figure()
plt.plot(fpr[4], tpr[4], label='ROC curve')
plt.plot([0, 1], [0, 1])
plt.xlim([0, 1.0])
plt.ylim([0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve: Class - INSULT')
plt.legend(loc="lower right")
plt.show()

plt.figure()
plt.plot(fpr[5], tpr[5], label='ROC curve')
plt.plot([0, 1], [0, 1])
plt.xlim([0, 1.0])
plt.ylim([0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve: Class - IDENTITY HATE')
plt.legend(loc="lower right")
plt.show()