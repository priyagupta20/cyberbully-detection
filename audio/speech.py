import speech_recognition as sr 
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras import backend as K
from keras.models import model_from_json
import nltk
from keras.preprocessing import sequence
import numpy as np
import pickle
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
file = input("Enter the name of audio file:")
AUDIO_FILE = (file) 

r = sr.Recognizer() 

with sr.AudioFile(AUDIO_FILE) as source: 
	audio = r.record(source) #read audio file

try: 
    text = r.recognize_google(audio)
    #print("The audio file contains: " + text) 
    texts=[]
    texts.append(text)
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
    with open('tokenizer.pickle','rb') as handle:
        tk=pickle.load(handle)
    texts = filt_sent(texts ,10)
    for i in range(len(texts)):
            texts[i] = tk.texts_to_sequences(texts[i])
            texts[i] = sequence.pad_sequences(texts[i],maxlen=30)
    texts=np.asarray(texts)
    y =loaded_model.predict(texts);
    print()
    print("Prediction:")
    print()
    print("------------------------------------------------")
    print("Toxic:", y[0][0])
    print("Severe toxic:", y[0][1])
    print("Obscene:", y[0][2])
    print("Threat:", y[0][3])
    print("Insult:", y[0][4])
    print("Identity hate:", y[0][5])
    print("------------------------------------------------")
    print()
    print("Cyberbully: ",end="")
    y[0].sort()
    print((y[0][5]+y[0][4])/2)
    print()
    print("------------------------------------------------")


except sr.UnknownValueError: 
	print("Google Speech Recognition could not understand audio") 

except sr.RequestError as e: 
	print("Could not request results from Google Speech Recognition service; {0}".format(e)) 