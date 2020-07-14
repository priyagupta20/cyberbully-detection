import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

train=pd.read_csv("train.csv")
print(train.shape)
train["comment_text"].fillna("fillna")
X = train["comment_text"].str.lower()
y = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X = list(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('split done')
tfv = TfidfVectorizer(min_df=3, max_df=0.9, max_features=None, strip_accents='unicode',\
               analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), use_idf=1,\
               smooth_idf=1, sublinear_tf=1, stop_words='english')
print("tfidf-vectorizing train ...")
tfv.fit(X_train)
X_train = tfv.transform(X_train)
print("tfidf-vectorizing test ...")
X_test = tfv.transform(X_test)

model_lr = LogisticRegression(C=1, solver = 'sag')
model_lr.fit(X_train,y_train[:,0])
y_pred = model_lr.predict(X_test)
print("Logistic regression:")
print(classification_report(y_test[:,0],y_pred))
print("auc:", roc_auc_score(y_test[:,0],y_pred))

model_nb = MultinomialNB()
model_nb.fit(X_train,y_train[:,0])
y_pred_nb = model_nb.predict(X_test)
print("Naive Bayes:")
print(classification_report(y_test[:,0],y_pred_nb))
print("auc:", roc_auc_score(y_test[:,0],y_pred_nb))

model_rf = RandomForestClassifier()
model_rf.fit(X_train,y_train[:,0])
y_pred_rf = model_rf.predict(X_test)
print("Random Forest:")
print(classification_report(y_test[:,0],y_pred_rf))
print("auc:", roc_auc_score(y_test[:,0],y_pred_rf))