# DATA ACQUISITION

import pandas as pd
import numpy as np
from collections import Counter
import nltk
import re as regex
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from time import time
import matplotlib.pyplot as plt
   train_data = pd.read_csv('data/Tweets.csv')
test_data = pd.read_csv('data/test.csv')
test_data.rename(columns={'Category': 'Tweet'}, inplace=True)

#DATA CLEANING
 	   train_data = train_data[train_data['text'] != "Not Available"]
def clean_tweets(tweet):
    tweet = re.sub(r"http\S+", "", tweet)
    
   tweet = re.sub(r"@[^\s]+[\s]?",'',tweet)
    
   
    tweet = re.sub('[^ a-zA-Z0-9]', '', tweet)
    
 
    tweet = re.sub('[0-9]', '', tweet)
    
    return tweet
train_data['text'] = train_data['text'].apply(clean_tweets)

#TOKENIZATION AND STEMMING
   from nltk.tokenize import TweetTokenizer
   tt = TweetTokenizer()
train_data['text'].apply(tt.tokenize)

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()
def tokenize(text):
    return word_tokenize(text)

def stemming(words):
    stem_words = []
    for w in words:
        w = ps.stem(w)
        stem_words.append(w)
    
    return stem_words
# apply tokenize function
train_data['text'] = train_data['text'].apply(tokenize)
# apply stemming function
  train_data['tokenized'] = train_data['text'].apply(stemming)

#STOPPING
   words = Counter()
for idx in train_data.index:
    words.update(train_data.loc[idx, "text"])
stopwords=nltk.corpus.stopwords.words("english")
whitelist = ["n't", "not"]
for idx, stop_word in enumerate(stopwords):
    if stop_word not in whitelist:
        del words[stop_word]

#VECTORIZATION

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(clean_text)

from sklearn.decomposition import PCA,TruncatedSVD
pca = TruncatedSVD(n_components=15)
X = pca.fit_transform(X)

#BUILDING THE MODEL

from xgboost import XGBClassifier as XGBoostClassifier
X_train, X_test, y_train, y_test = train_test_split(X, train_data.airline_sentiment, test_size=0.3)
xgb = XGBoostClassifier(eval_metric="mlogloss",silent=True,max_depth=5)
xgb.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test,y_test)])

#TESTING THE MODEL

predictions = xgb.predict(X_test)
test_data=list(clean_text)
test_data=vectorizer.fit_transform(test_data)
test_data = pca.fit_transform(test_data)
pred = xgb.predict(test_data[-1].reshape(1,-1))
xgb.predict_proba(test_data[-1].reshape(1,-1))[0][pred]
  def test_classifier(X_train, y_train, X_test, y_test, classifier):
    log("")
    log("---------------------------------------------------------")
    log("Testing " + str(type(classifier).__name__))
    now = time()
    list_of_labels = sorted(list(set(y_train)))
    model = classifier.fit(X_train, y_train)
    log("Learing time {0}s".format(time() - now))
    now = time()
    predictions = model.predict(X_test)
    log("Predicting time {0}s".format(time() - now))

    # Calculation Accuracy, Precision, recall
    
    precision = precision_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
    recall = recall_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
    
    log("=================== Results ===================")
    log("            Negative     Neutral     Positive")
    log("F1       " + str(f1))
    log("Precision" + str(precision))
    log("Recall   " + str(recall))
    log("Accuracy " + str(accuracy))
    log("===============================================")

    return precision, recall, accuracy, f1

#MAKING PREDICTIONS

from xgboost import XGBClassifier as XGBoostClassifier
precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, XGBoostClassifier())
