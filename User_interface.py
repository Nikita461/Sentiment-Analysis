from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from nltk.tokenize import word_tokenize
import string
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import sqlite3
from googletrans import Translator
import warnings
translator = Translator()
IMAGE_FOLDER = os.path.join('static', 'img_pool')
 app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
# with open('tokenizer.pickle', 'rb') as handle:
# tokenizer = pickle.load(handle)
def init():
 global model,graph
 graph = tf.Graph()

@app.route("/")
def home():
 return render_template("home.html")
@app.route('/logon')
def logon():
return render_template('signup.html')
@app.route('/login')
def login():
return render_template('signin.html')
@app.route("/signup")
def signup():
 username = request.args.get('user','')
 name = request.args.get('name','')
 email = request.args.get('email','')
 number = request.args.get('mobile','')
 password = request.args.get('password','')
 con = sqlite3.connect('signup.db')

 cur = con.cursor()
 cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?,
?)",(username,email,password,number,name))
 con.commit()
 con.close()
 return render_template("signin.html")
@app.route("/signin")
def signin():
 mail1 = request.args.get('user','')
 password1 = request.args.get('password','')
 con = sqlite3.connect('signup.db')
 cur = con.cursor()
 cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
 data = cur.fetchone()
 if data == None:
 return render_template("signin.html")
 elif mail1 == 'admin' and password1 == 'admin':
 return render_template("index.html")
 elif mail1 == str(data[0]) and password1 == str(data[1]):
 return render_template("index.html")
 else:
 return render_template("signup.html")
@app.route('/index')
def index():
return render_template('index.html')
@app.route('/about')
def about():
return render_template('about.html')
@app.route('/sentiment_prediction', methods = ['POST', "GET"])
def sent_anly_prediction():

 if request.method=='POST':
 text = request.form['text']
 translations = translator.translate(text, dest='en')
 text = translations.text
 with open(r'C:\Users\peddi\OneDrive\Documents\sentiment\clean_text.pkl', 'rb') as f:

 test_data = pickle.load(f)
 with open(r'C:\Users\peddi\OneDrive\Documents\sentiment\xgb_model.pkl', 'rb') as f:
 xgb = pickle.load(f)
 # Append the input with clean text file
 test_data.append(text)
 # load the pca,vectorizer pickle object
 with open(r'C:\Users\New\OneDrive\Documents\sentiment\pca.pkl', 'rb') as f:
 pca = pickle.load(f)
 with open(r'C:\Users\New\OneDrive\Documents\sentiment\vectorizer.pkl', 'rb') as f:
 vectorizer = pickle.load(f)
 test_data=vectorizer.fit_transform(test_data)
 test_data = pca.fit_transform(test_data)

 prediction = xgb.predict(test_data[-1].reshape(1,-1))

 probability=xgb.predict_proba(test_data[-1].reshape(1,-1))[0][prediction]
 templateData = {

 'temp' : prediction

 }

 if prediction == 0:
 sentiment = 'Negative'
 #probability=0
 img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Sad_Emoji.png')
 elif prediction == 1:
 sentiment = 'Positive'
 #probability=1
 img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Smiling_Emoji.png')
 else:
 sentiment = 'Neutral'
 #probability=2
 img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'neutral_face.png')

 return render_template('index.html', text=text, sentiment=sentiment, probability=probability, image=img_filename,
**templateData)
if __name__ == '__main__':
 init()
 app.run(debug=False)
