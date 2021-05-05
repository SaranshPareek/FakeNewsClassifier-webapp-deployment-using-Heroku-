#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 18:36:34 2021

@author: saransh
"""

from flask import Flask,render_template,request,url_for
import pickle 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app=Flask(__name__)
classifier=pickle.load(open('classifier.pkl','rb'))
cv=pickle.load(open('transform.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        news=request.form['news']
        data=[news]
        vectors=cv.transform(data).toarray()
        my_prediction=classifier.predict(vectors)
    return render_template('result.html',prediction=my_prediction)



if __name__ == "__main__":
    app.run(debug=True)
    
    