# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 14:35:48 2022

@author: onkar
"""

import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk import word_tokenize
import re
import pandas as pd
import contractions
import inflect
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tabulate import tabulate
import plotly.graph_objects as go
import plotly.express as px


pickle_in = open('D:\\Data Science\\All Projects\\ExcelR Projects Final file\\Hotel_Review_Classifier NLP (DS_P_139)\\lgb_model.pkl', 'rb')
lgb_model = pickle.load(pickle_in)

def main():
	st.title("Sentiment Analysis NLP")
	st.subheader("Analyze Hotel reviews as positive or negative")

if __name__ == '__main__':
	main()
    
    
stop_words = stopwords.words('english') # remove stop words

def get_percentage(num):
    return "{:.2f}".format(num*100)


def number_to_text(data):
    temp_str = data.split()
    string = []
    for i in temp_str:

    # if the word is digit, converted to
    # word else the sequence continues

        if i.isdigit():
            temp = inflect.engine().number_to_words(i)
            string.append(temp)
        else:
            string.append(i)
    outputStr = " ".join(string)
    return outputStr


ps = PorterStemmer()
def stem_text(data):
    tokens = word_tokenize(data)
    stemmed_tokens = [ps.stem(word) for word in tokens if word not in (stop_words)]
    return " ".join(stemmed_tokens)

lemma = WordNetLemmatizer()
def lemmatiz_text(data):    
    tokens = word_tokenize(data)
    lemma_tokens = [lemma.lemmatize(word, pos='v') for word in tokens if word not in (stop_words)]
    return " ".join(lemma_tokens)

def cleantext(text):
    text = re.sub(r'[^\w\s]', " ", text) # Remove punctuations
    text = re.sub(r"https?:\/\/\S+", ",", text) # Remove The Hyper Lin
    text = contractions.fix(text) # remove contractions 
    text = number_to_text(text) # convert numbers to text    
    text = text.lower() # convert to lower case
    # don't feel it's worth to use stemming as it may lead to some wrong words
    text = lemmatiz_text(text) # lemmatization
    return text

hotelReviewText = st.text_area('Enter the Hotel review text', '')

result=""
if st.button("Predict"):
   result=cleantext(hotelReviewText)
   st.success('Review is  {}'.format(result))   