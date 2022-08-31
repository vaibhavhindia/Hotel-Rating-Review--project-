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
from lightgbm import LGBMClassifier
import lightgbm as lgb

pickle_in = open("D:\\Data Science\\All Projects\\ExcelR Projects Final file\\Hotel_Review_Classifier NLP (DS_P_139)\\lr_model.pkl", 'rb')
model_sentiment = pickle.load(pickle_in)

pickle_in_tdidf = open("D:\\Data Science\\All Projects\ExcelR Projects Final file\\Hotel_Review_Classifier NLP (DS_P_139)\\model_sentiment_tfidf.pkl", 'rb') 
model_tfidf = pickle.load(pickle_in_tdidf)



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



if st.button("Analyze"):
    cleanReviewText = cleantext(hotelReviewText)
    #st.write(cleanReviewText)    
    tfIdfText = model_tfidf.transform([cleanReviewText])
    predictedVal=model_sentiment.predict(tfIdfText)
    predictedVal= predictedVal[0]
    #st.write(type(predictedVal))
    
    st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
        }
        
        </style>
        """, unsafe_allow_html=True)


    finalPrediction = "The review is positive 😀" if predictedVal=="1" else "The review is negative 😔"
      
    st.markdown("<p class='big-font'>{}</p>".format(finalPrediction),unsafe_allow_html=True)
    
           
    prdictionDist = model_sentiment.lr_model.predict(..., pred_leaf = True)(tfIdfText)

    resN = get_percentage(prdictionDist[0][0])
    resP = get_percentage(prdictionDist[0][1])


    dfRes = pd.DataFrame(columns=['Negative', 'Positive'])
    dfRes.loc[1, 'Negative'] = resN+"%"
    dfRes.loc[1, 'Positive'] = resP+"%"



    # CSS to inject contained in a string
    hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """

    # Inject CSS with Markdown
    st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

    st.dataframe(dfRes)


    #show pie chart
    reviewLabels = dfRes.keys()
    reviewPer = [resN,resP]
    colors=["#F72F35","#00A267"]
    
    fig = go.Figure(
    go.Pie(
    labels = reviewLabels,
    values = reviewPer,
    hoverinfo = "label+percent",
    textinfo = "percent",
    ))


    fig.update_layout(width=400, height=400,margin=dict(t=0, b=0, l=0, r=0))
    fig.update_traces(marker = dict(colors = colors))   
       
    st.header("Pie chart")
    st.plotly_chart(fig)