import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
from PIL import Image


df_titanic = pd.read_csv("Titanic-Dataset.csv")
df_titanic.head()
df_titanic.drop('Cabin',axis=1,inplace=True)
df_titanic.dropna(inplace=True)
df_titanic.isnull().sum()

df_titanic.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)
X = df_titanic.drop(['Survived','PassengerId','Name','Ticket'],axis=1)
y = df_titanic['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
lg = LogisticRegression()
lg.fit(X_train,y_train)
y_pred = lg.predict(X_test)
socre = accuracy_score(y_pred,y_test)


# web app
img = Image.open('img.jpeg')
st.image(img,width=500)
st.title("Titanic Classification Prediction Model")
input_txt = st.text_input("Enter Titanic passenger features")
input_txt_splt = input_txt.split(',')

try:
    features = np.asarray(input_txt_splt,dtype=float)
    prediction = lg.predict(features.reshape(1,-1))
    if prediction[0] == 0:
        st.write("Survived")
    else:
        st.write("Not Survived")
except ValueError:
    st.write("Please Input Comma separated features")
    st.write("\n about data : \n", df_titanic)
    st.write("\n")