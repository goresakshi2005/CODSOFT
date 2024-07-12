import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# load data
credit= pd.read_csv("D:/Credit Card Fraud Detection App/creditcard.csv", encoding= "latin1")

# separate fraudlent and legitimate transactions
legit= credit[credit.Class==0]
fraud= credit[credit.Class==1]

# undersample legitimate transactions to balance the classes
legit_sample= legit.sample(n= len(fraud), random_state= 2)
credit= pd.concat([legit_sample,fraud],axis= 0)

# split data into training and testing sets
X= credit.drop(columns= 'Class',axis= 1)
Y= credit['Class']
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size= 0.2,stratify= Y,random_state= 2)

# train logistic regression model
model= LogisticRegression()
model.fit(X_train, Y_train)

# evaluate model performance
train_acc= accuracy_score(model.predict(X_train),Y_train)
test_acc= accuracy_score(model.predict(X_test),Y_test)

# Web App

st.title("Credit Card Fraud Detection Model")
input_df= st.text_input("Enter All Required Features Values")
input_df_splited= input_df.split(',')

submit= st.button("Submit")

if submit:
    features= np.asarray(input_df_splited,dtype= np.float64)
    prediction= model.predict(features.reshape(1,-1))
    
    if prediction[0]== 0:
        st.write("Legitimate Transaction")
    else:
        st.write("Fraudlant Transaction")
        
    




























