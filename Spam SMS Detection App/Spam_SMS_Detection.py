import streamlit as st
import pickle
#loading the trained model
model= pickle.load(open('model.pkl','rb'))

st.title("Spam SMS Detection App")
message= st.text_input("Enter the message")
submit= st.button('predict')
if submit:
    prediction= model.predict([message])
    if prediction[0]== 'spam':
        st.warning("Given message is Spam")
    else:
        st.success("Given message is Ham")

st.balloons()