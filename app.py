import helper
import streamlit as st
import pickle

model = pickle.load(open('C:/Users/Right Click/Desktop/AMIT AI/final projects/project1/artifacts/dt.pkl', 'rb'))

text = st.text_input("Enter your text here")

if st.button('Predict'):
    if text.strip():  # Ensure text is not empty
        text = helper.text_preprocessing(text)
        prediction = model.predict(text)
        st.write(f"Prediction: {prediction[0]}")
    else:
        st.write("Please enter some text")