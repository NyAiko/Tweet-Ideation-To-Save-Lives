import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from text_cleaning import normalize_text
import math


st.title('Text Analysis - Detect Potential Suicide in Tweet')
text = st.text_input("Enter text here: ")
model = joblib.load('model.p')

if len(text.split())>0:
    text = normalize_text(text)
    prediction = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    confidence_score = int(proba[prediction]*100)
    label = ['Not suicide','Potential Suicide']
    st.write(f'Analysis Result :  {label[prediction]}')
    st.write(f'Confidence score  : {confidence_score}%')
