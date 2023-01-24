from fastapi import FastAPI
from transformers import pipeline
import streamlit as st

app = FastAPI()
classifier = pipeline("sentiment-analysis",   
                      "blanchefort/rubert-base-cased-sentiment")

st.write(classifier(st.text_input('Введите фразу для оценки...')))
