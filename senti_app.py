import io
import streamlit as st
import numpy as np
from transformers import pipeline

classifier = pipeline("sentiment-analysis",
                      "snunlp/KR-FinBert-SC")

form = st.form(key='my_form')
text = form.text_input(label='Enter some text')
submit_button = form.form_submit_button(label='Submit')

def create_score_text(clsfr, txt):
    result = clsfr(txt)
    st.write('Оценка фразы: ' + result[0].get('label', 'no res'))
    if result[0]['score'] >= 0.75:
        return 'Верим (score >= 0.75)'
    else:
        return 'Но это не точно (score < 0.75)'

if submit_button:
    res_func = create_score_text(classifier,text)
    st.write(res_func)
