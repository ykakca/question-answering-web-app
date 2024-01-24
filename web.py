# -*- coding: utf-8 -*-




from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import streamlit as st

@st.cache

st.title("BERT İle Türkçe Soru Cevaplama Sistemi")
st.divider()
st.caption("Yıldız Teknik Üniversitesi")
st.caption("Bilgisayar Mühendisliği Bölümü Bitirme Projesi")
st.caption("Selin Tipi, 19011051 - Yiğit Kağan Akça, 19011103")


model_name = "ykakca/bert-bitirme-128k"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

question_input = st.text_area("Soru: ")
context_input = st.text_area("Metin: ")

if question_input and context_input:
  #keywords = question_input.split()

  QA_input = {
      'question': question_input,
      'context': context_input
  }

  res = nlp(QA_input)

  st.text_area("Cevap: ", res['answer'])
  st.write("Score: ", res['score'])

#  !npm install localtunnel

#!streamlit run streamlitDemoColab.py &>/content/logs.txt &

#import urllib
#print("Password/Enpoint IP for localtunnel is:",urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n"))

#!npx localtunnel --port 8501

