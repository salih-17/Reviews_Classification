
# Import Libraries 
#-----------------------------------
# Import Text cleaning function
import streamlit as st
from Cleaning import clean
from tensorflow.keras.preprocessing import sequence
from transformers import BertTokenizer
from transformers import RobertaTokenizer
import pickle
import numpy as np
from keras.models import load_model
from numpy import argmax
from scipy import stats as stt
import pandas as pd
# -------------------------------
with open('tokenizer.pickle', 'rb') as handle:
    BiLSTM_tokenizer = pickle.load(handle)
    
BiLSTM_model = load_model('BiLSTM_CNN.h5')

Bert_tokenizer = BertTokenizer.from_pretrained('BertTokenizer')
Bert_model = load_model('BertCnn.h5')

# Load BERT tokenizer
Robert_tokenizer = RobertaTokenizer.from_pretrained('RoBertTokenizer')
RoBert_model = load_model('RoBertCnn.h5')
# -------------------------------
Class = {0:'Non-Spam' , 1 : 'Spam'}
# -------------------------------
def BiLSTM(text):
    sequences = BiLSTM_tokenizer.texts_to_sequences(text)
    sequences = sequence.pad_sequences(sequences, maxlen=50)
    Result = BiLSTM_model.predict(sequences)
    return Result  

def BertCNN (text):
    Bert_Sequences = Bert_tokenizer(text, padding= 'max_length' , truncation=True, max_length=80)
    Result = Bert_model.predict(Bert_Sequences['input_ids'])
    return Result  

def RoBertCNN (text):
    Robert_Sequences = Robert_tokenizer(text, padding= 'max_length' , truncation=True, max_length=80)
    Result = RoBert_model.predict(Robert_Sequences['input_ids'])
    return Result

def calssification (text):
    # Classification using classifier
    BiLSTM_Result = BiLSTM(text)
    Bert_Result = BertCNN (text)
    RoBertCNN_Result = RoBertCNN(text)

    # Calculate the result
    BiLSTM_output=np.argmax(BiLSTM_Result,axis=1)
    Bert_output=np.argmax(Bert_Result,axis=1)
    RoBertCNN_output=np.argmax(RoBertCNN_Result,axis=1)
    Result = Class[stt.mode ([BiLSTM_output ,Bert_output ,RoBertCNN_output  ])[0][0]]

    # Creat the dataframe
    BiLSTM_H = int (BiLSTM_Result[0][0]*100) 
    BiLSTM_S = int (BiLSTM_Result[0][1]*100)

    Bert_H = int (Bert_Result[0][0]*100) 
    Bert_S = int (Bert_Result[0][1]*100)

    Roberta_H = int(RoBertCNN_Result[0][0]*100) 
    Roberta_S = int (RoBertCNN_Result[0][1]*100)

    ix = ['BiLSTM','Bert','RoBerta']

    columns = {'Non-Spam':[BiLSTM_H ,Bert_H ,  Roberta_H] , 'Spam':[BiLSTM_S ,Bert_S ,  Roberta_S]}

    df = pd.DataFrame(columns  , index=ix )

    df['Non-Spam'] = df['Non-Spam'].apply( lambda x : str(x) + '%')
    df['Spam'] = df['Spam'].apply( lambda x : str(x) + '%')

    return (Result , df)
    #print (df)

#---------------------------------------------------------------------

prompt = st.chat_input("Say something")
if prompt:
    st.write('Input:')
    st.write(prompt)

    st.write('Cleaned Text:')
    CleanedText = clean (prompt)
    st.write(CleanedText)

    st.write("Result:")
    CL , df = calssification([CleanedText])
    st.warning(CL , icon="⚠️")

    st.write("The probabilities of classifiers")
    st.dataframe(df)
