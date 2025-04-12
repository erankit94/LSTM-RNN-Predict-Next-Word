import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load model
model= load_model('next_word_lstm.h5')

# load tokeniser
with open("tokenizer.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)

def pred_word(model, tokensizer, text, max_seq_len):
    token_list = tokensizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_seq_len:
        token_list = token_list[-(max_seq_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    pred_word_index = np.argmax(predicted, axis=1)
    for word, index in tokensizer.word_index.items():
        if index==pred_word_index:
            return word
    return None

# steamlit
st.title("Next word prediction with LSTM")
input_text  =st.text_input("Enter the seq of words", 'to be or not to be')
if st.button("Predict next word"):
    max_seq_len = model.input_shape[1]+1
    next_word = pred_word(model, tokenizer, input_text, max_seq_len)
    st.write(f"next word: {next_word}")
    
             

