import numpy as np
import pandas as pd


import os
import string
import matplotlib.pyplot as plt
%matplotlib inline
import re

from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense
from keras.layers import TimeDistributed
from keras.models import Model

print(os.listdir("../input"))

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', -1)
df=pd.read_csv("/kaggle/input/amazon/Reviews.csv",encoding='utf-8')
# 4,00,000 records used in this project
df=df[0:400000]
# Unique attributes are removed like Id,ProductId,UserId,Time,ProfileName,HelpfulnessNumerator,HelpfulnessDenominator, and Score

k=['Id','ProductId','UserId','Time','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator','Score']
for i in k:
    df=df.drop(i, axis=1)
df
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense
from keras.layers import TimeDistributed
from keras.models import Model
import re

# Text data is processed

def preprocess_text(text):
    text = str(text)
    text = re.sub(r'[^a-zA-Z0-9]',' ',text)
    return text

df.Summary=df.Summary.apply(preprocess_text)
df.Text=df.Text.apply(preprocess_text)

def preprocess_text(text):
    text = 'sostok '+text+' eostok'
    return text

df.Summary=df.Summary.apply(preprocess_text)
# Text data is converted into tokens

summarytokenizer = Tokenizer(num_words=None)
summarytokenizer.fit_on_texts(df['Summary'])
df['Summary'] = summarytokenizer.texts_to_sequences(df['Summary'])

texttokenizer = Tokenizer(num_words=None)
texttokenizer.fit_on_texts(df['Text'])
df['Text'] = texttokenizer.texts_to_sequences(df['Text'])

summary_vocabulary_size_val = len(summarytokenizer.index_word)+1
text_vocabulary_size_val = len(texttokenizer.index_word)+1
df
text_length_val=max([len(i) for i in df["Text"]])
summary_length_val=max([len(i) for i in df["Summary"]])
text_length_val
summary_length_val
# Number of words of each text and summary is calculated
df['length_text']=df['Text'].apply(lambda x:len(x))
df['length_summary']=df['Summary'].apply(lambda x:len(x))
df.head()
sum_voc   =  len(summarytokenizer.word_index)+1
print("Size of vocabulary in X = {}".format(sum_voc))
text_voc   =  len(texttokenizer.word_index)+1
print("Size of vocabulary in X = {}".format(text_voc))
df
X, y = df[['Text']], df['Summary']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True,
                                                    random_state=42)
X_train.shape, X_val.shape
y_train
df['length_text'].quantile(0.95)
df['length_summary'].quantile(0.95)
max_length_src=(int)(df['length_text'].quantile(0.95))
max_length_tar=(int)(df['length_summary'].quantile(0.95))
x_train    =   pad_sequences(X_train.Text,  maxlen=max_length_src, padding='post')
x_val   =   pad_sequences(X_val.Text, maxlen=max_length_src, padding='post')
y_train    =   pad_sequences(y_train,  maxlen=max_length_tar, padding='post')
y_val   =   pad_sequences(y_val, maxlen=max_length_tar, padding='post')
del(df,X,y)
# Encoder
embedding_dim = 1000
latent_dim = 1000

encoder_inputs = Input(shape=(max_length_src,))

# Embedding Layer
enc_emb =  Embedding(text_voc, embedding_dim)(encoder_inputs)

# LSTM layer using the input product description
encoder_lstm = LSTM(latent_dim, return_state=True, recurrent_dropout=0.2)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

# The encoder lstm states are taken to the decoder
encoder_states = [state_h, state_c]
decoder_inputs = Input(shape=(None,))

# Embedding layer
dec_emb_layer = Embedding(sum_voc, embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)

# The decoder lstm using summary
# Set up the decoder, using 'encoder_states' as initial input.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, recurrent_dropout=0.2)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

# Dense layer
decoder_dense = TimeDistributed(Dense(sum_voc, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# compile the model
# using sparse categorical crossentropy as we are using time distributed output

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
model = tf.keras.models.load_model("/kaggle/input/model9/model.hdf5")
#model = create_model()
#model.load_weights('/kaggle/input/model9/model.hdf5')
model.summary()
train_samples = len(x_train)
val_samples = len(x_val)
batch_size = 128
epochs = 12
history = model.fit([x_train,y_train[:,:-1]], y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:,1:], 
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data = ([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]),
                  validation_steps = val_samples//batch_size
                   )
import tensorflow.keras.models 
model.save_weights('my_checkpoint.model')
model.save('model.hdf5')
from matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.title('Train and Validation Loss Representation')
pyplot.xlabel("Number of epochs")
pyplot.ylabel("Sparse categorical crossentropy loss")
pyplot.legend()
pyplot.show()

reverse_target_word_index = summarytokenizer.index_word
reverse_source_word_index = texttokenizer.index_word
target_word_index = summarytokenizer.word_index
reverse_target_word_index
# Encoder model to get the lstm states for the input sequence
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder gets lstm states
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Get the embeddings of the decoder sequence
dec_emb2 = dec_emb_layer(decoder_inputs)

# To predict the next word in the sequence,
# set the initial states to the states from the previous time step which in this is sostok
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, 
                                                    initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]

# A dense softmax layer to generate probability distribution over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_outputs2) 

# Define Decoder model
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)
decoder_model.summary()
def decode_sequence(input_seq):
    # Get the states for the previous time step usng input sequence
    e_h, e_c = encoder_model.predict(input_seq)
    
    target_seq = np.zeros((1,1))
    
    # Set first word as sostok since all the summary starts with sostok
    target_seq[0, 0] = target_word_index['sostok']

    # Predict word and states of next time step
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_h, e_c])

        # Finding word with highest probability distribution
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        # Exit condition: if eostok is predicted, all summary ends with eostok \
        # and there is no need to predict anymore
        
        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token
            
        # Exit condition: hit max length.
        if (sampled_token == 'eostok' or len(decoded_sentence.split()) >= (max_length_tar-1)):
            stop_condition = True

        # Update the target sequence with the new predicted token.
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states for next time step
        e_h, e_c = [h, c]

    return decoded_sentence
def descriptext(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString

def summarytext(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_target_word_index[i]+' '
    return newString
for i in range(0,10):
    print("Description Sentence: ",descriptext(x_train[i]))
    print('Summary Translation:', summarytext(y_train[i]))
    print("Predicted Summary Translation: ",decode_sequence(x_train[i].reshape(1,max_length_src)))
    print("\n")

input_text = 'Phi starts with a rich fruity line of Apricot. An all natural apricot extract with its surprising richness enchants and blends into a Cinnamon line and hints of bitter almond, softened by Bergamot essential oil.'
ip_seq = texttokenizer.texts_to_sequences([input_text])
ip_pad = pad_sequences(ip_seq,  maxlen=max_length_src, padding='post')
print("Description Sentence: ", input_text)
print("Predicted Summary Translation: ", decode_sequence(ip_pad[0].reshape(1,max_length_src)))
