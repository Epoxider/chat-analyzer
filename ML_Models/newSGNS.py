import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.layers import Embedding, Dot, Flatten, Input
from tensorflow.keras.models import Model
import tensorflow as tf

# Load and preprocess the data
df = pd.read_csv('../ChatData/data_set/d132e42bde1b8d87d58fcfb8838429ce697c3ef4_6.csv')
df.dropna(inplace=True)

# Tokenize the text messages
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Message'])
sequences = tokenizer.texts_to_sequences(df['Message'])

# Tokenize the user
user_tokenizer = Tokenizer()
user_tokenizer.fit_on_texts(df['User'])
user_sequences = user_tokenizer.texts_to_sequences(df['User'])

# Generate skip-grams
word_target, word_context, labels = [], [], []
for i, seq in enumerate(sequences):
    target, context, label = skipgrams(seq, vocabulary_size=len(tokenizer.word_index) + 1, window_size=5, negative_samples=1.0)
    word_target.extend(target)
    word_context.extend(context)
    labels.extend(label)

word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")
labels = np.array(labels, dtype="int32")

# User sequences for each skip-gram
user_sequences_skipgrams = []
for i, target in enumerate(word_target):
    user_sequences_skipgrams.append(user_sequences[i // len(sequences)])

user_sequences_skipgrams = np.array(user_sequences_skipgrams, dtype="int32")

# Model definition
input_target = Input((1,))
input_context = Input((1,))
input_user = Input((1,))

embedding_dim = 128

word_embedding = Embedding(len(tokenizer.word_index) + 1, embedding_dim, input_length=1)(input_target)
context_embedding = Embedding(len(tokenizer.word_index) + 1, embedding_dim, input_length=1)(input_context)
user_embedding = Embedding(len(user_tokenizer.word_index) + 1, embedding_dim, input_length=1)(input_user)

dot_product = Dot(axes=2)([word_embedding, context_embedding])
dot_product = Flatten()(dot_product)

# Incorporate user embedding
merged = tf.keras.layers.Concatenate(axis=-1)([dot_product, Flatten()(user_embedding)])
output = tf.keras.layers.Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[input_target, input_context, input_user], outputs=output)

# Custom loss with time decay
def time_decay_loss(y_true, y_pred):
    # Assuming df['Time'] is a timestamp, and we normalize it into a decay factor between 0 and 1
    decay_factor = (df['Time'] - df['Time'].min()) / (df['Time'].max() - df['Time'].min())
    decay_factor = decay_factor.values
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) * decay_factor

model.compile(loss=time_decay_loss, optimizer='adam')

# Model training
model.fit([word_target, word_context, user_sequences_skipgrams], labels, epochs=10, batch_size=256)
