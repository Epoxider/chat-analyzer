import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.layers import Embedding, Dot, Flatten, Input, Dense, Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf

# Load and preprocess the data
df = pd.read_csv('./ChatData/data_set/0b7f9f8e3f811e4e5ce8ac43975c7beeab1fe829_2.csv')
df.dropna(inplace=True)

# Tokenize the text messages
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Message'])
sequences = tokenizer.texts_to_sequences(df['Message'])
assert len(tokenizer.word_index) > 0, "Word index is empty. Ensure that your text data is correctly loaded and preprocessed."


# Tokenize the user
user_tokenizer = Tokenizer()
user_tokenizer.fit_on_texts(df['User'])
user_sequences = user_tokenizer.texts_to_sequences(df['User'])

# Generate skip-grams
word_target, word_context, all_labels = [], [], []
for seq in sequences:
    if seq:  # Check if the sequence is not empty
        couples, labels_output = skipgrams(seq, vocabulary_size=len(tokenizer.word_index) + 1, window_size=5, negative_samples=1.0)
        if couples:  # Check if couples were generated
            word_targets, word_contexts = zip(*couples)
            word_target.extend(word_targets)
            word_context.extend(word_contexts)
            all_labels.extend(labels_output)

word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")
all_labels = np.array(all_labels, dtype="int32")

# Flatten the user sequences to match the number of skip-grams
flat_user_sequences = [user for seq in user_sequences for user in seq]
user_sequences_skipgrams = np.array([flat_user_sequences[i] for i in range(len(word_target))], dtype="int32")

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
merged = Concatenate(axis=-1)([dot_product, Flatten()(user_embedding)])
output = Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[input_target, input_context, input_user], outputs=output)

# Compute decay factors for the custom loss function
decay_factors = (df['Time'] - df['Time'].min()) / (df['Time'].max() - df['Time'].min())
decay_factors = decay_factors.values

# Custom loss function with time decay
def time_decay_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) * decay_factors

# Use a lambda function to pass additional arguments to the loss function
model.compile(loss=lambda y_true, y_pred: time_decay_loss(y_true, y_pred), optimizer='adam')

# Model training
model.fit([word_target, word_context, user_sequences_skipgrams], all_labels, epochs=10, batch_size=256)
