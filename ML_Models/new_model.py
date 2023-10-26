# Importing the necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample code to generate some random data
df = pd.read_csv('../ChatData/data_set/d132e42bde1b8d87d58fcfb8838429ce697c3ef4_6.csv')


# Drop missing or NaN values
df.dropna(inplace=True)

# Tokenizing text messages
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df['Message'])
X_text = tokenizer.texts_to_sequences(df['Message'])
X_text = tf.keras.preprocessing.sequence.pad_sequences(X_text)

# Normalizing Time feature
scaler = StandardScaler()
X_time = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Encoding user behavior as frequency of each user's messages
user_freq = df['User'].value_counts().to_dict()
X_user = df['User'].map(user_freq).values.reshape(-1, 1)

# Define input layers
text_input = tf.keras.Input(shape=(X_text.shape[1],), dtype='int32', name='text')
time_input = tf.keras.Input(shape=(1,), name='time')
user_input = tf.keras.Input(shape=(1,), name='user')

# Embedding layer for text
embedding_layer = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64)(text_input)

# Temporal Attention Mechanism
temporal_layer = tf.keras.layers.Dense(64, activation='relu')(time_input)
sequence_length = X_text.shape[1]
temporal_repeat = tf.keras.layers.RepeatVector(sequence_length)(temporal_layer)

# User behavior layer
user_layer = tf.keras.layers.Dense(64, activation='relu')(user_input)
user_repeat = tf.keras.layers.RepeatVector(sequence_length)(user_layer)

# Concatenating all layers
concat_layer = tf.keras.layers.Concatenate(axis=-1)([embedding_layer, temporal_repeat, user_repeat])

# Adding a flatten layer to convert 3D output to 2D
flatten_layer = tf.keras.layers.Flatten()(concat_layer)

# Final dense layer to match the output shape
output_layer = tf.keras.layers.Dense(X_text.shape[1], activation='linear')(flatten_layer)

# Compile the model
model = tf.keras.Model(inputs=[text_input, time_input, user_input], outputs=output_layer)
model.compile(optimizer='adam', loss='mse')

# Display model architecture
model.summary()

# Split data into training and validation sets
X_train_text, X_val_text, X_train_time, X_val_time, X_train_user, X_val_user = train_test_split(
    X_text, X_time, X_user, test_size=0.2, random_state=42
)

# Model Training
model.fit([X_train_text, X_train_time, X_train_user], X_train_text, epochs=5, validation_data=([X_val_text, X_val_time, X_val_user], X_val_text))

# Performance Metrics
val_loss = model.evaluate([X_val_text, X_val_time, X_val_user], X_val_text)
print(f'Validation Loss: {val_loss}')
