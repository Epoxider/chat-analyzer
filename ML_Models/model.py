# Importing required libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Data Preprocessing
# Load the data (assuming the CSV file is named 'chat_data.csv' and has columns 'Time', 'User', 'Message')
df = pd.read_csv('../ChatData/data_set/0a633aa33731985aa52985e79bb12f38b95b9404_0.csv')

# Tokenization for text messages
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df['Message'])
X_text = tokenizer.texts_to_sequences(df['Message'])
X_text = tf.keras.preprocessing.sequence.pad_sequences(X_text)

# Make sure all values are valid after tokenization
assert not np.any(X_text == -1), "Found invalid value -1 in tokenized sequences"

# Normalizing the time feature
scaler = StandardScaler()
X_time = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# User behavior encoding (using frequency as an example)
user_frequency = df['User'].value_counts().to_dict()
X_user = df['User'].map(user_frequency).values.reshape(-1, 1)

# Combining features and splitting data
X = np.hstack([X_text, X_time, X_user])
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Step 2: Model Construction
# Input Layer
input_layer = tf.keras.Input(shape=(X_train.shape[1],), name='combined_input')

# Embedding Layer for text (assuming a maximum of 5000 unique words and an output dimension of 64 for embeddings)
embedding_dim = 64
vocab_size = min(len(tokenizer.word_index) + 1, 5000)
embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)

# Temporal Encoding Layer (Dense layer with 64 units and ReLU activation)
temporal_layer = tf.keras.layers.Dense(64, activation='relu')(input_layer)

# User Behavior Encoding Layer (Dense layer with 64 units and ReLU activation)
user_layer = tf.keras.layers.Dense(64, activation='relu')(input_layer)

# Flatten the embedding layer or use GlobalAveragePooling1D
flattened_embedding = tf.keras.layers.GlobalAveragePooling1D()(embedding_layer)
# Concatenation Layer
concat_layer = tf.keras.layers.Concatenate()([flattened_embedding, temporal_layer, user_layer])

# Autoencoder
# Encoder Part
encoder = tf.keras.layers.Dense(128, activation='relu')(concat_layer)
encoder = tf.keras.layers.Dense(64, activation='relu')(encoder)

# Decoder Part
decoder = tf.keras.layers.Dense(128, activation='relu')(encoder)
decoder = tf.keras.layers.Dense(X_train.shape[1], activation='sigmoid')(decoder)

# Step 3: Model Compilation and Training
# Constructing the model
model = tf.keras.Model(inputs=input_layer, outputs=decoder)

# Compiling the model
model.compile(optimizer='adam', loss='mse')

# Model summary
model.summary()

# Training the model (Using 20% of the data for validation and training for 50 epochs)
model.fit(X_train, X_train, epochs=50, validation_split=0.2)

# Step 4: Validation and Performance Metrics
# Reconstruction error on test data
reconstruction_error = model.evaluate(X_test, X_test)

print("Reconstruction Error on Test Data:", reconstruction_error)
