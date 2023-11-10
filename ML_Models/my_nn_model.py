# Importing the necessary libraries
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report


# Load the labeled dataset
df = pd.read_csv('./ChatData/labeled_dataset.csv')


# Drop missing or NaN values
df.dropna(inplace=True)

# Tokenizing text messages
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df['message'])
X_text = tokenizer.texts_to_sequences(df['message'])
X_text = tf.keras.preprocessing.sequence.pad_sequences(X_text)

# Function to convert ISO 8601 timestamp to Unix timestamp
def convert_to_unix(timestamp):
    return datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()

# Apply the conversion to the 'date' column
df['unix_time'] = df['date'].apply(convert_to_unix)

# Now, normalize this unix_time column
scaler = StandardScaler()
X_time = scaler.fit_transform(df['unix_time'].values.reshape(-1, 1))

# Encoding user behavior as frequency of each user's messages
user_freq = df['user'].value_counts().to_dict()
X_user = df['user'].map(user_freq).values.reshape(-1, 1)

# Sentiment labels
y = to_categorical(df['sentiment'])  # One-hot encoding of sentiment labels

# Define input layers
text_input = tf.keras.Input(shape=(X_text.shape[1],), dtype='int32', name='text')
time_input = tf.keras.Input(shape=(1,), name='time')
user_input = tf.keras.Input(shape=(1,), name='user')

# Embedding layer for text (with dropout for regularization)
embedding_layer = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64)(text_input)
dropout_emb = tf.keras.layers.Dropout(0.5)(embedding_layer)  # Dropout added

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

# Final dense layer for classification
output_layer = tf.keras.layers.Dense(y.shape[1], activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))(flatten_layer)

# Compile the model
model = tf.keras.Model(inputs=[text_input, time_input, user_input], outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping Callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# Display model architecture
model.summary()

# Split data into training and validation sets
X_train_text, X_val_text, X_train_time, X_val_time, X_train_user, X_val_user, y_train, y_val = train_test_split(
    X_text, X_time, X_user, y, test_size=0.2, random_state=42
)

# Model Training
model.fit([X_train_text, X_train_time, X_train_user], y_train, epochs=50, validation_data=([X_val_text, X_val_time, X_val_user], y_val))

# Make predictions on the validation set
y_val_pred_probs = model.predict([X_val_text, X_val_time, X_val_user])
y_val_pred = np.argmax(y_val_pred_probs, axis=1)

# Convert one-hot encoded y_val back to label encoding for comparison
y_val_true = np.argmax(y_val, axis=1)

# Calculate additional performance metrics
# Calculate the number of unique classes
num_classes = len(np.unique(y_val_true))

# Generate a list of class names based on the number of classes
class_names = [f'Class {i}' for i in range(num_classes)]

# Generate the classification report
report = classification_report(y_val_true, y_val_pred, target_names=class_names)
print(report)

# Performance Metrics
val_loss, val_accuracy = model.evaluate([X_val_text, X_val_time, X_val_user], y_val)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')
