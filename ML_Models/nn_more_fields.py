from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import StringLookup, Embedding, Input, Flatten, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report


# Load the labeled dataset
#df = pd.read_csv('./CSV_Data/manual_labeled_has.csv')
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
    #return datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f').timestamp()
    return datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()

# Apply the conversion to the 'date' column
df['unix_time'] = df['date'].apply(convert_to_unix)
scaler = StandardScaler()
X_time = scaler.fit_transform(df['unix_time'].values.reshape(-1, 1))

# Convert 'subscriber' and 'mod' to integers (1 for True, 0 for False)
df['subscriber'] = df['subscriber'].astype(int)
df['mod'] = df['mod'].astype(int)

# Prepare StringLookup layer for 'game' and 'channel'
game_lookup = StringLookup(output_mode='int')
channel_lookup = StringLookup(output_mode='int')
game_lookup.adapt(df['game'])
channel_lookup.adapt(df['channel'])
# Transform 'game' and 'channel' into integer indices
X_game = game_lookup(df['game']).numpy()
X_channel = channel_lookup(df['channel']).numpy()
X_subscriber = df['subscriber'].values.reshape(-1, 1)
X_mod = df['mod'].values.reshape(-1, 1)

# Encoding user behavior as frequency of each user's messages
user_freq = df['user'].value_counts().to_dict()
X_user = df['user'].map(user_freq).values.reshape(-1, 1)

# Sentiment labels
df['sentiment'] = df['sentiment'].map({-1: 0, 0: 1, 1: 2})
y = to_categorical(df['sentiment'])  # One-hot encoding of sentiment labels

# Input layers
text_input = Input(shape=(X_text.shape[1],), dtype='int32', name='text')
time_input = Input(shape=(1,), name='date')
user_input = Input(shape=(1,), name='user')
game_input = Input(shape=(1,), name='game', dtype='int32')
channel_input = Input(shape=(1,), name='channel', dtype='int32')
subscriber_input = Input(shape=(1,), name='subscriber')
mod_input = Input(shape=(1,), name='mod')

sequence_length = X_text.shape[1]

# Embedding layer for text (with dropout for regularization)
text_embedding = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64)(text_input)
dropout_emb = tf.keras.layers.Dropout(0.5)(text_embedding)  # Dropout added

# Temporal Attention Mechanism
temporal_layer = tf.keras.layers.Dense(64, activation='relu')(time_input)
temporal_repeat = tf.keras.layers.RepeatVector(sequence_length)(temporal_layer)

user_layer = tf.keras.layers.Dense(64, activation='relu')(user_input)
user_repeat = tf.keras.layers.RepeatVector(sequence_length)(user_layer)

# Embedding layers for 'game' and 'channel'
game_embedding = Embedding(input_dim=game_lookup.vocabulary_size(), output_dim=64)(game_input)
game_dropout_emb = tf.keras.layers.Dropout(0.5)(game_embedding)  # Dropout added

channel_embedding = Embedding(input_dim=channel_lookup.vocabulary_size(), output_dim=64)(channel_input)
channel_dropout_emb = tf.keras.layers.Dropout(0.5)(channel_embedding)  # Dropout added

# Flatten the embedding outputs for 'game' and 'channel'
game_flat = Flatten(name='game_flat')(game_embedding)
game_repeat = tf.keras.layers.RepeatVector(sequence_length)(game_flat)
channel_flat = Flatten(name='channel_flat')(channel_embedding)
channel_repeat = tf.keras.layers.RepeatVector(sequence_length)(channel_flat)

# Sub layer - is user a sub or not
subscriber_layer = tf.keras.layers.Dense(64, activation='relu')(subscriber_input)
subscriber_repeat = tf.keras.layers.RepeatVector(sequence_length)(subscriber_layer)

# Mod layer - is user a mod or not
mod_layer = tf.keras.layers.Dense(64, activation='relu')(mod_input)
mod_repeat = tf.keras.layers.RepeatVector(sequence_length)(mod_layer)

# Concatenating all layers
concat_layer = Concatenate(axis=-1, name='concatenate_layer')([
    text_embedding, 
    temporal_repeat, 
    user_repeat, 
    game_repeat,
    channel_repeat,
    subscriber_repeat, 
    mod_repeat
])

# Adding a flatten layer to convert 3D output to 2D
flatten_layer = tf.keras.layers.Flatten()(concat_layer)

# Final dense layer for classification
output_layer = tf.keras.layers.Dense(y.shape[1], activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))(flatten_layer)

# Compile the model
model = tf.keras.Model(inputs=[text_input, time_input, user_input, game_input, channel_input, subscriber_input, mod_input], outputs=output_layer)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping Callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Display model architecture
model.summary()

# Split data into training and validation sets for all inputs
X_train_text, X_val_text, X_train_time, X_val_time, X_train_user, X_val_user, X_train_game, X_val_game, X_train_channel, X_val_channel, X_train_subscriber, X_val_subscriber, X_train_mod, X_val_mod, y_train, y_val = train_test_split(
    X_text, X_time, X_user, X_game, X_channel, X_subscriber, X_mod, y, test_size=0.2, random_state=42
)

# Model Training with all inputs
model.fit(
    [X_train_text, X_train_time, X_train_user, X_train_game, X_train_channel, X_train_subscriber, X_train_mod], 
    y_train, 
    epochs=50, 
    validation_data=(
        [X_val_text, X_val_time, X_val_user, X_val_game, X_val_channel, X_val_subscriber, X_val_mod], 
        y_val
    )
)
model.save('./new_model.keras')

############ END MODEL TRAINING ###############

# Make predictions on the validation set
y_val_pred_probs = model.predict([X_val_text, X_val_time, X_val_user, X_val_game, X_val_channel, X_val_subscriber, X_val_mod])
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
val_loss, val_accuracy = model.evaluate([X_val_text, X_val_time, X_val_user, X_val_game, X_val_channel, X_val_subscriber, X_val_mod], y_val)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')

with open('./Eval_Reports/new_labeled_data_eval.txt', 'a+') as f:
    f.truncate(0)
    f.write(report)