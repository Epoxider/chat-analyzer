import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define the maximum sequence length based on your training data
maximum_sequence_length = 97  # Update this based on your model's training

# Function to preprocess the time feature (update or remove according to your model's training)
def preprocess_time(time_values):
    scaler = StandardScaler()
    return scaler.fit_transform(time_values.reshape(-1, 1))

# Load and preprocess the unlabeled data
unlabeled_df = pd.read_csv('./ChatData/data_set/0b7f9f8e3f811e4e5ce8ac43975c7beeab1fe829_3.csv')

# Tokenize and pad the 'Message' column
# Assuming you have already trained and saved a tokenizer during model training
# Load the tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(unlabeled_df['Message'])
X_unlabeled_text = tokenizer.texts_to_sequences(unlabeled_df['Message'])
X_unlabeled_text = tf.keras.preprocessing.sequence.pad_sequences(X_unlabeled_text, maxlen=maximum_sequence_length)

# Preprocess time feature
X_unlabeled_time = preprocess_time(unlabeled_df['Time'].values)  # Only if you used time feature

# Preprocess user feature
# Assuming user IDs are categorical and were label encoded during training
user_encoder = LabelEncoder()  # Replace with the actual preprocessing used for the user data
X_unlabeled_user = user_encoder.fit_transform(unlabeled_df['User'].values.reshape(-1, 1))

# Load the trained model
loaded_model = tf.keras.models.load_model('./my_model')

# Predict sentiments
predicted_sentiments = loaded_model.predict([X_unlabeled_text, X_unlabeled_time, X_unlabeled_user])
predicted_labels = np.argmax(predicted_sentiments, axis=1)

# Add predictions to the DataFrame
unlabeled_df['PredictedSentiment'] = predicted_labels

# Export to a new CSV file
unlabeled_df.to_csv('augmented_data_long.csv', index=False)
