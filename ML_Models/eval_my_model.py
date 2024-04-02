import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the labeled dataset
#df = pd.read_csv('./CSV_Data/manual_labeled_has.csv')
df = pd.read_csv('./ChatData/labeled_dataset.csv')

# Drop missing or NaN values
df.dropna(inplace=True)

# Load the saved model
model = tf.keras.models.load_model('./my_model.keras')

# Tokenize text messages
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df['message'])
X_text = tokenizer.texts_to_sequences(df['message'])
X_text = tf.keras.preprocessing.sequence.pad_sequences(X_text)

# Convert timestamp to Unix time
def convert_to_unix(timestamp):
    return pd.Timestamp(timestamp).timestamp()

df['unix_time'] = df['date'].apply(convert_to_unix)

# Normalize Unix time
scaler = StandardScaler()
X_time = scaler.fit_transform(df['unix_time'].values.reshape(-1, 1))

# Encode user behavior
user_freq = df['user'].value_counts().to_dict()
X_user = df['user'].map(user_freq).values.reshape(-1, 1)

# Make predictions using the loaded model
y_true = df['sentiment'].map({-1: 0, 0: 1, 1: 2})
y_pred_probs = model.predict([X_text, X_time, X_user])
y_pred = np.argmax(y_pred_probs, axis=1)

# Generate the classification report
labels = np.unique(np.concatenate((y_true, y_pred)))
class_names = [f'Class {i}' for i in labels]
report = classification_report(y_true, y_pred, target_names=class_names, labels=labels)
print(report)

with open('./Eval_Reports/my_model_eval_report.txt', 'a+') as f:
    f.truncate(0)
    f.write(report)