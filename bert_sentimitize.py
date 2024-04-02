import json
import numpy as np
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import classification_report
import torch

with open('config.json', 'r') as f:
    config = json.load(f)
    channel = config['channel']

# Load the CSV file into a DataFrame
#df = pd.read_csv(f'./{channel}.csv')
df = pd.read_csv(f'./ChatData/labeled_dataset.csv')


# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("../bert-base-multilingual-uncased-sentiment")
model = BertForSequenceClassification.from_pretrained("../bert-base-multilingual-uncased-sentiment")

# Function to encode text
def encode_review(text):
    return tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

# Function to predict sentiment
def predict_sentiment(text):
    # Encode text
    encoded_text = encode_review(text)
    # Get model output
    with torch.no_grad():
        output = model(**encoded_text)
    # Get predicted label index
    label_idx = torch.argmax(output[0]).item()
    return label_idx

# Mapping from label index to human-readable label
label_map = {
    0: 'very negative',
    1: 'negative',
    2: 'neutral',
    3: 'positive',
    4: 'very positive'
}

# Apply the function to your text column
#df['predicted_label_idx'] = df['text'].apply(predict_sentiment)
#df['predicted_label'] = df['predicted_label_idx'].apply(lambda x: label_map[x])

df['predicted_label_idx'] = df['Message'].apply(predict_sentiment)
df['predicted_label'] = df['predicted_label_idx'].apply(lambda x: label_map[x])

# Save the DataFrame with the new columns

#pd.DataFrame(df[['username', 'text', 'predicted_label_idx', 'predicted_label', 'timestamp']]).to_csv(channel + "_analysis" + ".csv", index = False)
pd.DataFrame(df[['User', 'Message', 'predicted_label_idx', 'predicted_label', 'Time']]).to_csv('sample_data' + "_analysis" + ".csv", index = False)
