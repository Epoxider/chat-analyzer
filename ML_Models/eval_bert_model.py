import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import classification_report
import torch

# Load the CSV file
print("Loading CSV file...")
df = pd.read_csv('./ChatData/labeled_dataset.csv')
df.dropna(inplace=True)
print("CSV file loaded successfully.")

# Initialize tokenizer and model
print("Initializing BERT tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained("../bert-base-multilingual-uncased-sentiment")
model = BertForSequenceClassification.from_pretrained("../bert-base-multilingual-uncased-sentiment")
print("BERT tokenizer and model initialized successfully.")

# Define the text encoding function
def encode_text(text):
    return tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

# Predict sentiment function
def predict_sentiment(text):
    encoded_text = encode_text(text)
    with torch.no_grad():
        output = model(**encoded_text)
    probs = torch.nn.functional.softmax(output.logits, dim=-1)
    label_idx = torch.argmax(probs, dim=-1).item()
    return label_idx

print("Predicting sentiment...")
df['predicted_label_idx'] = df['message'].apply(predict_sentiment)
print("Sentiment prediction complete.")

# Prepare data for classification report
y_true = df['sentiment'].map({-1: 0, 0: 1, 1: 2})
y_pred = df['predicted_label_idx'].values

# Generate and print the classification report
expected_labels = [0, 1, 2]
class_names = ['Class 0', 'Class 1', 'Class 2']
report = classification_report(y_true, y_pred, labels=expected_labels, target_names=class_names)
print("Classification report:")
print(report)

# Save the report
with open('./Eval_Reports/bert_eval_report.txt', 'a+') as f:
    f.truncate(0)
    f.write(report)
