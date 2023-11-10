import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the labeled dataset
labeled_chat_data = pd.read_csv('./ChatData/labeled_dataset.csv')  # Update the path accordingly

# Tokenize the input (text data)
tokenizer = BertTokenizer.from_pretrained('../bert-base-multilingual-uncased-sentiment')

def tokenize_function(examples):
    return tokenizer(examples, padding='max_length', truncation=True, max_length=128)

# Apply the tokenizer to our dataset
X = labeled_chat_data['message']
y = labeled_chat_data['sentiment']
encodings = tokenize_function(X.tolist())

# Define a custom dataset class
class ChatDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = ChatDataset(encodings, y.tolist())

# Load your pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('../bert-base-multilingual-uncased-sentiment')  # Update the path accordingly

# Define evaluation function
def evaluate(model, dataset):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=16):
            inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(model.device)
            outputs = model(**inputs)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, axis=1).tolist())
            true_labels.extend(labels.tolist())

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    return accuracy, precision, recall, f1

# Evaluate the model
accuracy, precision, recall, f1 = evaluate(model, dataset)
print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")
