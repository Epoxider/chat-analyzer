# Twitch Chat Sentiment Analysis using BERT

## Overview

This repository showcases an innovative Natural Language Processing (NLP) pipeline, where at the heart is a custom-trained neural network designed for real-time sentiment analysis on livestream chats. The neural net dramatically outperforms Google's renowned BERT (Bidirectional Encoder Representations from Transformers) model, achieving an accuracy of 90% compared to BERT's 41%. This project is made to offer a nuanced understanding of a Twitch channel's community, setting a new standard for sentiment analysis. Tailored for streamers, content creators, and those interested in the intricacies of online social interactions, this tool marks a significant leap forward in assessing and engaging with live audience sentiments.

## Features

- **Real-time Data Scraping**: Efficiently scrapes Twitch chat messages and associated metadata.
- **Deep learning neural net designed for live chat**
- **BERT-based Sentiment Analysis model**: Utilizes the BERT model for sequence classification to perform sentiment analysis on chat messages.
- **Data Serialization**: Stores the analyzed data in a structured CSV format for further analysis.
- **Configurable**: Easily configurable via a JSON file to specify the Twitch channel of interest.

## Technical Stack

- Python
- Pandas for data manipulation
- Transformers library for BERT model
- Torch for tensor operations

## Code Structure

The code is organized into modular functions for ease of understanding and extensibility.

- `config.json`: Configuration file containing channel information.
- `encode_review(text)`: Function to tokenize and encode the chat messages using BERT tokenizer.
- `predict_sentiment(text)`: Function to predict the sentiment label of an encoded message.

### How To Run

1. Run scrape_chat.py to collect live chat data
2. Run ML_Models/my_nn_model.py to train the neural net on the collected data
3. Run ML_Models/eval_my_model.py to see performance metrics
  
### Sentiment Prediction Logic

The function `encode_review` tokenizes the input text and pads it to a maximum length of 512 tokens. It returns a PyTorch tensor that is fed into the BERT model.

```python
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
```
