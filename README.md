# Twitch Chat Sentiment Analysis using BERT

## Overview

This repository contains an advanced Natural Language Processing (NLP) pipeline that performs real-time sentiment analysis on Twitch livestream chats. Leveraging the state-of-the-art BERT (Bidirectional Encoder Representations from Transformers) model, this project aims to provide a granular understanding of the emotional landscape within a Twitch channel's community. This is an invaluable tool for streamers, marketers, and sociologists interested in the dynamics of online social interactions.

## Features

- **Real-time Data Scraping**: Efficiently scrapes Twitch chat messages and associated metadata.
- **BERT-based Sentiment Analysis**: Utilizes the BERT model for sequence classification to perform sentiment analysis on chat messages.
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

1. Run scrape_chat.py to collect data
2. Run pretrained_analyzer.py to generate CSV containing sentiment label
3. Run analyzer_chat.py for data visualizations
  
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
