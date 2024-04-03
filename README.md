# Live Chat Sentiment Analysis - LiSA

## Overview

This repository showcases an innovative Natural Language Processing (NLP) pipeline. At the heart is a custom-trained recurrent neural network (RNN) designed for real-time sentiment analysis on livestream chats. Nicknamed LiSA, this model dramatically outperforms Google's renowned BERT (Bidirectional Encoder Representations from Transformers) model. 

LiSA is lightweight and designed to be trained on both sequential & categorical data. In roughly 30 seconds of training, this model achieves an impressive accuracy of 90%, drastically outperforming Google's BERT accuracy of 41%.

### LiSA performace report
              precision    recall  f1-score   support

     Class 0       0.91      0.84      0.87       404
     Class 1       0.86      0.90      0.88       748
     Class 2       0.92      0.92      0.92       770

     accuracy                            0.89      1922
     macro avg       0.90      0.89      0.89      1922
     weighted avg    0.90      0.89      0.89      1922
### BERT performace report
              precision    recall  f1-score   support

     Class 0       0.29      0.51      0.37       404
     Class 1       0.47      0.03      0.06       748
     Class 2       0.37      0.19      0.25       770

     micro avg       0.33      0.20      0.25      1922
     macro avg       0.38      0.24      0.23      1922
     weighted avg    0.39      0.20      0.20      1922

## Architecture of LiSA

LiSA (Live Chat Sentiment Analysis) leverages a custom-trained Recurrent Neural Network (RNN) architecture optimized for understanding and analyzing the dynamic flow of conversations in live streaming chats. This section outlines the core components and design philosophy behind LiSA's architecture.

### Core Components:

- **Sequential Input Processing**: At the heart of LiSA lies its ability to process sequential text data. Using advanced RNN layers, including LSTM (Long Short-Term Memory) units, LiSA efficiently handles varying lengths of chat messages, preserving the temporal context essential for accurate sentiment analysis.

- **Categorical Data Integration**: Beyond text, LiSA incorporates categorical data such as user roles (moderator status, subscriber status) and contextual information (game being played, channel) into its analysis. This multi-dimensional input strategy is facilitated through embedding layers that transform categorical variables into dense vectors, seamlessly integrating with sequential text data for a holistic analysis.

- **Dropout for Regularization**: To prevent overfitting and ensure the model generalizes well to unseen chat data, dropout layers are strategically placed throughout the network. This regularization technique randomly sets a fraction of the input units to 0 at each step during training, enhancing the model's robustness.

- **Dense Layers and Activation**: After processing through RNN and embedding layers, the combined data passes through dense layers with ReLU activation for non-linear transformation. The final output layer utilizes a softmax activation function to classify the sentiment into categories (e.g., positive, neutral, negative), providing a probabilistic understanding of each chat message's sentiment.

### Design Philosophy:

LiSA's architecture is designed with real-time performance and scalability in mind. It aims to provide streamers and content creators with immediate insights into the sentiment trends of their chat, enabling responsive and informed interaction with their audience. The model's lightweight nature and rapid training capability ensure that it remains practical for live streaming contexts, where speed and efficiency are paramount.

LiSA's performance, as evidenced by its accuracy and precision metrics, underscores the effectiveness of combining sequential and categorical data processing in understanding complex, real-time conversational data. By outperforming traditional models like BERT in the specific context of Twitch chat sentiment analysis, LiSA demonstrates the value of custom-tailored neural network solutions in specialized domains.



## Code Features

- **Real-time Data Scraping**: Efficiently scrapes Twitch chat messages and associated metadata.
- **LiSA: Deep learning neural net designed for live chat**: Lightweight RNN built for sentiment analysis on live chats
- **BERT-based Sentiment Analysis model**: Utilizes the BERT model for sequence classification to perform sentiment analysis on chat messages.
- **Data Serialization**: Stores the analyzed data in a structured CSV format for further analysis.
- **Configurable**: Easily configurable via a JSON file to specify the Twitch channel of interest.

## Technical Stack

- Python
- Pandas
- Tensorflow
- Torch
- Transformers library for BERT model

## Code Structure

The code is organized into modular functions for ease of understanding and extensibility.

- `config.json`: Configuration file containing channel information.
- `encode_review(text)`: Function to tokenize and encode the chat messages using BERT tokenizer.
- `predict_sentiment(text)`: Function to predict the sentiment label of an encoded message.

### How To Run

1. Run scrape_chat.py to collect live chat data
2. Run ML_Models/my_nn_model.py to train the neural net on the collected data
3. Run ML_Models/eval_my_model.py to see performance metrics
  