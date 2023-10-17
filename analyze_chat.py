import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Download the NLTK resources needed for stopwords and tokenization
nltk.download('punkt')
nltk.download('stopwords')

channel = 'hasanabi'

# Load the CSV file into a DataFrame
df = pd.read_csv(f'./{channel}.csv')

# Function to preprocess text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Convert to lowercase
    tokens = [word.lower() for word in tokens]
    # Remove punctuations
    tokens = [word for word in tokens if word not in string.punctuation]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Apply the preprocessing function to the 'text' column of the DataFrame
df['processed_text'] = df['text'].apply(preprocess_text)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Create a Naive Bayes classifier
nb_classifier = MultinomialNB()

# Create a pipeline that first applies the TfidfVectorizer and then the Naive Bayes classifier
model = make_pipeline(tfidf_vectorizer, nb_classifier)

# Assume that the sentiment is positive if the word 'good' is in the text,
# and negative otherwise. This is just for demonstration purposes.
df['sentiment'] = df['text'].apply(lambda x: 'positive' if 'good' in x.lower() else 'negative')

# Train the model on the processed text and the naive sentiment labels
model.fit(df['processed_text'], df['sentiment'])

# Run sentiment analysis on the processed text
df['predicted_sentiment'] = model.predict(df['processed_text'])

# Show some results
print(df[['text', 'sentiment', 'predicted_sentiment']].head())

pd.DataFrame(df).to_csv(channel + "analysis" + ".csv", index = False)