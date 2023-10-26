import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

with open('config.json', 'r') as f:
    config = json.load(f)
    channel = config['channel']
    df = pd.read_csv(f'{channel}_analysis.csv')

# Get the number of comments for each sentiment
counts = df['predicted_label'].value_counts()
usr_msgs = df.groupby('username').count().sort_values(by='text', ascending=False).head(10)
print(usr_msgs)

# Calculate the mean sentiment per user
mean_sentiment = df.groupby(['username', 'predicted_label'])['predicted_label_idx'].mean().unstack(fill_value=0)
print(mean_sentiment)

# Create a bar chart of the sentiment counts
plt.bar(counts.index, counts.values)

# Add labels and title
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment')
plt.show()
