import pandas as pd

df = pd.read_csv('./labeled_dataset.csv')

print(df['sentiment'].value_counts())