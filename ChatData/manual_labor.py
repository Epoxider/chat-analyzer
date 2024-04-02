import pandas as pd

def label_sentiment(csv_file_path, output_file_path, starting_row=0):
    # Load the CSV file
    data = pd.read_csv(csv_file_path)

    # Add a new column for sentiment if not present
    if 'sentiment' not in data.columns:
        data['sentiment'] = None

    # Initialize the row index
    index = starting_row

    while True:
        # Display the current message
        msg = data.iloc[index]['text']
        print(f"Row {index}: {msg}")

        # Get user input for sentiment
        sentiment = input("Enter sentiment (1, 2, 3) or 'q' to quit, 'b' to go back, 'n' to skip: ").strip()

        # Input validation
        if sentiment in ['1', '2', '3']:
            # Map input to sentiment score and update the data frame
            sentiment_map = {'1': -1, '2': 0, '3': 1}
            data.at[index, 'sentiment'] = sentiment_map[sentiment]
            index += 1  # Move to next row
        elif sentiment == 'b' and index > 0:
            index -= 1  # Move back one row
        elif sentiment == 'n':
            index += 1  # Skip to next row without changing sentiment
        elif sentiment == 'q':
            # Save and exit
            data.to_csv(output_file_path, index=False)
            break
        else:
            print("Invalid input. Please enter 1, 2, 3, 'b', 'n', or 'q'.")

        # Check if it's the end of the data
        if index >= len(data):
            print("End of data reached.")
            break

    # Save the updated data to a new CSV file
    data.to_csv(output_file_path, index=False)
    print(f"Data saved to {output_file_path}")

# Example usage
#csv_file_path = './ChatData/data_set/0ae1dab77433009d23876383fd394ed22a9f6a9c_2.csv'  # Replace with your CSV file path
csv_file_path = './manual_labeled_hasan.csv'  # Replace with your CSV file path
output_file_path = 'manual_labeled_hasan.csv'  # Replace with your desired output file path
label_sentiment(csv_file_path, output_file_path, starting_row=820)
