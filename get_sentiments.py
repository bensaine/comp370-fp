# Install necessary libraries if not already installed
# pip install flair pandas

from flair.models import TextClassifier
from flair.data import Sentence
import pandas as pd

# Load the Flair sentiment classifier
classifier = TextClassifier.load('sentiment')

# Function to assign sentiment with neutral label support
def assign_sentiment_flair(title, description, neutral_threshold=0.6):
    combined_text = f"{title} {description}"
    sentence = Sentence(combined_text)
    classifier.predict(sentence)
    label = sentence.labels[0].value  # 'POSITIVE' or 'NEGATIVE'
    confidence = sentence.labels[0].score  # Confidence score

    # Add neutral classification
    if confidence < neutral_threshold:
        return "Neutral"
    return label.capitalize()  # Return 'Positive' or 'Negative'

# Load the data (replace 'annotated_articles.tsv' with your actual file path)
file_path = 'annotated_articles.tsv'
data = pd.read_csv(file_path, sep='\t')

# Apply the function to the dataset
data['Flair_Sentiment'] = data.apply(lambda row: assign_sentiment_flair(row['title'], row['description']), axis=1)

# Save the updated dataset to a new file
output_file_path = 'flair_updated_annotated_articles_with_neutral.tsv'
data.to_csv(output_file_path, sep='\t', index=False)

print(f"Sentiment analysis complete. Updated file saved to {output_file_path}")
