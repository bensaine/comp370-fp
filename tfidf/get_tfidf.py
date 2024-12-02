import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
from collections import defaultdict

# Ensure NLTK stopwords are downloaded
# nltk.download('stopwords')

from nltk.corpus import stopwords

# Load the data
annotated_data_path = "../annotated_articles.tsv"
data = pd.read_csv(
    sep='\t',
    filepath_or_buffer=os.path.join(os.path.dirname(__file__), annotated_data_path)
)

print(data.head())

''' Sentiments data '''
sentiments = {
    s: data[data["Sentiment"]==s] for s in ["Positive", "Negative", "Neutral"]
}

for s in sentiments:
    print(f'==================> {s} results <==================', 
          sentiments[s], '\n\n', 
          f'\t\ttotal {s} -> {len(sentiments[s])} \n\n',
          '=========================*****======================\n\n'
          )


''' tfidf data '''   
# Combine 'title' and 'description' into a single text field
data['text'] = data['title'].fillna('') + ' ' + data['description'].fillna('')

# Define additional stop words
additional_stopwords = {'donald', 'trump'}

# Preprocess the text data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation except hyphens
    punctuation_without_hyphen = string.punctuation.replace('-', '')
    text = text.translate(str.maketrans('', '', punctuation_without_hyphen))
    # Tokenize
    tokens = text.split()
    # Remove stopwords and additional words
    tokens = [word for word in tokens if word not in stopwords.words('english') and word not in additional_stopwords]
    return ' '.join(tokens)

# Apply preprocessing
data['cleaned_text'] = data['text'].apply(preprocess_text)

# Custom token pattern to include hyphenated words
token_pattern = r'(?u)\b\w[\w-]+\b'

# Compute TF-IDF scores across the entire corpus
tfidf_vectorizer = TfidfVectorizer(token_pattern=token_pattern)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['cleaned_text'])
feature_names = tfidf_vectorizer.get_feature_names_out()

# Map each topic to the indices of the documents belonging to it
topic_to_indices = defaultdict(list)
for idx, topic in enumerate(data['topic']):
    topic = str(topic)
    topic_to_indices[topic].append(idx)

# For each topic, compute the average TF-IDF score for each word
for topic, indices in topic_to_indices.items():
    # Get the rows corresponding to the current topic
    topic_tfidf = tfidf_matrix[indices]
    # Compute the mean TF-IDF score for each word in the topic
    mean_tfidf = topic_tfidf.mean(axis=0).A1
    # Get the top words excluding 'donald' and 'trump'
    top_indices = mean_tfidf.argsort()[::-1]
    top_words = [feature_names[i] for i in top_indices if feature_names[i] not in additional_stopwords][:10]
    print(f"Top words for topic '{topic}':")
    print(top_words)
    print("\n")
