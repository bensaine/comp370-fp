import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import os

from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
from nltk.corpus import stopwords

# Path to your data file
# Adjust the path as needed
datapath = os.path.join(os.path.dirname(__file__), '../flair_updated_annotated_articles_with_neutral.tsv')

# Read the data
df = pd.read_csv(filepath_or_buffer=datapath, sep='\t')

# Preprocess the data
df['title'] = df['title'].fillna('')
df['description'] = df['description'].fillna('')
df['text'] = df['title'] + ' ' + df['description']

# Define stopwords and add 'donald' and 'trump' to the list
stop_words = set(stopwords.words('english'))
stop_words.update(['donald', 'trump', 'former', 'vice', 
                        'president', 'election', 'vote', 'kamala', 'harris',
                        'biden', 'american', 'great', 'new', 'president-elect'])  # Exclude more generic terms

# Preprocess text function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize text
    tokens = text.split()
    # Remove stopwords and 'donald', 'trump'
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into a string
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(preprocess_text)

# Get unique topics
topics = df['topic'].unique()

# Create output directories
output_dir_unigrams = './top_unigrams'
output_dir_bigrams = './top_bigrams'
output_dir_trigrams = './top_trigrams'
output_dir_exclusive_unigrams = './top_exclusive_unigrams'

os.makedirs(output_dir_unigrams, exist_ok=True)
os.makedirs(output_dir_bigrams, exist_ok=True)
os.makedirs(output_dir_trigrams, exist_ok=True)
os.makedirs(output_dir_exclusive_unigrams, exist_ok=True)

# Loop through each topic
for topic in topics:
    # Subset data
    df_topic = df[df['topic'] == topic]
    
    # Get the text data
    texts = df_topic['clean_text'].tolist()
    
    # Remove empty strings
    texts = [text for text in texts if text.strip()]
    
    # Check if there is data to process
    if not texts:
        print(f"No data available for topic: {topic}")
        continue
    
    # Clean topic name for filenames
    safe_topic = re.sub(r'[^\w\s-]', '', str(topic)).strip().replace(' ', '_')
    
    # Compute TF-IDF scores for trigrams
    vectorizer_tri = TfidfVectorizer(ngram_range=(3,3))
    X_tri = vectorizer_tri.fit_transform(texts)
    term_scores_tri = np.asarray(X_tri.sum(axis=0)).ravel()
    terms_tri = vectorizer_tri.get_feature_names_out()
    df_trigrams = pd.DataFrame({'term': terms_tri, 'score': term_scores_tri})
    df_trigrams = df_trigrams.sort_values(by='score', ascending=False).head(10)
    
    # Compute TF-IDF scores for bigrams
    vectorizer_bi = TfidfVectorizer(ngram_range=(2,2))
    X_bi = vectorizer_bi.fit_transform(texts)
    term_scores_bi = np.asarray(X_bi.sum(axis=0)).ravel()
    terms_bi = vectorizer_bi.get_feature_names_out()
    df_bigrams = pd.DataFrame({'term': terms_bi, 'score': term_scores_bi})
    df_bigrams = df_bigrams.sort_values(by='score', ascending=False).head(10)
    
    # Collect words from top 10 bigrams and top 10 trigrams
    words_in_top_bi_tri = set()
    # From bigrams
    for term in df_bigrams['term']:
        words_in_top_bi_tri.update(term.split())
    # From trigrams
    for term in df_trigrams['term']:
        words_in_top_bi_tri.update(term.split())
    
    # Exclude these words from the unigrams
    # Compute TF-IDF scores for unigrams
    vectorizer_uni = TfidfVectorizer(ngram_range=(1,1))
    X_uni = vectorizer_uni.fit_transform(texts)
    terms_uni = vectorizer_uni.get_feature_names_out()
    term_scores_uni = np.asarray(X_uni.sum(axis=0)).ravel()
    df_unigrams = pd.DataFrame({'term': terms_uni, 'score': term_scores_uni})
    # Exclude unigrams that are in words_in_top_bi_tri
    df_unigrams_excl = df_unigrams[~df_unigrams['term'].isin(words_in_top_bi_tri)]
    df_unigrams_excl = df_unigrams_excl.sort_values(by='score', ascending=False).head(10)
    
    # Also get the plain top 10 unigrams
    df_unigrams_top10 = df_unigrams.sort_values(by='score', ascending=False).head(10)
    
    # Plot and save top 10 unigrams (plain)
    # Plot and save top 10 unigrams (plain)
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("muted", len(df_unigrams_top10))
    sns.barplot(data=df_unigrams_top10, x='score', y='term', palette=colors)
    plt.title(f"Top 10 Unigrams for Topic: {topic}")
    plt.xlabel("TF-IDF Score")
    plt.ylabel("Unigram")
    plt.tight_layout()
    filename_uni = f"{output_dir_unigrams}/top_unigrams_{safe_topic}.png"
    plt.savefig(filename_uni)
    plt.close()
    
    # Plot and save top 10 bigrams
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("muted", len(df_bigrams))
    sns.barplot(data=df_bigrams, x='score', y='term', palette=colors)
    plt.title(f"Top 10 Bigrams for Topic: {topic}")
    plt.xlabel("TF-IDF Score")
    plt.ylabel("Bigram")
    plt.tight_layout()
    filename_bi = f"{output_dir_bigrams}/top_bigrams_{safe_topic}.png"
    plt.savefig(filename_bi)
    plt.close()
    
    # Plot and save top 10 trigrams
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("muted", len(df_trigrams))
    sns.barplot(data=df_trigrams, x='score', y='term', palette=colors)
    plt.title(f"Top 10 Trigrams for Topic: {topic}")
    plt.xlabel("TF-IDF Score")
    plt.ylabel("Trigram")
    plt.tight_layout()
    filename_tri = f"{output_dir_trigrams}/top_trigrams_{safe_topic}.png"
    plt.savefig(filename_tri)
    plt.close()
    
    # Plot and save top 10 unigrams excluding words in top bigrams and trigrams
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("muted", len(df_unigrams_excl))
    sns.barplot(data=df_unigrams_excl, x='score', y='term', palette=colors)
    plt.title(f"Top 10 Unigrams Excluding Words in Top Bigrams and Trigrams for Topic: {topic}")
    plt.xlabel("TF-IDF Score")
    plt.ylabel("Unigram")
    plt.tight_layout()
    filename_uni_excl = f"{output_dir_exclusive_unigrams}/top_exclusive_unigrams_{safe_topic}.png"
    plt.savefig(filename_uni_excl)
    plt.close()
    
    # Optional: Print the results
    print(f"\nTop 10 Trigrams for Topic: {topic}")
    print(df_trigrams[['term', 'score']])
    print(f"\nTop 10 Bigrams for Topic: {topic}")
    print(df_bigrams[['term', 'score']])
    print(f"\nTop 10 Unigrams Excluding Words in Top Bigrams and Trigrams for Topic: {topic}")
    print(df_unigrams_excl[['term', 'score']])
    print(f"\nTop 10 Unigrams for Topic: {topic}")
    print(df_unigrams_top10[['term', 'score']])
