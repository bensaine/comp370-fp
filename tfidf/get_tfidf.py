from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import string
import os
# import nltk
# Ensure NLTK stopwords are downloaded
# nltk.download('stopwords')



# Load the data
annotated_data_path = "../annotated_articles.tsv"
data = pd.read_csv(
    filepath_or_buffer=os.path.join(
        os.path.dirname(__file__), annotated_data_path),
    sep='\t')
# print(data.head())

''' Sentiments data '''
sentiments = {s: data[data["Sentiment"]==s] \
              for s in ["Positive", "Negative", "Neutral"]}

''' tfidf data '''   
# Combine 'title' and 'description' into a single text field
data['text'] = data['title'].fillna('') + ' ' + data['description'].fillna('')

# Define additional stop words
additional_stopwords = {'donald', 'trump'}

# Preprocess the text data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    for first, last in [('elon', 'musk'), ('kamala', 'harris')]:
        text = text.replace(first, f'{first}-{last}')
        text = text.replace(f' {last}', '')

    # text = text.replace('joe', 'joe-biden')
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
tfidf_vectorizer = TfidfVectorizer(token_pattern=token_pattern, 
                                   ngram_range=(1,2)
                                   )
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


sns.set_theme()
''' potentially useful '''
def bar_chart_topic_vis():  
    # For each topic, create a bar chart of the top words
    for topic, indices in topic_to_indices.items():
        # Get the rows corresponding to the current topic
        topic_tfidf = tfidf_matrix[indices]
        # Compute the mean TF-IDF score for each word in the topic
        mean_tfidf = topic_tfidf.mean(axis=0).A1
        # Get the top words and their scores
        top_indices = mean_tfidf.argsort()[::-1]
        top_words = [feature_names[i] for i in top_indices if feature_names[i] not in additional_stopwords][:10]
        top_scores = [mean_tfidf[i] for i in top_indices if feature_names[i] not in additional_stopwords][:10]
        
        # Plotting
        plt.figure(figsize=(10,6))
        plt.bar(top_words, top_scores, color='skyblue')
        plt.title(f"Top Words for Topic: {topic}")
        plt.xlabel("Words")
        plt.ylabel("Average TF-IDF Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


''' potentially useful '''
def wordcloud_vis(): 
    for topic, indices in topic_to_indices.items():
        # Get the rows corresponding to the current topic
        topic_tfidf = tfidf_matrix[indices]
        # Compute the mean TF-IDF score for each word in the topic
        mean_tfidf = topic_tfidf.mean(axis=0).A1
        # Create a dictionary of word: score
        word_scores = {feature_names[i]: mean_tfidf[i] for i in range(len(feature_names)) if feature_names[i] not in additional_stopwords}
        # Generate a word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_scores)
        
        # Plotting
        plt.figure(figsize=(15, 7.5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f"Word Cloud for Topic: {topic}", fontsize=20)
        plt.axis('off')
        plt.show()


''' very useful '''
def sentiment_vis():  
    # Count the number of articles for each sentiment
    sentiment_counts = data['Sentiment'].value_counts()

    # Plotting
    plt.figure(figsize=(8,6))
    plt.pie(x=sentiment_counts.values, labels=sentiment_counts.index, )
    plt.title("Distribution of Sentiments")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Articles")
    plt.show()

    
''' meh '''
def heatmap_vis():  

    # Get top words across all topics
    top_n = 10
    topics = list(topic_to_indices.keys())
    top_words_per_topic = {}

    for topic, indices in topic_to_indices.items():
        topic_tfidf = tfidf_matrix[indices]
        mean_tfidf = topic_tfidf.mean(axis=0).A1
        top_indices = mean_tfidf.argsort()[::-1]
        top_words = [feature_names[i] for i in top_indices if feature_names[i] not in additional_stopwords][:top_n]
        top_words_per_topic[topic] = top_words

    # Create a set of unique top words
    unique_top_words = set()
    for words in top_words_per_topic.values():
        unique_top_words.update(words)
    unique_top_words = list(unique_top_words)
    # Create a DataFrame to store TF-IDF scores
    heatmap_data = pd.DataFrame(index=unique_top_words, columns=topics)

    for topic, indices in topic_to_indices.items():
        topic_tfidf = tfidf_matrix[indices]
        mean_tfidf = topic_tfidf.mean(axis=0).A1
        word_scores = {feature_names[i]: mean_tfidf[i] for i in range(len(feature_names))}
        for word in unique_top_words:
            heatmap_data.loc[word, topic] = word_scores.get(word, 0)

    # Convert data to float
    heatmap_data = heatmap_data.astype(float)

    # Plot heatmap
    plt.figure(figsize=(12,10))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='YlGnBu')
    plt.title("Heatmap of Top Words TF-IDF Scores Across Topics")
    plt.xlabel("Topics")
    plt.ylabel("Words")
    plt.show()


''' very useful '''
def sent_per_topic_vis(): 
    topics = data['topic'].unique()
    # Create a DataFrame to store sentiment counts per topic
    sentiment_per_topic = pd.DataFrame(0, index=topics, columns=['Positive', 'Negative', 'Neutral'])

    for topic in topics:
        topic_data = data[data['topic'] == topic]
        sentiment_counts = topic_data['Sentiment'].value_counts()
        for sentiment in sentiment_counts.index:
            sentiment_per_topic.loc[topic, sentiment] = sentiment_counts[sentiment]

    # Plotting
    sentiment_per_topic.plot(kind='bar', stacked=True, figsize=(12,8), colormap='viridis')
    plt.title("Sentiment Distribution per Topic")
    plt.xlabel("Topic")
    plt.ylabel("Number of Articles")
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.show()


''' potentially useful '''
def freq_entire_corpus_vis():

    # Combine all cleaned text
    all_words = ' '.join(data['cleaned_text']).split()

    # Count word frequencies
    word_counts = Counter(all_words)
    most_common_words = word_counts.most_common(20)

    # Separate words and counts
    words, counts = zip(*most_common_words)

    # Plotting
    plt.figure(figsize=(12,6))
    sns.barplot(x=list(words), y=list(counts), palette='cubehelix')
    plt.title("Most Frequent Words Across All Articles")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    freq_entire_corpus_vis()