import matplotlib.pyplot as plt
import pandas as pd

# Create dummy data
data = {
    'Word': [f'Word_{i}' for i in range(1, 11)],
    'TF-IDF': [0.1 * i for i in range(1, 11)]
}
df = pd.DataFrame(data)

# Plot the data
plt.figure(figsize=(10, 6))
plt.barh(df['Word'], df['TF-IDF'], color='skyblue')
plt.xlabel('TF-IDF Score')
plt.ylabel('Words')
plt.title('Top 10 Words with Highest TF-IDF Scores (Sample Topic)')
plt.gca().invert_yaxis()
plt.tight_layout()

# Save the plot
plt.savefig('top_tfidf_words.png')
plt.show()
