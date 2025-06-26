# Task-04: Twitter Entity Sentiment Analysis

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 1. Load dataset (replace with your own path if downloaded)
df = pd.read_csv('twitter_training.csv', names=["entity", "sentiment", "content"])

# 2. Basic cleanup
df.dropna(inplace=True)
df = df[df['sentiment'].isin(['Positive', 'Negative', 'Neutral'])]

# 3. Visualize sentiment distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=df, palette='coolwarm')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Tweet Count')
plt.show()

# 4. Word Cloud for each sentiment
def plot_wordcloud(sentiment):
    text = ' '.join(df[df['sentiment'] == sentiment]['content']).lower()
    tokens = [word for word in text.split() if word.isalpha() and word not in stop_words]
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(tokens))
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"{sentiment} Tweets Word Cloud")
    plt.show()

for s in ['Positive', 'Negative', 'Neutral']:
    plot_wordcloud(s)

# 5. Entity-based sentiment counts
entity_counts = df.groupby(['entity', 'sentiment']).size().unstack().fillna(0)
top_entities = entity_counts.sum(axis=1).sort_values(ascending=False).head(5)

entity_counts.loc[top_entities.index].plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
plt.title("Sentiment by Top Entities")
plt.xlabel("Entity")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=45)
plt.legend(title="Sentiment")
plt.tight_layout()
plt.show()
