# Step 1: Import Libraries
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Step 2: Load Dataset with explicit encoding
file_path = "C:\\Users\\Admin\\Downloads\\labeled_dataset.csv"  # Modify the file path here
df = pd.read_csv(file_path, encoding='latin1')  # Try different encodings if needed

# Print column names to identify the correct column for text
print(df.columns)

# Step 3: Data Exploration
print(df.head())

# Step 4: TextBlob Sentiment Analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Assuming 'clean_text' is the correct column name for the text
df['Sentiment'] = df['clean_text'].apply(analyze_sentiment)

# Print 10 texts with their sentiment
print(df[['clean_text', 'Sentiment']].head(10))

# Step 5: Data Analysis
sentiment_counts = df['Sentiment'].value_counts()

# Plotting the sentiment distribution as a pie chart
plt.figure(figsize=(8, 6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['green', 'blue', 'red'], startangle=140)
plt.title('Sentiment Analysis')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Display the plot
plt.show()

print(sentiment_counts)
