import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
from tabulate import tabulate

# Download required resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Load your review data into Python
sentences_array = sentences = [   # All the sentences as a string array ----- INPUT -----
    "The weather is beautiful today.",
    "I love spending time with my family.",
    "The movie was fantastic!",
    "He is a kind and caring person.",
    "She always makes me smile.",
    "I had a terrible day at work.",
    "The food was disappointing.",
    "I'm feeling sad and tired.",
    "He never listens to me.",
    "I'm frustrated with the situation.",
]


# Clean the text by removing unnecessary characters, numbers, or special symbols
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = text.lower().strip()  # Convert to lowercase and remove leading/trailing whitespaces
    return text

# Tokenize the reviews into sentences using NLTK's sent_tokenize function
def tokenize_sentences(review):
    sentences = sent_tokenize(review)
    return sentences

# Remove stop words from sentences
def remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))
    sentence = ' '.join([word for word in sentence.split() if word.lower() not in stop_words])
    return sentence

# Function to analyze the sentiment of a sentence
def analyze_sentiment(sentence):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(sentence)
    sentiment = 'positive' if sentiment_scores['compound'] >= 0 else 'negative'
    positivity = sentiment_scores['pos']
    negativity = sentiment_scores['neg']
    return sentiment, positivity, negativity

# Apply data preprocessing steps to the sentences_array
preprocessed_sentences = []
for review in sentences_array:
    review = clean_text(review)
    sentences = tokenize_sentences(review)
    sentences = [remove_stopwords(sentence) for sentence in sentences]
    preprocessed_sentences.append(sentences)

# Extract sentiments, positivity percentages, and negativity percentages as arrays
sentiments = []
positivity_percentages = []
negativity_percentages = []
for review_sentences in preprocessed_sentences:
    if len(review_sentences) == 0:
        sentiments.append('positive')
        positivity_percentages.append(0)
        negativity_percentages.append(0)
    else:
        positivity_sum = 0
        negativity_sum = 0
        for sentence in review_sentences:
            sentiment, positivity, negativity = analyze_sentiment(sentence)
            positivity_sum += positivity
            negativity_sum += negativity

        total_sentences = len(review_sentences)
        positivity_percentage = (positivity_sum / total_sentences) * 100
        negativity_percentage = (negativity_sum / total_sentences) * 100

        sentiments.append(sentiment)
        positivity_percentages.append(positivity_percentage)
        negativity_percentages.append(negativity_percentage)

# Create a DataFrame with the preprocessed data
data = pd.DataFrame({'Review': sentences_array, 'Sentiment': sentiments, 'Positivity Percentage': positivity_percentages, 'Negativity Percentage': negativity_percentages})

# Display the DataFrame

print(tabulate(data))