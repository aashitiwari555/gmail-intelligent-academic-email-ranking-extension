"""
Email Dataset Exploratory Data Analysis (EDA)
Author: Aashi Tiwari

This script performs:
1. Dataset inspection
2. Text preprocessing
3. Exploratory data analysis
4. Visualization
"""

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Download required resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load Dataset
df = pd.read_csv("emails.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nDataset Description:")
print(df.describe(include='all'))

print("\nMissing Values:")
print(df.isnull().sum())

# Drop unnecessary column
df = df.drop(columns=['timestamp'])

# Remove duplicates
df.drop_duplicates(inplace=True)

print("\nFinal Dataset Shape:", df.shape)


# Text Preprocessing

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess_text(text):

    text = str(text).lower()

    # Remove non alphabet characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = nltk.word_tokenize(text)

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # POS tagging
    tagged_words = pos_tag(words)

    # Lemmatization
    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged_words
    ]

    return " ".join(lemmatized)


# Apply preprocessing
df['clean_text'] = df['email_text'].apply(preprocess_text)

# Feature engineering
df['email_length'] = df['clean_text'].apply(len)

print("\nProcessed Data Sample:")
print(df.head())


# Exploratory Data Analysis

print("\nIntent Distribution:")
print(df['intent'].value_counts())

print("\nPriority Distribution:")
print(df['priority'].value_counts())

print("\nEmail Length Statistics:")
print(df['email_length'].describe())

print("\nAverage Email Length by Priority:")
print(df.groupby('priority')['email_length'].mean())


# Visualization

# Intent distribution
plt.figure(figsize=(8,5))
sns.countplot(x='intent', data=df)
plt.title("Intent Distribution")
plt.xticks(rotation=45)
plt.show()

# Priority distribution
plt.figure(figsize=(6,4))
sns.countplot(x='priority', data=df)
plt.title("Priority Distribution")
plt.show()

# Email length histogram
plt.figure(figsize=(8,5))
sns.histplot(df['email_length'], bins=30)
plt.title("Email Length Distribution")
plt.show()

# Email length by priority
plt.figure(figsize=(6,4))
sns.boxplot(x='priority', y='email_length', data=df)
plt.title("Email Length by Priority")
plt.show()

# Wordcloud
text = " ".join(df['clean_text'])

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Most Frequent Words")
plt.show()