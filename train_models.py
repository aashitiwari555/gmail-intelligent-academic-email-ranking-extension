"""
Model Training Pipeline

Author: Aashi Tiwari

This script trains two machine learning models for the
Gmail Intelligent Academic Email Ranking Extension:

1. Intent Classification Model
   - TF-IDF + Logistic Regression

2. Urgency Prediction Model
   - SentenceTransformer embeddings + XGBoost

The models are used to detect the intent of emails and
predict their urgency score for ranking.
"""


import pandas as pd
import numpy as np
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

from xgboost import XGBRegressor
from sentence_transformers import SentenceTransformer

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Dataset not included in this repository
df = pd.read_csv('emails.csv')

print("Dataset Shape:", df.shape)
display(df.head())

df = df.drop(columns=['timestamp'])

df.drop_duplicates(inplace=True)
print("Final Dataset Shape:", df.shape)

"""Train Test Split"""

from sklearn.model_selection import train_test_split

X = df['email_text']
y_intent = df['intent']
y_priority = df['priority']

y_both = pd.concat([y_intent, y_priority], axis=1)

print(y_both)

# 2. Split everything in one go.
# This ensures X_train matches BOTH sets of labels perfectly.
X_train, X_test, y_train_both, y_test_both = train_test_split(
    X, y_both, test_size=0.2, random_state=42
)

print(X_train)

print(y_train_both)

# 3. SEPARATION: Now we unglue them so we can train separate models
y_train_intent = y_train_both['intent']
y_train_priority = y_train_both['priority']

y_test_intent = y_test_both['intent']
y_test_priority = y_test_both['priority']

print(y_train_intent)

print(y_train_priority)


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

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

def preprocess_intent(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]

    tagged_words = pos_tag(words)

    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged_words
    ]

    return " ".join(lemmatized)

X_train_intent = X_train.apply(preprocess_intent)
X_test_intent = X_test.apply(preprocess_intent)

print(X_train_intent)

custom_stopwords = set(stopwords.words('english')) - {
    'before', 'after', 'tomorrow', 'today',
    'urgent', 'very', 'immediately',
    'soon', 'asap', 'early', 'late',
    'must', 'need', 'required',
    'until', 'now', 'during',
    'too', 'just', 'should', 'attention',
    'deadline', 'short', 'notice'
}

def preprocess_priority(text):
    #text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in custom_stopwords]
    return " ".join(words)

X_train_priority = X_train.apply(preprocess_priority)
X_test_priority = X_test.apply(preprocess_priority)

print(X_train_priority)

intent_encoder = LabelEncoder()

y_train_intent_encoded = intent_encoder.fit_transform(y_train_intent)
y_test_intent_encoded = intent_encoder.transform(y_test_intent)

# Map priority to numeric score for regression
priority_map = {'Low': 0, 'Medium': 1, 'High': 2}

y_train_priority_encoded = y_train_priority.map(priority_map)
y_test_priority_encoded = y_test_priority.map(priority_map)


intent_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2),)

X_train_intent_vec = intent_vectorizer.fit_transform(X_train_intent)
X_test_intent_vec = intent_vectorizer.transform(X_test_intent)

intent_model = LogisticRegression(max_iter=1000)

intent_model.fit(X_train_intent_vec, y_train_intent_encoded)

intent_preds = intent_model.predict(X_test_intent_vec)

print("Intent Accuracy:", accuracy_score(y_test_intent_encoded, intent_preds))
print(classification_report(y_test_intent_encoded, intent_preds))

test_emails = [
    "I cannot submit my assignment because the portal is not working.",
    "Could you please share the timetable for next semester?",
    "I would like information about hostel accommodation.",
    "There is an issue with my fee payment receipt.",
    "How can I access digital resources from the library?"
]

clean_test_emails = [preprocess_intent(email) for email in test_emails]
X_test_vec = intent_vectorizer.transform(clean_test_emails)
intent_predictions = intent_model.predict(X_test_vec)

# Convert numeric labels back to original intent names
intent_labels = intent_encoder.inverse_transform(intent_predictions)

# Display results
for email, intent in zip(test_emails, intent_labels):
    print("Email:", email)
    print("Predicted Intent:", intent)
    print("-" * 60)


# Load lightweight transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert training emails to embeddings
X_train_priority_vec = embedding_model.encode(
    X_train_priority.tolist(),
    batch_size=32,
    show_progress_bar=True
)

# Convert test emails to embeddings
X_test_priority_vec = embedding_model.encode(
    X_test_priority.tolist(),
    batch_size=32
)

# Convert to numpy arrays
X_train_priority_vec = np.array(X_train_priority_vec)
X_test_priority_vec = np.array(X_test_priority_vec)

priority_model = XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

priority_model.fit(X_train_priority_vec, y_train_priority_encoded)

priority_preds = priority_model.predict(X_test_priority_vec)

priority_preds_class = priority_preds.round().clip(0,2)
print(classification_report(y_test_priority_encoded, priority_preds_class))

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test_priority_encoded, priority_preds)
print("Priority MSE:", mse)

priority_scores = priority_model.predict(X_test_priority_vec)

# 2. Combine them into a single score
# You can weight them differently (e.g., priority matters more than intent)

test_results = pd.DataFrame({
    'text': X_test,
    'urgency_score': priority_scores
})
# 3. Sort with most urgent on top
sorted_results = test_results.sort_values(by='urgency_score', ascending=False)

pd.set_option('display.max_colwidth', None)

# To see the top 20 "Most Urgent" rows with full text
display(sorted_results.head(20))

# To see the bottom 20 "Least Urgent" rows
display(sorted_results.tail(20))

stress_test_emails = [

    "URGENT: My admit card has not been generated and my exam is tomorrow. Please resolve immediately.",

    "Emergency hospitalization. I am unable to submit my final assignment before today's deadline.",

    "If this issue is not resolved, I will not be able to sit for the final exam tomorrow.",

    "I cannot upload my project submission due to portal error before the deadline tonight.",

    "My scholarship application deadline is today and the portal is not working.",

    "There is an error in my fee payment receipt and registration closes tomorrow.",

    "Critical issue: The student portal is not working and I cannot access my exam form.",

    "The electricity in my hostel room has not been working since morning.",

    "I am unable to access the university WiFi network.",

    "My final exam marks seem incorrect. Please verify.",

    "Could you re-evaluate my midterm answer sheet?",

    "Kindly extend the submission deadline for the group project.",

    "Please guide me on scholarship application procedures.",

    "What documents are required for internship registration?",

    "There seems to be an issue in my fee statement.",

    "Could you please share the timetable for the next semester?",

    "Kindly inform me about scholarship eligibility criteria.",

    "Where can I find previous year question papers?",

    "I would like clarification regarding the grading rubric used.",

    "How can I access library digital resources?"
]

clean_stress_test = [preprocess_priority(email) for email in stress_test_emails]

X_stress_embeddings = embedding_model.encode(
    clean_stress_test,
    batch_size=32
)

X_stress_embeddings = np.array(X_stress_embeddings)

urgency_scores = priority_model.predict(X_stress_embeddings)

results = pd.DataFrame({
    'Email Text': stress_test_emails,
    'Urgency Score': urgency_scores
}).sort_values(by='Urgency Score', ascending=False)

display(results)

print(df.columns)

