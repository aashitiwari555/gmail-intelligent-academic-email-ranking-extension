import pandas as pd
import numpy as np
from nltk import pos_tag
import re
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from scipy.sparse import hstack

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

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

X_train, X_test, y_train_both, y_test_both = train_test_split(
    X, y_both, test_size=0.2, random_state=42
)

print(X_train)

print(y_train_both)

y_train_intent = y_train_both['intent']
y_train_priority = y_train_both['priority']

y_test_intent = y_test_both['intent']
y_test_priority = y_test_both['priority']

print(y_train_intent)

print(y_train_priority)

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
    text = str(text).lower()
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

priority_encoder = LabelEncoder()

y_train_priority_encoded = priority_encoder.fit_transform(y_train_priority)
y_test_priority_encoded = priority_encoder.transform(y_test_priority)

intent_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2),)

X_train_intent_vec = intent_vectorizer.fit_transform(X_train_intent)
X_test_intent_vec = intent_vectorizer.transform(X_test_intent)

priority_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2))

X_train_priority_vec = priority_vectorizer.fit_transform(X_train_priority)
X_test_priority_vec = priority_vectorizer.transform(X_test_priority)

intent_model = LogisticRegression(max_iter=1000)

intent_model.fit(X_train_intent_vec, y_train_intent_encoded)

intent_preds = intent_model.predict(X_test_intent_vec)

print("Intent Accuracy:", accuracy_score(y_test_intent_encoded, intent_preds))
print(classification_report(y_test_intent_encoded, intent_preds))

priority_model = LogisticRegression(max_iter=1000)

priority_model.fit(X_train_priority_vec, y_train_priority_encoded)

priority_preds = priority_model.predict(X_test_priority_vec)

print("Priority Accuracy:", accuracy_score(y_test_priority_encoded, priority_preds))
print(classification_report(y_test_priority_encoded, priority_preds))

intent_probs = intent_model.predict_proba(X_test_intent_vec)[:, 0]
priority_probs = priority_model.predict_proba(X_test_priority_vec)[:, 0]


test_results = pd.DataFrame({
    'text': X_test,
    'urgency_score': (0.4 * intent_probs) + (0.6 * priority_probs)
})


sorted_results = test_results.sort_values(by='urgency_score', ascending=False)

pd.set_option('display.max_colwidth', None)

display(sorted_results.head(20))

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
X_stress_intent_vec = intent_vectorizer.transform(clean_stress_test)
X_stress_priority_vec = priority_vectorizer.transform(clean_stress_test)

high_idx = list(priority_encoder.classes_).index('High')

p_intent = intent_model.predict_proba(X_stress_intent_vec)[:, high_idx]
p_priority = priority_model.predict_proba(X_stress_priority_vec)[:, high_idx]

urgency_scores = (p_intent + p_priority) / 2

results = pd.DataFrame({
    'Email Text': stress_test_emails,
    'Score': urgency_scores
}).sort_values(by='Score', ascending=False)

with pd.option_context('display.max_colwidth', None):
    display(results)

