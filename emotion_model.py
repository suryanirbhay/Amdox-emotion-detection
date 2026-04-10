import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load dataset
data = pd.read_csv("data.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Keyword-based emotion detection for better accuracy
EMOTION_KEYWORDS = {
    'happy': ['happy', 'glad', 'exciting', 'excited', 'wonderful', 'amazing', 'great', 'love', 'appreciated', 'pleased', 'delighted', 'joyful', 'cheerful'],
    'sad': ['sad', 'unhappy', 'disappointed', 'tired', 'down', 'depressed', 'blue', 'miserable', 'grief', 'sorrowful', 'heartbroken', 'gloomy', 'melancholy'],
    'angry': ['angry', 'mad', 'furious', 'frustrated', 'annoyed', 'irritated', 'outraged', 'enraged', 'livid', 'bitter', 'resentful'],
    'stressed': ['stressed', 'pressure', 'overwhelmed', 'anxious', 'worried', 'nervous', 'tense', 'exhausted', 'overworked', 'burnout', 'deadline'],
    'neutral': ['normal', 'ordinary', 'okay', 'fine', 'calm', 'relaxed', 'peaceful', 'quiet', 'stable']
}

data['text'] = data['text'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
X = vectorizer.fit_transform(data['text'])
y = data['emotion']

# Use Random Forest for better non-linear classification
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X, y)

def keyword_based_detection(text):
    """Keyword-based emotion detection for high confidence"""
    text_lower = text.lower()
    emotion_scores = {}
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(text_lower.count(keyword) for keyword in keywords)
        emotion_scores[emotion] = score
    
    max_score = max(emotion_scores.values())
    
    if max_score > 0:
        detected_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = min(95, 60 + (max_score * 10))  # High confidence for keyword matches
        return detected_emotion, confidence
    
    return None, 0

def predict_emotion(text):
    """Hybrid emotion detection: keyword-based + ML model"""
    # First try keyword-based detection
    keyword_emotion, keyword_confidence = keyword_based_detection(text)
    
    if keyword_emotion and keyword_confidence > 70:
        return keyword_emotion, keyword_confidence
    
    # Fallback to ML model
    cleaned_text = clean_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    
    confidence = max(probabilities) * 100
    
    # If keyword-based found something with medium confidence, use it
    if keyword_emotion and keyword_confidence > 50:
        return keyword_emotion, keyword_confidence
    
    return prediction, round(confidence, 2)
