
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Nettoyage du texte
def clean_text(text):
    text = re.sub(r'http\S+', '', str(text))  # liens
    text = re.sub(r'@\w+', '', text)         # mentions
    text = re.sub(r'#\w+', '', text)         # hashtags
    text = re.sub(r'[^\w\s]', '', text)     # ponctuation
    return text.lower().strip()

# TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
def compute_tfidf(df):
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return tfidf_df

# Feature engineering complet
def engineer_features(df):
    df_feat = pd.DataFrame()

    # Features de base
    df_feat['question_mark'] = df['text'].str.endswith('?').astype(int)
    df_feat['exclam_count'] = df['text'].str.count('!')
    df_feat['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]) if x else 0)
    df_feat['num_hashtags'] = df['text'].str.count('#')
    df_feat['num_mentions'] = df['text'].str.count('@')
    df_feat['num_links'] = df['text'].str.count('http')

    # Temporal features
    df_feat['created_at_hour'] = pd.to_datetime(df['created_at'], errors='coerce').dt.hour.fillna(0).astype(int)

    # Social user features
    df_feat['followers_count'] = df['followers_count'].fillna(0).astype(float)
    df_feat['friends_count'] = df['friends_count'].fillna(0).astype(float)

    # TF-IDF text representation
    tfidf_df = compute_tfidf(df)

    # Event/category/user one-hot
    cat_df = pd.get_dummies(df[['event_type', 'category', 'author']].fillna('missing'), drop_first=True)

    # Fusion
    full_df = pd.concat([df_feat.reset_index(drop=True), tfidf_df.reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)
    full_df['annotation_postPriority'] = df['annotation_postPriority'].reset_index(drop=True)

    return full_df
