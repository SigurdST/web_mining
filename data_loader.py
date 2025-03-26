import pandas as pd
import json
import re
import nltk
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

def load_and_preprocess_with_embeddings():
    # Load Event JSON
    events = []
    with open('data/database/Nodes/Event.json', 'r') as f:
        for line in f:
            obj = json.loads(line.strip())
            events.append(obj['n']['properties'])
    df_events = pd.DataFrame(events)

    # Load Tweets JSON
    tweets = []
    with open('data/database/Nodes/Tweet.json', 'r') as f:
        for line in f:
            obj = json.loads(line.strip())
            tweet = obj['n']['properties']
            tweets.append(tweet)
    df_tweets = pd.DataFrame(tweets)

    # Merge tweets and events
    df = df_tweets.merge(df_events, left_on='topic', right_on='trecisid', how='left')
    df['timestamp'] = pd.to_datetime(df['created_at'])
    df['text'] = df['text'].astype(str)

    # Preprocess
    df['clean_text'] = df['text'].apply(lambda x: re.sub(r'http\S+|@\S+|#\S+|[^A-Za-z\s]', '', x.lower()))
    df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    df['tokens'] = df['clean_text'].apply(nltk.word_tokenize)

    # Load Sentence-BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate tweet embeddings
    df['tweet_embedding'] = model.encode(df['clean_text'].tolist()).tolist()

    return df