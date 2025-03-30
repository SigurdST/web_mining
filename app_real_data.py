import streamlit as st
import pandas as pd
import json
import re
import nltk
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
import networkx as nx
import plotly.graph_objects as go
from sklearn.manifold import TSNE

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# ----------------------------
# LOAD EVENTS + TWEETS DATA
# ----------------------------

@st.cache_data
def load_data():
    # Load Event data
    events = []
    with open('data/database/Nodes/Event.json', 'r') as f:
        for line in f:
            obj = json.loads(line.strip())
            events.append(obj['n']['properties'])
    df_events = pd.DataFrame(events)

    # Load Tweets data
    tweets = []
    with open('data/database/Nodes/Tweet.json', 'r') as f:
        for line in f:
            obj = json.loads(line.strip())
            tweet = obj['n']['properties']
            tweets.append(tweet)
    df_tweets = pd.DataFrame(tweets)
    
    # Merge tweets and events based on topic <-> trecisid relationship
    df = df_tweets.merge(df_events, left_on='topic', right_on='trecisid', how='left')
    df['timestamp'] = pd.to_datetime(df['created_at'])
    df['text'] = df['text'].astype(str)
    return df

# ----------------------------
# TEXT PREPROCESSING FUNCTION
# ----------------------------

def preprocess_tweets(df):
    # Remove URLs, mentions, hashtags, special characters, lowercase
    df['clean_text'] = df['text'].apply(lambda x: re.sub(r'http\S+|@\S+|#\S+|[^A-Za-z\s]', '', x.lower()))
    # Remove stopwords
    df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    # Tokenize text
    df['tokens'] = df['clean_text'].apply(nltk.word_tokenize)
    # Extract date for timeline analysis
    df['date'] = df['timestamp'].dt.date
    return df

# ----------------------------
# TIMELINE PLOT (tweets over time)
# ----------------------------

def plot_timeline(df_event):
    df_grouped = df_event.groupby(df_event['date']).size().reset_index(name='tweet_count')
    fig = px.line(df_grouped, x='date', y='tweet_count', title='Temporal distribution of tweets')
    st.plotly_chart(fig)

# ----------------------------
# WORD CLUSTERS BASED ON TF-IDF + KMeans
# ----------------------------

def show_clusters(df_event, n_clusters=3):
    # TF-IDF
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=1)
    X = vectorizer.fit_transform(df_event['clean_text'])
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    df_event['cluster'] = kmeans.labels_

    st.subheader("Top words per cluster")
    # Extract top words for each cluster based on frequency
    for i in range(n_clusters):
        cluster_data = df_event[df_event['cluster'] == i]
        words = ' '.join(cluster_data['clean_text'])
        top_words = pd.Series(words.split()).value_counts().head(5)
        st.write(f"**Cluster {i}:**", top_words.to_dict())

# ----------------------------
# WORD2VEC EXAMPLE (word embeddings)
# ----------------------------

def word2vec_demo(df_event):
    model = Word2Vec(sentences=df_event['tokens'], vector_size=50, window=5, min_count=1, workers=2)
    word = model.wv.index_to_key[0]
    similar = model.wv.most_similar(word)
    st.subheader("Word2Vec Example")    
    st.write(f"Words similar to **{word}**:", similar)

# ----------------------------
# TWEET EMBEDDINGS + t-SNE (sentence-level embeddings)
# ----------------------------

def tweet_embeddings_tsne(df_event):
    st.subheader("Projection of Tweet Vectors (t-SNE)")

    # Use Sentence-BERT to get tweet-level embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    tweet_vecs = model.encode(df_event['clean_text'].tolist())

    # Dimensionality reduction with t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    reduced = tsne.fit_transform(tweet_vecs)
    
    df_event['x'] = reduced[:, 0]
    df_event['y'] = reduced[:, 1]

    # Scatter plot of tweets
    fig = px.scatter(
        df_event, x='x', y='y', 
        hover_data=['text', 'eventType'], 
        title="Tweets Map (t-SNE)"
    )
    st.plotly_chart(fig)

# ----------------------------
# SOCIAL GRAPH (based on mentions)
# ----------------------------

def show_social_graph(df_event):
    G = nx.Graph()
    for _, row in df_event.iterrows():
        # Simulate user IDs based on tweet ID
        id_str_safe = str(row['id_str']).split('.')[0]
        user = f"user_{id_str_safe[-2:]}" 
        G.add_node(user)
        mentions = re.findall(r'@(\w+)', row['text'])
        for mention in mentions:
            G.add_edge(user, mention)
    
    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Network graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5), mode='lines', name='Edges'))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        mode='markers+text',
        marker=dict(size=10),
        text=[n for n in G.nodes()],
        name='Users'
    ))
    st.subheader("Social Graph based on mentions")
    st.plotly_chart(fig)

from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# TOY QUERY DATASET (can be extended)
# ----------------------------
toy_queries = [
    ["fire", "evacuation", "emergency"],
    ["flood", "damage", "water"],
    ["earthquake", "victims", "rescue"],
    ["shooting", "incident", "police"],
    ["typhoon", "alert", "disaster"]
]

# ----------------------------
# SEARCH ENGINE (TF-IDF based IR)
# ----------------------------

def search_engine(df, query_list, top_k=5):
    # Prepare TF-IDF vectorizer on all tweets
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
    
    # Prepare query as a space-separated string
    query = ' '.join(query_list)
    query_vector = vectorizer.transform([query])
    
    # Compute cosine similarity between query and all tweets
    cos_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top-k tweet indices
    top_indices = cos_similarities.argsort()[-top_k:][::-1]
    
    # Return top-k tweets + similarity scores
    return df.iloc[top_indices][['text', 'eventType', 'timestamp']], cos_similarities[top_indices]

# ----------------------------
# STREAMLIT UI FOR SEARCH SYSTEM
# ----------------------------

def show_search_system(df):
    st.subheader("🔍 IR-based Search Engine")
    
    # Toy queries dropdown for testing
    selected_query = st.selectbox("Select a Toy Query", toy_queries)
    
    # Input field for custom query
    custom_query = st.text_input("Or enter your own keywords separated by space (e.g., fire evacuation)")
    
    if custom_query:
        query_list = custom_query.lower().split()
    else:
        query_list = selected_query
    
    k = st.slider("Top-k results", 1, 10, 5)
    
    if st.button("Search"):
        top_tweets, scores = search_engine(df, query_list, top_k=k)
        for idx, row in top_tweets.iterrows():
            st.write(f"**[{row['timestamp']}] {row['eventType']}**")
            st.write(row['text'])
            st.write(f"Relevance Score: `{scores[top_tweets.index.get_loc(idx)]:.4f}`")
            st.markdown("---")

# ----------------------------
# STREAMLIT APP MAIN INTERFACE
# ----------------------------

st.title("🌍 Real Event-based Twitter Dashboard")

df = load_data()
df = preprocess_tweets(df)

# Dropdown to select the event type
event_types = df['eventType'].dropna().unique()
selected_event = st.selectbox('Select Event Type', event_types)
df_event = df[df['eventType'] == selected_event]

st.header(f"Dashboard for {selected_event.capitalize()} Events")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Timeline", 
    "Clusters", 
    "Word2Vec", 
    "Tweet Embeddings", 
    "Social Graph",
    "Search System",
    "Model Evaluation"
])


with tab1:
    plot_timeline(df_event)

with tab2:
    show_clusters(df_event, n_clusters=3)

with tab3:
    word2vec_demo(df_event)

with tab4:
    tweet_embeddings_tsne(df_event)

with tab5:
    show_social_graph(df_event)

with tab6:
    show_search_system(df)

with tab7:
    st.header("Model Evaluation Summary")

    models = ["xgboost", "random_forest", "lightgbm"]

    for model_name in models:
        st.subheader(model_name.replace("_", " ").title())
        try:
            report_path = f"results/{model_name}/report.csv"
            df = pd.read_csv(report_path, index_col=0)

            df.index = df.index.str.strip().str.lower()
            if 'accuracy' in df.index:
                accuracy = df.loc['accuracy', 'f1-score']
                st.markdown(f"**Accuracy:** `{accuracy:.4f}`")

            # Remove accuracy row for metric table
            df_metrics = df.drop(index=['accuracy'], errors='ignore')
            st.dataframe(df_metrics)

        except FileNotFoundError:
            st.warning(f"No report found for {model_name}.")

def load_my_dataframe():
    df = df_event
    return df
