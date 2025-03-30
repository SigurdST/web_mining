import pandas as pd
import json

def load_nodes(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            obj = json.loads(line.strip())
            node = obj['n']
            node['properties']['type'] = node['labels'][0]
            node['properties']['node_id'] = node['id']
            data.append(node['properties'])
    return pd.DataFrame(data)

def load_relationships(file_path):
    relations = []
    with open(file_path, 'r') as f:
        for line in f:
            obj = json.loads(line.strip())
            r = obj['r']
            relations.append({
                'rel_id': r['id'],
                'type': r['label'],
                'start_id': r['start']['id'],
                'start_label': r['start']['labels'][0],
                'end_id': r['end']['id'],
                'end_label': r['end']['labels'][0]
            })
    return pd.DataFrame(relations)

def load_and_merge_all_data():
    tweets = load_nodes('data/database/Nodes/Tweet.json')
    users = load_nodes('data/database/Nodes/User.json')
    hashtags = load_nodes('data/database/Nodes/Hashtag.json')
    events = load_nodes('data/database/Nodes/Event.json')
    categories = load_nodes('data/database/Nodes/PostCategory.json')

    has_category = load_relationships('data/database/Relationships/HAS_CATEGORY.json')
    has_hashtag = load_relationships('data/database/Relationships/HAS_HASHTAG.json')
    is_about = load_relationships('data/database/Relationships/IS_ABOUT.json')
    mentions = load_relationships('data/database/Relationships/MENTIONS.json')
    posted = load_relationships('data/database/Relationships/POSTED.json')

    has_category = has_category.merge(categories, left_on='end_id', right_on='node_id', how='left')
    has_category = has_category.rename(columns={'id': 'category'})
    tweets = tweets.merge(has_category[['start_id', 'category']], left_on='node_id', right_on='start_id', how='left')

    has_hashtag = has_hashtag.merge(hashtags, left_on='end_id', right_on='node_id', how='left')
    has_hashtag_grouped = has_hashtag.groupby('start_id')['id'].apply(list).reset_index(name='hashtags')
    tweets = tweets.merge(has_hashtag_grouped, left_on='node_id', right_on='start_id', how='left')

    is_about = is_about.merge(events, left_on='end_id', right_on='node_id', how='left')
    is_about = is_about.rename(columns={'id': 'event_id', 'eventType': 'event_type'})
    tweets = tweets.merge(is_about[['start_id', 'event_id', 'event_type']], left_on='node_id', right_on='start_id', how='left')

    posted = posted.merge(users, left_on='start_id', right_on='node_id', how='left')
    posted = posted.rename(columns={'screen_name': 'author'})
    tweets = tweets.merge(posted[['end_id', 'author', 'followers_count', 'friends_count']], left_on='node_id', right_on='end_id', how='left')

    tweets = tweets[tweets['text'].notna() & tweets['annotation_postPriority'].notna()].copy()
    tweets = tweets[tweets['annotation_postPriority'].isin(['Low', 'Medium', 'High'])]
    tweets['annotation_postPriority'] = tweets['annotation_postPriority'].astype(str)

    return tweets