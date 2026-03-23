from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from newsapi import NewsApiClient
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from datetime import datetime
import os
import threading

app = Flask(__name__, static_folder='.')
CORS(app)

# Replace with your actual API Key
import os
from dotenv import load_dotenv

# Load the variables from .env into the system environment
load_dotenv()

# Access the variables using os.getenv
api_key = os.getenv('API_KEY')



# Global state
data_cache = {}
data_lock = threading.Lock()
is_loading = False
load_error = None

def fetch_and_process():
    global data_cache, is_loading, load_error
    try:
        is_loading = True
        load_error = None

        newsapi = NewsApiClient(api_key=API_KEY)
        
        # --- 1. Fetch more data using Everything endpoint ---
        all_articles = []
        for p in range(1, 3):
            try:
                response = newsapi.get_everything(
                    q='business OR technology OR world OR science',
                    language='en',
                    sort_by='publishedAt',
                    page_size=100,
                    page=p
                )
                if response.get('status') == 'ok':
                    all_articles.extend(response.get('articles', []))
            except Exception as e:
                print(f"Page {p} fetch failed: {e}")
                break

        if not all_articles:
            raise Exception("No articles returned from API")

        df = pd.DataFrame([{
            'title': a.get('title', '') or '',
            'description': a.get('description', '') or '',
            'source': a.get('source', {}).get('name', '') or '',
            'publishedAt': a.get('publishedAt', ''),
            'url': a.get('url', ''),
            'urlToImage': a.get('urlToImage', '') or None,
        } for a in all_articles])

        # --- 2. Cleaning ---
        df = df[df['title'].notna() & (df['title'] != '[Removed]')]
        df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
        df = df.sort_values('publishedAt', ascending=False)
        df = df.drop_duplicates(subset='title')
        
        df['title_clean'] = df['title'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True)
        df['desc_clean'] = df['description'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True).fillna('')
        df['clean_text'] = df['title_clean'] + ' ' + df['desc_clean']
        df = df[df['clean_text'].str.strip().str.len() > 30].reset_index(drop=True)

        # --- 3. TF-IDF ---
        tfidf = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2, ngram_range=(1,2), max_features=2000)
        tfidf_matrix = tfidf.fit_transform(df['clean_text'])
        feature_names = tfidf.get_feature_names_out()

        # --- 4. LDA Topics ---
        n_topics = 5
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
        lda.fit(tfidf_matrix)
        topic_vals = lda.transform(tfidf_matrix)
        df['dominant_topic'] = topic_vals.argmax(axis=1)

        topic_labels = {}
        topic_keywords = {}
        for idx, comp in enumerate(lda.components_):
            # Top words for the label
            label_words = [feature_names[i] for i in comp.argsort()[:-4:-1]]
            label_str = " · ".join(label_words)
            topic_labels[idx] = label_str
            
            # Full keyword list for the renderKeywords frontend function
            topic_keywords[label_str] = [feature_names[i] for i in comp.argsort()[:-8:-1]]
            
        df['dominant_topic_label'] = df['dominant_topic'].map(topic_labels)

        # --- 5. Sentiment ---
        sia = SentimentIntensityAnalyzer()
        df['sentiment_score'] = df['clean_text'].apply(lambda t: sia.polarity_scores(t)['compound'])
        df['sentiment'] = df['sentiment_score'].apply(
            lambda s: 'Positive' if s >= 0.05 else ('Negative' if s <= -0.05 else 'Neutral')
        )

        # --- 6. ML Model ---
        le = LabelEncoder()
        y = le.fit_transform(df['sentiment'])
        acc = 0
        if len(df) > 15 and len(np.unique(y)) > 1:
            X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, y, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=1000, class_weight='balanced')
            model.fit(X_train, y_train)
            acc = round(accuracy_score(y_test, model.predict(X_test)) * 100, 1)
            df['sentiment_pred'] = le.inverse_transform(model.predict(tfidf_matrix))
        else:
            df['sentiment_pred'] = df['sentiment']
            acc = "N/A"

        # --- 7. Output Prep ---
        df['date'] = df['publishedAt'].dt.date.astype(str)
        df = df.replace({np.nan: None})

        articles_out = []
        for _, row in df.iterrows():
            articles_out.append({
                'title': row['title'],
                'description': row['description'],
                'source': row['source'],
                'publishedAt': row['publishedAt'].isoformat() if pd.notna(row['publishedAt']) else '',
                'url': row['url'],
                'urlToImage': row['urlToImage'],
                'topic': row['dominant_topic_label'],
                'topic_id': int(row['dominant_topic']),
                'sentiment': row['sentiment'],
                'sentiment_score': round(float(row['sentiment_score']), 3),
                'sentiment_pred': row['sentiment_pred'],
                'date': row['date'],
            })

        # Top keywords overall for word cloud or lists
        mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_idx = mean_tfidf.argsort()[-30:][::-1]
        top_keywords = [{'word': feature_names[i], 'score': round(float(mean_tfidf[i]), 4)} for i in top_idx]

        # --- 8. Build Cache ---
        with data_lock:
            data_cache = {
                'articles': articles_out,
                'stats': {
                    'total': len(df),
                    'sources': len(df['source'].unique()),
                    'topics': n_topics,
                    'model_accuracy': acc,
                },
                'sentiment_counts': df['sentiment'].value_counts().to_dict(),
                'source_counts': df['source'].value_counts().head(10).to_dict(),
                'topic_counts': df['dominant_topic_label'].value_counts().to_dict(),
                'topic_keywords': topic_keywords, # Fixes kws is not iterable
                'top_keywords': top_keywords,
                'sentiment_by_topic': df.groupby('dominant_topic_label')['sentiment_score'].mean().round(3).to_dict(),
                'timeline': df.groupby('date').size().reset_index(name='count').to_dict(orient='records'),
                'last_updated': datetime.utcnow().isoformat(),
            }

    except Exception as e:
        load_error = str(e)
        print(f"Error in fetch_and_process: {e}")
    finally:
        is_loading = False

# Thread Management
t = threading.Thread(target=fetch_and_process)
t.daemon = True
t.start()

@app.route('/api/status')
def status():
    return jsonify({'loading': is_loading, 'error': load_error, 'ready': bool(data_cache)})

@app.route('/api/data')
def get_data():
    with data_lock:
        if not data_cache:
            return jsonify({'error': 'Data not ready yet', 'loading': is_loading}), 503
        return jsonify(data_cache)

@app.route('/api/refresh', methods=['POST'])
def refresh():
    global is_loading
    if not is_loading:
        t = threading.Thread(target=fetch_and_process)
        t.daemon = True
        t.start()
    return jsonify({'message': 'Refresh started'})

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)