import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and clean data
movies = pd.read_csv('movies.csv')
movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].fillna('')

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index map
indices = pd.Series(movies.index, index=movies['title'].str.lower())

API_KEY = '3872f8d9722700b18846b161366e7c92'

def fetch_poster(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}'
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    poster_path = data.get('poster_path')
    return f'https://image.tmdb.org/t/p/w500{poster_path}' if poster_path else None

def get_genres():
    genre_list = set()
    for g in movies['genres']:
        try:
            items = eval(g) if isinstance(g, str) else []
            for item in items:
                if isinstance(item, dict) and 'name' in item:
                    genre_list.add(item['name'])
                elif isinstance(item, str):
                    genre_list.add(item.strip())
        except:
            continue
    return sorted(genre_list)

def recommend(title, genre=None, n=5):
    title = title.lower()
    if title not in indices:
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    results = []
    for i, _ in sim_scores:
        row = movies.iloc[i]
        if genre:
            if genre not in str(row['genres']):
                continue
        movie_id = row['id']
        poster_url = fetch_poster(movie_id)
        results.append((row['title'], poster_url))
        if len(results) == n:
            break

    return results