from unittest import result
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Step 1: Load your data
movies = pd.read_csv("movies.csv")

# Step 2: Keep only the important columns
movies = movies[['id', 'title', 'genres', 'overview', 'keywords']]

# Step 3: Handle missing data
movies.dropna(inplace=True)

# Step 4: Combine relevant text columns into a 'tags' column
def clean(x):
    return x.replace(" ", "").replace("-", "").lower()

movies['tags'] = movies['genres'] + ' ' + movies['overview'] + ' ' + movies['keywords']
movies['tags'] = movies['tags'].apply(lambda x: x.lower())

# Step 5: Vectorize the text
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Step 6: Calculate similarity
similarity = cosine_similarity(vectors)

# Step 7: Save the data
pickle.dump(movies[['id', 'title']], open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("✅ Files saved: movies.pkl and similarity.pkl")
# Step 8: Save the CountVectorizer
pickle.dump(cv, open('cv.pkl', 'wb'))   
