from flask import Flask, render_template, request
from recommender import recommend, get_genres
import os

app = Flask(__name__)
app.config['TMDB_API_KEY'] = os.getenv('TMDB_API_KEY', '3872f8d9722700b18846b161366e7c92')

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    error = None
    genres = get_genres()
    
    if request.method == 'POST':
        movie = request.form['movie']
        selected_genre = request.form.get('genre')
        try:
            recommendations = recommend(movie, selected_genre)
            if not recommendations:
                error = "No recommendations found. Try a different movie."
        except Exception as e:
            error = f"An error occurred: {str(e)}"
    
    return render_template('index.html', 
                        recommendations=recommendations, 
                        genres=genres,
                        error=error)

if __name__ == '__main__':
    app.run(debug=True)