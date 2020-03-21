from flask import Flask, render_template

import pickle
from dataprocess import getPopularMovies, getFullMovies, getRecommendedMovies, searchForMovies

# EB looks for an 'application' callable by default.
application = Flask(__name__, static_url_path='',
                    static_folder='dist',
                    template_folder='dist')

mfModel = pickle.load(open('mfModel.pkl', "rb"))
knnModel = pickle.load(open('knnModel.pkl', "rb"))


@application.route("/")
def index():
    return render_template("index.html")


@application.route('/api/fullMovies')
def getFullMovieList():
    return getFullMovies()


@application.route('/api/popularMovies')
def getPopularMoviesList():
    return getPopularMovies()


@application.route('/api/recommendedMovies/<movieIndex>')
def getRecommendedMoviesList(movieIndex):
    movieFeatures = mfModel.Q
    selectedMovieFeature = movieFeatures[int(movieIndex), :]
    dist, indices = knnModel.kneighbors([selectedMovieFeature], 20)
    movies = getRecommendedMovies(dist, indices)
    return movies


@application.route('/api/search/<param>')
def getMoviesByTitle(param):
    selectedMovies = searchForMovies(param)
    return selectedMovies


# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()
