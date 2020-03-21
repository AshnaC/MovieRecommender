import pandas as pd

movies = pd.read_csv("data/movies.dat", sep="::", header=None, names=['movieId', 'title', 'genres'])
ratings = pd.read_csv("data/ratings.dat", sep="::", header=None, names=['userId', 'movieId', 'rating', 'timestamp'])
links = pd.read_csv("data/links.csv")

movies = pd.merge(movies, links, on="movieId", how="left")
movies['imdbId'] = movies['imdbId'].apply(lambda x: -99 if pd.isnull(x) else 'tt0' + str(int(x)))
movieIndex = ratings.groupby("movieId").ngroup()
ratings['movieIndex'] = movieIndex
movies = pd.merge(movies, ratings.drop_duplicates(subset=['movieId'])[['movieIndex', 'movieId']], on="movieId",
                  how="left")
movies['movieIndex'] = movies['movieIndex'].fillna(-99).astype(int)

ratings_mean_group = ratings.groupby("movieIndex").mean()
ratings_count_group = ratings.groupby("movieIndex").count()
ratings_group = pd.merge(ratings_count_group[['movieId']],
                         ratings_mean_group[['rating']],
                         on="movieIndex",
                         how="left")

meanRatings = ratings_mean_group['rating']

ratings_group.rename(columns={"movieId": "counts"}, inplace=True)
full_mean_rating = ratings['rating'].mean()


def getWeightedRating(x, min_votes, mean_rate):
    v = x['counts']
    r = x['rating']
    wr = ((v * r) + (min_votes * mean_rate)) / (v + min_votes)
    return wr


ratings_group['weighted_rate'] = ratings_group.apply(lambda x: getWeightedRating(x, 100, full_mean_rating), axis=1)
ratings_group.sort_values(by="weighted_rate", ascending=False, inplace=True)
ratings_group = ratings_group.merge(movies, how="left", on="movieIndex")
popular_movies = ratings_group.head(21)


def getPopularMovies():
    return popular_movies.to_json(orient='records')


def getFullMovies():
    return movies.to_json(orient='records')


def getRecommendedMovies(dist, indices):
    recommended_df = pd.DataFrame(indices[0], columns=['movieIndex'])
    recommended_df['dist'] = dist[0]
    recommended_df = pd.merge(recommended_df, movies, how="left", on="movieIndex")
    return recommended_df.to_json(orient='records')


def searchForMovies(title):
    selectedMovies = movies[movies['title'].apply(lambda x: title.lower() in x.lower())]
    return selectedMovies.to_json(orient='records')
