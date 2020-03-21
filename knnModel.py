from sklearn.neighbors import NearestNeighbors
import pickle

mfModel = pickle.load(open('mfModel.pkl', "rb"))

knn = NearestNeighbors(metric="cosine", algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(mfModel.Q)

pickle.dump(knn, open("knnModel.pkl", 'wb'))
