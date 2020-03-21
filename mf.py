import numpy as np
import time


class MF:
    def __init__(self, movieCount, userCount, meanRate, alpha=0.01, reg=0.01, iterations=20, K=30):
        self.alpha = alpha
        self.reg = reg
        self.iterations = iterations
        self.K = K
        self.userCount = userCount
        self.movieCount = movieCount
        self.P = np.random.normal(scale=1. / self.K, size=(userCount, self.K))
        self.Q = np.random.normal(scale=1. / self.K, size=(movieCount, self.K))
        self.meanRate = meanRate
        self.userBias = np.zeros(userCount)
        self.itemBias = np.zeros(movieCount)

    def train(self, rating_df):
        for i in range(self.iterations):
            start_time = time.time()
            for row in rating_df[['userId', 'movieIndex', 'rating']].values:
                self.sgd(int(row[0] - 1), int(row[1]), row[2])
            end_time = time.time()
            print('iteration {} Time {}'.format(i, end_time - start_time))

    def sgd(self, user, movie, rate):
        prediction = self.P[user, :].dot(self.Q[movie, :]) + \
                     self.meanRate[movie] + \
                     self.userBias[user] + \
                     self.itemBias[movie]
        # Find error
        err = rate - prediction
        self.P[user, :] = self.P[user, :] + (2 * self.alpha) * ((err * self.Q[movie, :]) - (self.reg * self.P[user, :]))
        self.Q[movie, :] = self.Q[movie, :] + (2 * self.alpha) * (
                (err * self.P[user, :]) - (self.reg * self.Q[movie, :]))
        self.userBias[user] = self.userBias[user] + (2 * self.alpha) * (err - (self.reg * self.userBias[user]))
        self.itemBias[movie] = self.itemBias[movie] + (2 * self.alpha) * (err - (self.reg * self.itemBias[movie]))
