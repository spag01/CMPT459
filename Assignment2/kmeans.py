import numpy as np

np.random.seed(0)

class KMeans():

    def __init__(self, n_clusters: int, init: str='random', max_iter = 300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None # Initialized in initialize_centroids()

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)
        # print(self.centroids.shape)
        iteration = 0
        clustering = np.zeros(X.shape[0])
        # while iteration < self.max_iter:
        for i in range(self.max_iter):
            # calculate pairwise dist and get the argmin index as cluster label
            clustering = self.euclidean_distance(self.centroids, X).argmin(axis=0)
            self.update_centroids(clustering, X)
            # iteration += 1

        # define the final cluster    
        clustering = self.euclidean_distance(self.centroids, X).argmin(axis=0)
        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        #your code
        for i in range(self.n_clusters):
            # get rows for cluster i
            rows = X[clustering == i]
            # update centroid with mean of the cluster (if cluster not empty)
            if rows.shape[0] != 0:
                self.centroids[i] = rows.mean(axis=0)

    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        if self.init == 'random':
            # your code
            self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        elif self.init == 'kmeans++':
            # your code
            self.centroids = np.zeros((self.n_clusters, X.shape[1]))

            # Randomly select the first centroid from the data points
            self.centroids[0] = X[np.random.choice(X.shape[0], 1)]

            for i in range(1, self.n_clusters):
                # shape(X.shape[0], ) row number = number of nearest distance
                nearest_dist = self.euclidean_distance(self.centroids[0:i], X).min(axis=0)
                # normalize into probability
                p = nearest_dist / nearest_dist.sum()
                # sample against distribution
                row = np.random.choice(np.arange(p.shape[0]), p=p)
                # use the sampled row as the next centroid
                self.centroids[i] = X[row]
        else:
            raise ValueError('Centroid initialization method should either be "random" or "k-means++"')

    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        # your code
        # pairwise broadcast minus
        
        return np.sqrt(((X1[:, :, np.newaxis] - X2[:, :, np.newaxis].T) ** 2).sum(axis=1))

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        # your code

        # Count the size of each cluster
        cluster_freq = np.bincount(clustering)

        # scores for each sample
        scores = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            # sample distance to all other points
            dist = self.euclidean_distance(X[i].reshape(1, -1), X)[0]
            # group by each cluster, sum the distance of sameple i to other samples in that cluster
           
            cluster_dist = np.bincount(clustering, weights=dist, minlength=self.n_clusters)
            # get the cluster
            cluster = clustering[i]
            # calculate intra dist using formula
            intra_dist = cluster_dist[cluster] / (cluster_freq[cluster] - 1)
            # set the intra cluster distance sum to inf so we can ignore it later
            cluster_dist[cluster] = np.inf
            # take the minimum inter cluster distance mean of all other cluster
            inter_dist = (cluster_dist / cluster_freq).min()
            # silhouette formula
            numerator = inter_dist - intra_dist
            denominator = np.maximum(inter_dist, intra_dist)
            # use nan to num to make nan 0 (sometime cluster freq will be 0)
            score = np.nan_to_num(numerator / denominator)
            # store score
            scores[i] = score
        
        #take score mean
        return scores.mean()