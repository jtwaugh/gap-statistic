# Gap Statistic Method for K-Means Clustering

This is a script for running the gap statistic method outlined in [Tibishirani, et al. (2001)](https://statweb.stanford.edu/~gwalther/gap).

In short, when we use the K-means method for clustering, we often want to know how may clusters we need, i.e. what's an optimal value for k.

The way this is accomplished is to sum, over all clusters, the sum of distances of the points of each from its centroid, thereby obtaining the "compactness" of a k-means clustering. Thus each k has a compactness. Then, we compute to compactness of a k-means clustering on uniformly distributed random data with N samples (N being the number of data we're originally trying to cluster). Then we subtract the log of the former from the log of the latter at each k, i.e. we obtain the "gap" between k-means clustering. This value is the amount of information gained from clustering the input data versus clustering random data. The optimal value k is the greatest k whose gap value is greater than the gap value at k-1 plus the standard deviation of the random sampling. That is, we take the largest k past which we're simply losing information to noise by adding more clusters.

Anyway, the script takes in a list of data and returns a list of labels to each datum. The optimal k is max(labels) + 1.
