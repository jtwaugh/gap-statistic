import numpy as np

from sklearn.cluster import KMeans

printResults = True

# Helper functions for the below
def bounding_box(X):
    xmin, xmax = min(X, key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
    ymin, ymax = min(X, key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
    return (xmin,xmax), (ymin,ymax)

# Compactness of a given clustering
def compactness(data, centers, labels):
    kSum = 0.0
    for i in range(len(data)):
        clusterNum = labels.item(i)
        kSum += np.linalg.norm(centers[clusterNum] - data[i])**2
    return kSum

# Use the gap method to determine a k-value and then cluster
# This will break if we suffer the curse
# Returns a dict of lists of points
# Spoiler alert: this doesn't get us more than two on anything in this dataset
# Was a cool exercise, though
def gapKMeans(data, kRange=30, sampleTrials=10, printResults=True):
    # Dimensions of the data
    numData, numFeatures = data.shape
    # Get the bounding cell of the data
    extrema = [[min(data[:,dim]), max(data[:,dim])] \
                for dim in range(numFeatures)]
    # We'll use the gap statistic to estimate the best clustering between k=1 and 30
    gap = np.zeros(kRange)
    s = np.zeros(kRange)
    # At each value in the range, try k clusters and estimate
    for k in range(kRange):
        # Get a k-means object
        kclusters = KMeans(n_clusters=(k + 1))
        # Get the clusters in a dict
        labels = kclusters.fit_predict(data)
        # Compute compactness
        kCompactness = np.log(compactness(data, kclusters.cluster_centers_, labels))
        # Take some trials of random data to compare
        trialCompactness = np.zeros(sampleTrials)
        for trial in range(sampleTrials):
            # Generate a set of points
            samples = np.zeros((numData, numFeatures))
            for sample in range(numData):
                samples[sample] = [np.random.uniform(extrema[dim][0], \
                                     extrema[dim][1]) for dim in range(numFeatures)]
            # Get the clusters
            sampleLabels = kclusters.fit_predict(samples)
            trialCompactness[trial] = \
                np.log(compactness(samples, kclusters.cluster_centers_, \
                                         sampleLabels))
        gap[k] = sum(trialCompactness - kCompactness)**2 / sampleTrials
        # Average compactness over the trials
        wbar = sum(trialCompactness) / sampleTrials
        # Take the standard deviation of the clustering accuracy over the trials
        sd = np.sqrt(sum((trialCompactness - wbar)**2) / sampleTrials)
        s[k] = np.sqrt(1 + 1/sampleTrials) * sd
        # If we cluster unambiguously better than k+1, we're done
        # This is because of some monotonicity condition
        if printResults:
            print("Gap value at ", k, "clusters: ", gap[k - 1])
            print("Need to beat ", gap[k] - s[k - 1])
        if k > 1 and gap[k - 1] > gap[k] - s[k - 1]:
            return labels
    return labels