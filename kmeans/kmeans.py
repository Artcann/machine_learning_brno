import seaborn as sns
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import random

def euclidian_distance(centroid, data) -> list:
    arr = np.empty(len(data))
    for i in range(len(data)):
        point = data[i]
        dist = 0
        for j in range(len(centroid)):
            dist += np.square(point[j] - centroid[j])
        arr[i] = np.sqrt(dist)
    return arr

def kmean(centroids, data):
    dist_mat = np.zeros((len(data), len(centroids)))

    for n in range(len(centroids)):
        distance = euclidian_distance(centroids[n], data)
        dist_mat[:, n] = distance

    clusters = []
    for row in dist_mat:
        clusters.append(np.argmin(row))
    
    return clusters

centers = 3
X_train, true_labels = make_blobs(n_samples=1000, centers=centers, random_state=40)
X_train = StandardScaler().fit_transform(X_train)

#TODO make a method that determines the  optimal number of clusters. Try elbow curve.
centroids = random.sample(list(X_train), centers) 
new_centroids = np.copy(centroids)

centroid_moving = True
current_iteration = 0
maximum_iteration = 100

while centroid_moving:
    clusters = kmean(centroids, X_train)

    for i in range(len(centroids)):
        proxy_mat = np.c_[clusters, X_train]
        new_centroids[i] = np.mean(proxy_mat[proxy_mat[:, 0] == i, 1:], axis=0)
    
    current_iteration += 1

    if current_iteration > maximum_iteration:
        print("Maximum precision reached")
        centroid_moving = False
        
    elif np.allclose(centroids, new_centroids):
        print("Centroids not moving in iterations", current_iteration)
        centroid_moving = False

    centroids = np.copy(new_centroids)

x = [X[0] for X in X_train]
x.extend(centroids[:, 0])
y = [X[1] for X in X_train]
y.extend(centroids[:, 1])

clusters.extend([9] * centers)

sns.scatterplot(x=x,
                y=y,
                legend=None,
                hue=clusters,
                palette="deep"
                )

plt.xlabel("x")
plt.ylabel("y")
plt.show()
