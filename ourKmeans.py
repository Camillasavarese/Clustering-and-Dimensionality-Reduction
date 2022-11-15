import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt


def k_means_mio(X, k, max_iter=100):
    ''' It performs out version of kmeans '''

    if isinstance(X, pd.DataFrame):  # True if it is a dataframe
        X = X.values

    # Pick indices of k random point without replacement
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx, :]

    # Calculate the class of each point using euclidean distance
    matrice = distance.cdist(X, centroids, 'euclidean')
    C = np.argmin(matrice, axis=1)
    for _ in range(max_iter):
        centroids = np.vstack([X[C == i, :].mean(axis=0) for i in range(k)])
        matrice_def = distance.cdist(X, centroids, 'euclidean')
        t = np.argmin(matrice_def, axis=1)

        # Stop when C matrix doesn't change
        if np.array_equal(C, t):
            break

        C = t

    # Return an array containg class of each data point, and the centroids
    return C, centroids, matrice_def


def gap_stat(data_cluster, wcss):
    ''' It performs gap statistic '''
    randomunif = np.random.random_sample(size=data_cluster.shape)
    cost_u = []
    gaps = []
    s_k = []
    for i in range(3, 15):
        # Compute k-means and get only the distance metrix and the disctionary

        _, _, matrice_dist_unif = k_means_mio(randomunif, i)
        costo = sum(matrice_dist_unif.min(axis=1)**2)

        cost_u.append(costo)
        # Compute the gap statistic
        gap = np.log(np.mean(cost_u)) - np.log(wcss[i-3])

        # Keep the value of gap statistic for each value of k
        gaps.append(gap)

        # Compute the standard deviation (of the random part)
        s_k.append(np.std(np.log(np.mean(cost_u)))*np.sqrt(1+(1/len(cost_u))))

    # Plot the curve
    plt.figure(figsize=(12, 5))
    plt.plot(list(range(3, 15)), gaps, linestyle='-', marker='o',
             color='lightseagreen', lw=3, alpha=.7, markersize=6)
    plt.xlabel("Numbers of clusters")
    plt.ylabel("Gap Stistics")

    # Get the better value of k according to gap statistics
    # By definition k_star is the min {k | G[k] >= G[k+1] - s_k[k+1]
    k_ = []
    for j in range(0, 15-4):
        if (gaps[j] >= gaps[j+1] - s_k[j+1]):
            k_.append(j)
    k_star = min(k_)
    plt.plot(k_star+3, gaps[k_star],  marker='o',
             markerfacecolor="darkcyan", markersize=10)
    plt.show()
    print(k_star+3)
