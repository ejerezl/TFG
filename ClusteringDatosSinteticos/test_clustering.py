import warnings
import time
from itertools import islice, cycle
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import multivariate_normal

from sklearn import cluster, datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_gaussian_quantiles
from sklearn.preprocessing import StandardScaler
import math
import time


n_samples = 1500


def hierarchical_clustering():
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5,
                                          noise=0.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)

    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)

    no_structure = np.random.rand(n_samples, 2), None

    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5],
        random_state=random_state
    )

    # Set up cluster parameters
    plt.figure(figsize=(9 * 1.3 + 2, 14.5))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01
    )

    plot_num = 1
    n_clusters = list(range(45, 50))
    default_base = {"n_neighbors": 10, "n_clusters": 3}
    data = []
    for ele in n_clusters:
        data += [
            (
                varied,
                {
                    "n_clusters": ele,
                    "eps": 0.18,
                    "n_neighbors": 2,
                    "min_samples": 5,
                    "xi": 0.035,
                    "min_cluster_size": 0.2,
                },
            )]

    data1 = [
        (noisy_circles, {"n_clusters": 2}),
        (noisy_moons, {"n_clusters": 2}),
        (
            varied,
            {
                "n_clusters": 3,
                "eps": 0.18,
                "n_neighbors": 2,
                "min_samples": 5,
                "xi": 0.035,
                "min_cluster_size": 0.2,
            },
        ),
        (
            aniso,
            {
                "eps": 0.15,
                "n_neighbors": 2,
                "min_samples": 20,
                "xi": 0.1,
                "min_cluster_size": 0.2,
            },
        ),
        (blobs, {}),
        (no_structure, {}),
    ]

    for i_dataset, (dataset, algo_params) in enumerate(data):
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)

        X, y = dataset

        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # ============
        # Create cluster objects
        # ============
        print(params["n_clusters"])
        print("\n")
        complete = cluster.AgglomerativeClustering(
            n_clusters=params["n_clusters"], linkage="complete"
        )
        average = cluster.AgglomerativeClustering(
            n_clusters=params["n_clusters"], linkage="average"
        )
        single = cluster.AgglomerativeClustering(
            n_clusters=params["n_clusters"], linkage="single"
        )

        clustering_algorithms = (
            ("Single Linkage", single),
            ("Average Linkage", average),
            ("Complete Linkage", complete),
        )

        for name, algorithm in clustering_algorithms:

            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the "
                            + "connectivity matrix is [0-9]{1,2}"
                            + " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning,
                )
                algorithm.fit(X)

            t1 = time.time()
            if hasattr(algorithm, "labels_"):
                y_pred = algorithm.labels_.astype(int)
            else:
                y_pred = algorithm.predict(X)

            plt.subplot(len(data), len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)

            colors = np.array(
                list(
                    islice(
                        cycle(
                            [
                                "#377eb8",
                                "#ff7f00",
                                "#4daf4a",
                                "#f781bf",
                                "#a65628",
                                "#984ea3",
                                "#999999",
                                "#e41a1c",
                                "#dede00",
                            ]
                        ),
                        int(max(y_pred) + 1),
                    )
                )
            )
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += 1

    plt.show()


def all_together():
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5,
                                          noise=0.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)

    blobs3 = datasets.make_blobs(n_samples=n_samples,
                                 centers=[[0, 0], [0, 4], [0, 8]])
    blobs = datasets.make_blobs(n_samples=n_samples,
                                centers=[[0, 0], [1, 1], [2, 2]],
                                cluster_std=[0.5, 0.5, 0.5])
    blobs2 = datasets.make_blobs(n_samples=n_samples,
                                 centers=[[0, 0], [2, 2], [3, 3]],
                                 cluster_std=[0.5, 0.5, 0.5])

    no_structure = np.random.rand(n_samples, 2), None

    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5],
        random_state=random_state
    )

    # Set up cluster parameters
    plt.figure(figsize=(9 * 1.3 + 2, 14.5))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01
    )

    plot_num = 1

    default_base = {"n_neighbors": 5, "n_clusters": 3}

    data = [
        (noisy_circles, {"n_clusters": 2, }),
        (noisy_moons, {"n_clusters": 2, }),
        (varied,
         {
             "n_clusters": 3,
             "eps": 0.18,
             "n_neighbors": 2,
             "min_samples": 5,
             "xi": 0.035,
             "min_cluster_size": 0.2,
         },
         ),
        (
            aniso,
            {
                "eps": 0.15,
                "n_neighbors": 2,
                "min_samples": 20,
                "xi": 0.1,
                "min_cluster_size": 0.2,
            },
        ),
        (blobs, {}),
    ]
    seed = math.floor(time.time())
    for i_dataset, (dataset, algo_params) in enumerate(data):
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)
        seed += i_dataset
        X, y = dataset

        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # ============
        # Create cluster objects
        # ============
        complete = cluster.AgglomerativeClustering(
            n_clusters=params["n_clusters"], linkage="complete"
        )
        average = cluster.AgglomerativeClustering(
            n_clusters=params["n_clusters"], linkage="average"
        )
        single = cluster.AgglomerativeClustering(
            n_clusters=params["n_clusters"], linkage="single"
        )
        kmeans = KMeans(n_clusters=params["n_clusters"], random_state=seed)
        clustering_algorithms = (
            ("Similitud simple", single),
            ("Similitud media", average),
            ("Similitud completa", complete),
            ("K medias", kmeans)
        )

        for name, algorithm in clustering_algorithms:

            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the "
                            + "connectivity matrix is [0-9]{1,2}"
                            + " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning,
                )
                algorithm.fit(X)

            t1 = time.time()
            if hasattr(algorithm, "labels_"):
                y_pred = algorithm.labels_.astype(int)
            else:
                y_pred = algorithm.predict(X)

            plt.subplot(len(data), len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)

            colors = np.array(
                list(
                    islice(
                        cycle(
                            [
                                "#377eb8",
                                "#ff7f00",
                                "#4daf4a",
                                "#f781bf",
                                "#a65628",
                                "#984ea3",
                                "#999999",
                                "#e41a1c",
                                "#dede00",
                            ]
                        ),
                        int(max(y_pred) + 1),
                    )
                )
            )
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += 1

    plt.show()


def k_means():
    '''noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5,
                                          noise=0.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
    '''

    blobs = datasets.make_blobs(n_samples=n_samples,
                                centers=[[0, 0], [0, 4], [0, 8]],
                                cluster_std=0.5)
    centers = [[0, 0], [0, 1], [1, 1]]
    blobs2 = datasets.make_blobs(n_samples=n_samples, centers=centers)

    blobs3 = datasets.make_blobs(n_samples=n_samples,
                                 centers=[[0, 0], [0, 4], [0, 8]])

    datasetes = [blobs, blobs2, blobs3, blobs3]

    kmeans = KMeans(n_clusters=3)
    plt.figure(figsize=(4 * 2 + 2, 4 * 2 + 2))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01
    )
    plt.suptitle("K-means with k = 2")
    i = 0
    for dataset in datasetes:
        X, y = dataset
        X = StandardScaler().fit_transform(X)

        pred = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_
        if i == 0:
            plt.subplot(221)
        elif i == 1:
            plt.subplot(222)
        elif i == 2:
            plt.subplot(223)
        else:
            plt.subplot(224)
        plt.scatter(X[:, 0], X[:, 1], c=pred, s=10)
        for center in centers:
            plt.scatter(center[0], center[1], marker="X", color='red')

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        i = i + 1
    plt.show()


def single_linkage_test():
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    no_structure = np.random.rand(n_samples, 2), None

    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5],
        random_state=random_state
    )

    # Set up cluster parameters
    plt.figure(figsize=(9 * 1.3 + 2, 14.5))
    plt.subplots_adjust(
        left=0.08, right=0.92, bottom=0.001, top=0.96, wspace=0.2, hspace=0.01
    )
    n_clusters = [[3, 44, 45, 46], [3, 4, 5, 16, 17],
                  [math.floor(math.exp(i)) for i in [2, 3, 4, 5]]]
    default_base = {"n_neighbors": 10, "n_clusters": 3}
    data = [
        (
            varied,
            {
                "eps": 0.18,
                "n_neighbors": 2,
                "min_samples": 5,
                "xi": 0.035,
                "min_cluster_size": 0.2,
            },
        ),
        (
            aniso,
            {
                "eps": 0.15,
                "n_neighbors": 2,
                "min_samples": 20,
                "xi": 0.1,
                "min_cluster_size": 0.2,
            },
        ),
        (no_structure, {})
    ]

    for i_cluster, n_cluster in enumerate(n_clusters):

        # Obtain data
        dataset, algo_params = data[i_cluster]
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)

        X, y = dataset

        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)
        plot_num = i_cluster + 1
        for num in n_cluster:
            single = cluster.AgglomerativeClustering(
                n_clusters=num, linkage="single"
            )
            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the "
                            + "connectivity matrix is [0-9]{1,2}"
                            + " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning,
                )
                single.fit(X)
            if hasattr(single, "labels_"):
                y_pred = single.labels_.astype(int)
            else:
                y_pred = single.predict(X)
            plt.subplot(len(n_cluster), len(data), plot_num)
            title = "n_clusters = " + str(num)
            plt.ylabel(title, size=16)

            colors = np.array(
                list(
                    islice(
                        cycle(
                            [
                                "#377eb8",
                                "#ff7f00",
                                "#4daf4a",
                                "#f781bf",
                                "#a65628",
                                "#984ea3",
                                "#999999",
                                "#e41a1c",
                                "#dede00",
                            ]
                        ),
                        int(max(y_pred) + 1),
                    )
                )
            )
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += len(data)

    plt.suptitle("Single Linkage", size=20)
    plt.show()

def print_executing_options():

    print("Options are:")
    print("    -\'all\': for executing a test where all options are included ")
    print("    -\'kmeans\': for executing a test where only K-means is included ")
    print("    -\'hierarchical\': for executing a test where all hierarchical clustering algorithms are included ")
    print("    -\'singleLinkage\': for executing a test where only hierarchical clustering with simple linkage is included ")


if __name__ == '__main__':
    print(arg)
    if len(arg) < 2:
        print("You need to introduce an option when executing this file.")
        print_executing_options()
    elif arg[1] == 'all':
        all_together()
    elif arg[1] == 'kmeans':
        k_means()
    elif arg[1] == 'hierarchical':
        hierarchical_clustering()
    elif arg[1] == 'singleLinkage':
        single_linkage_test()
    else:
        print("wrong option.")
        print_executing_options()