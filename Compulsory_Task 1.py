import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.metrics import pairwise_distances_argmin
from collections import Counter, defaultdict
import random
import math

"""
according to sources from the below linke, the steps to follow are:

(15 October, 2022), Example of K-Means Clustering in Python,
Available at: https://datatofish.com/k-means-clustering-python/

Zack, (31 August 2022), K-Means Clustering in Python: Step-by-Step Example,
Available at: https://www.statology.org/k-means-clustering-in-python/

Vikas Paruchuri, (11 Jul 2022), K-means Clustering From Scratch In Python [Machine Learning Tutorial], 
Available at: www.youtube.com available at: https://www.youtube.com/watch?v=lX-3nGHDhQg

Aden Haussmann, (Nov 11 2020), K-Means Clustering for Beginners, 
Available at: https://towardsdatascience.com/k-means-clustering-for-beginners-ea2256154109

1.  create dataFrame
2.  clean and prep dataFrame
3.  determine the amount of cluster you will require
4.  randomly assign center points to the clusters
5.  calculate the distance from each point to the centroid of the cluster
6.  assign each point to a cluster
7.  using the mean of all points in one cluster, determine a new centroid for that cluster
8.  repeat steps 3 - 5 until convergence.

according to what is required in this task, the above list is modified

1.  read/store data from csv file
2.  make it so that the user has access to 3 different csv files
3.  from user input determine
    * the file to access
    * number of clusters
    * number of iterations
4.  initial random cluster centroids positions
5.  determine distances from each datapoint to centroid
6.  assign each point to the closest centroid
7.  update centroid positions using mean of datapoints in a cluster
8.  display movement of centroids
8.  update centroids until number if iterations is reached

The 
	
"""


def read_csv_sourcefile(filename):
    """
    read function:
    takes in the name of the file used for test
    returns the following lists:
    list of x and y values separately
    list of countries
    x and y labels separately
    """

    x_data = []
    y_data = []
    countries_list = []
    x_data_label = ""
    y_data_label = ""

    with open(filename + '.csv') as csvfile:
        source_file = csv.reader(csvfile, delimiter=',')
        lines = 0
        for row in source_file:
            """
            ignores first row of data containing column headers
            x data from 2nd column (index 1),
            y data from 3rd column (index 2),
            country data from 1st column (index 0) of sourcefile
            """
            if lines >= 1:
                x_data.append(float(row[1]))
                y_data.append(float(row[2]))
                countries_list.append(row[0])
                lines += 1
            # extract labels from first row of sourcefile
            else:
                x_data_label = row[1]
                y_data_label = row[2]
                lines += 1
    # returning all required data
    return x_data, y_data, x_data_label, y_data_label, countries_list


def clustering(data_array, data_list_2D, num_clusters, num_iterations):
    """
    this function does the following
    1. determines centroids of each cluster
    2. assigns data point to cluster
    3. determine new centroids from sumative mean of points in cluster
    4. displays movement in scatter plot using loop and num_iterations
    5. output the following:
        *The number of countries belonging to each cluster
        *The list of countries belonging to each cluster
        *The mean Life Expectancy and Birth Rate for each cluster
    """
    
    # 1. Random cluster centroids
    sorted_data = sorted(data_array)
    centroids = random.sample(sorted_data, num_clusters)
    print(centroids)

    # The main loop
    iteration = 0
    while iteration != num_iterations:
        # 2. pairwise_distance_argmin method to assign datapoints to clusters that are closest
        labels = pairwise_distances_argmin(data_list_2D, centroids)

        # 4. displaying results
        plt.scatter(data_list_2D[:, 0], data_list_2D[:, 1], c=labels, s=50, cmap='viridis')
        plt.title('Countries by birth rate vs life expectancy'
                  '\nIteration: ' + str(iteration + 1))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        centroid_plot = np.reshape(centroids, (num_clusters, 2))
        plt.plot(centroid_plot[0:, 0], centroid_plot[0:, 1],
                 c='#000000', marker="*", markersize=15, linestyle=None, linewidth=0)
        plt.show()

        # 3. determining new centroids
        new_centroids = np.array([data_list_2D[labels == i].mean(0) for i in range(num_clusters)])
        centroids = new_centroids
        print(centroids)
        # increment until user input is matched
        iteration += 1

    """
    5. required outputs displayed after loop is broken
        1.) The number of countries belonging to each cluster
        2.) The list of countries belonging to each cluster
        3.) The mean Life Expectancy and Birth Rate for each cluster
    """

    print("\nNumber of countries in each cluster:")
    for clusters in Counter(sorted(labels)):
        print("Cluster " + str(clusters + 1) + ":\t" + str(Counter(labels)[clusters]))

    # Get cluster indices
    clusters_indices = defaultdict(list)
    for index, c in enumerate(labels):
        clusters_indices[c].append(index)

    # Print countries in each cluster and means
    cluster_num = 0
    while cluster_num < num_clusters:
        print("\nCluster " + str(cluster_num + 1))
        print("----------")
        for i in clusters_indices[cluster_num]:
            print(countries[i])
        print("----------")
        print("Mean birth rate:")
        print(math.ceil(centroids[cluster_num][0]*100)/100)
        print("Mean life expectancy:")
        print(math.ceil(centroids[cluster_num][1]*100)/100)
        cluster_num += 1

    return centroids, labels


# requesting filename from the user
source_file = input("Enter the name of the file you would like to read:\t"
                    "\n1.\tdata1953"
                    "\n2.\tdata2008"
                    "\n3.\tdataBoth"
                    "\n:")

# calling read function using user input
# storing data appropriately
x_data, y_data, x_label, y_label, countries = read_csv_sourcefile(source_file)

# data zipped in order to sample random centroid 
data_array = zip(x_data, y_data)
data_array = list(data_array)

data_list_2D = np.vstack((x_data, y_data)).T

# requesting number of clusters from user
num_clusters = int(input("How many clusters would you like to use:\t"))
# requesting number of iterations from user
num_iterations = int(input("How many iterations would you like to loop through:\t"))
centers, labels = clustering(data_array, data_list_2D, num_clusters, num_iterations)

