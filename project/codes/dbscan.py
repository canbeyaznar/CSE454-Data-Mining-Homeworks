import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math

def dbscan(Datas, Epsilon, Min_points):
    labels_result = [0] * len(Datas)    # mark all objects as unvisited;
    Cluster = 0
    for p in range(0, len(Datas)):
        if (labels_result[p] != 0):
            continue
        # Find all of P's neighboring points.
        Neighbour_Points = GetRegionQuery(Datas, p, Epsilon)
        if len(Neighbour_Points) < Min_points:  # if the ε-neighborhood of p has at least MinPts objects
            labels_result[p] = -1               # make noise
        else:
            Cluster += 1
            # create a new cluster C, and add p to C;
            MakeCluster(Datas, labels_result, p, Neighbour_Points, Cluster, Epsilon, Min_points)

    print(labels_result)
    return labels_result


def MakeCluster(Datas, labels_result, p, Neighbour_Points, Cluster, Epsilon, Min_points):
    labels_result[p] = Cluster
    index = 0
    while index < len(Neighbour_Points):
        each_p = Neighbour_Points[index]
        if labels_result[each_p] == -1: # if it is noise add to cluster
            labels_result[each_p] = Cluster
        # if it is unvisited call GetRegionQuery and add it's cluster to itself
        elif labels_result[each_p] == 0:
            labels_result[each_p] = Cluster
            Neighbour_Points_for_p = GetRegionQuery(Datas, each_p, Epsilon)
            if (len(Neighbour_Points_for_p) >= Min_points):
                Neighbour_Points = Neighbour_Points + Neighbour_Points_for_p # add two clusters
        index += 1

def GetRegionQuery(Datas, p, Epsilon):
    Neighbours_Arr = []
    for each_p in range(0, len(Datas)):
        # Calculate distance
        # if it is less than Epsilon
        if get_distance(Datas[p], Datas[each_p]) < Epsilon:
            Neighbours_Arr.append(each_p)
    return Neighbours_Arr

def get_distance(z1, z2):
    X = (z1[0] - z2[0]) ** 2
    Y = (z1[1] - z2[1]) ** 2
    return (math.sqrt(X + Y))

def draw(Cluster_Arr, values):
    colors = ['blue', 'green','yellow', 'red', 'purple','gray','orange', 'brown', 'pink', 'cyan']
    num_cluster = len(Cluster_Arr)
    plt.title('(DBSCAN) Number of clusters : %d'% max(Cluster_Arr))
    for i in range(num_cluster):
        if(Cluster_Arr[i] == -1):
            plt.scatter(values[i][0], values[i][1], color='black', marker='o')
        else:
            plt.scatter(values[i][0], values[i][1], color=colors[(Cluster_Arr[i] + 1) % len(colors)], marker='o')
    #plt.legend()
    plt.show()

