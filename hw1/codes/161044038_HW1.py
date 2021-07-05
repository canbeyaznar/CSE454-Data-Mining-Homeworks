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
    #return np.linalg.norm(z1 - z2)

def draw(Cluster_Arr, values):
    colors = ['blue', 'green','yellow', 'red', 'purple','gray','orange', 'brown', 'pink', 'cyan']
    num_cluster = len(Cluster_Arr)
    plt.title('Number of clusters : %d'% max(Cluster_Arr))
    for i in range(num_cluster):
        if(Cluster_Arr[i] == -1):
            plt.scatter(values[i][0], values[i][1], color='black', marker='o')
        else:
            plt.scatter(values[i][0], values[i][1], color=colors[(Cluster_Arr[i] + 1) % len(colors)], marker='o')
    plt.legend()
    plt.show()

def main():
    # Search growth for Data Science terms
    # https://www.kaggle.com/leonardopena/search-growth-for-data-science-terms
    Data = pd.read_csv('data.csv')
    X = Data.iloc[:, [7, 8]].values
    DataTemp = X
    X = StandardScaler().fit_transform(X)
    my_labels = dbscan(X, Epsilon=0.2, Min_points=4)
    print("Number of clusters : ", max(my_labels))
    draw(my_labels,DataTemp)

if __name__ == "__main__":
    main()

'''
(1) mark all objects as unvisited;
(2) do
(3)     randomly select an unvisited object p;
(4)     mark p as visited;
(5)     if the ε-neighborhood of p has at least MinPts objects
(6)         create a new cluster C, and add p to C;
(7)         let N be the set of objects in the ε-neighborhood of p;
(8)         for each point p' in N
(9)             if p' is unvisited
(10)                mark p' as visited;
(11)                if the ε-neighborhood of p0 has at least MinPts points,
                    add those points to N;
(12)            if p0 is not yet a member of any cluster, add p0 to C;
(13)        end for
(14)        output C;
(15)    else mark p as noise;
(16) until no object is unvisited;
'''