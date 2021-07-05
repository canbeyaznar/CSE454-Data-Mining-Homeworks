# Can BEYAZNAR
# 161044038
# Algorithm: https://www.datanovia.com/en/lessons/agglomerative-hierarchical-clustering
import pandas as pd
import matplotlib.pyplot as plt

class Point:
    def __init__(self, x=0, y=0, id=0):
        self.x, self.y, self.id = x, y, id

def get_distance(pointA, pointB):
    X_dist = (pointA.x - pointB.x) * (pointA.x - pointB.x)
    Y_dist = (pointA.y - pointB.y) * (pointA.y - pointB.y)
    result = X_dist + Y_dist
    return result

def get_points_distance_list(points):
    distances_list = []
    len_points = len(points)
    for i in range(0, len_points):
        for j in range(i + 1, len_points):
            temp = [i,j, get_distance(points[i], points[j])]
            distances_list.append(temp)
    distances_list = sorted(distances_list, key=lambda dist: dist[2], reverse=False)
    return distances_list

def Agglomerative_Hierarchical_Clustering(points, distanceMap, mergeRatio, ClusterCenter_Index):
    UnsortedClusters = {index: 1 for index in range(len(points))}

    for key in distanceMap:
        low_index = key[0]
        high_index = key[1]

        # if cluster ids do not match:
        if points[low_index].id != points[high_index].id:
            lowindex_ClusterID = points[low_index].id
            highindex_ClusterID = points[high_index].id
            UnsortedClusters[lowindex_ClusterID] += UnsortedClusters[highindex_ClusterID]
            del UnsortedClusters[highindex_ClusterID]

            for each_point in points:
                if each_point.id == highindex_ClusterID:
                    each_point.id = lowindex_ClusterID
        if len(UnsortedClusters) <= int(len(points) * mergeRatio):
            break

    SortedCluster = sorted(UnsortedClusters.items(), key=lambda group: group[1], reverse=True)
    ClusterCenterCount = 0
    print(SortedCluster, len(SortedCluster))
    for key, each_dist in SortedCluster:
        ClusterCenterCount += 1
        for each_point in points:
            if each_point.id == key:
                each_point.id = -1 * ClusterCenterCount
        if ClusterCenterCount >= ClusterCenter_Index:
            break
    return points

# converts all the data points as point class
def load_data(Data):
    points = []
    for i in range(0, len(Data)):
        points.append(Point(x=Data[i][0], y=Data[i][1], id=i))
    return points

def draw(points):
    colors = ['blue', 'green', 'yellow', 'red', 'purple', 'gray', 'orange', 'brown', 'pink', 'cyan']
    plt.title('Agglomerative Hierarchical Clustering')
    max = 0
    for point in points:
        color = ''
        if (point.id < 0):
            color = colors[-1 * point.id - 1]
        else:
            color = 'black'
        plt.scatter(point.x, point.y, color=color, marker='o')
    #plt.legend()
    plt.show()
'''
def main():
    clusterCenterNumber = 10
    Merge_Ratio = 0.3

    Data = pd.read_csv('world_happiness_report/2015.csv')
    X=3
    Y=7
    datas = Data.iloc[:, [X, Y]].values

    datas_to_points = load_data(datas)
    distanceMap = get_points_distance_list(datas_to_points)

    result = Agglomerative_Hierarchical_Clustering(datas_to_points, distanceMap, Merge_Ratio, clusterCenterNumber)
    draw(result)
main()
'''