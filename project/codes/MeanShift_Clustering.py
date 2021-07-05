# Can BEYAZNAR
# 161044038

# algorithm : https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68

import math
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler

# Global variables
MIN_DISTANCE = 0.98
CLUSTER_DISTANCE_CONDITION = 0.01

def get_euclidean_distance(X1, X2):
    if(len(X1) != len(X2)):
        print("(function name : euclidean_dist) length of X1 and X2 are not matching!!")
        return
    total = 0.0
    len_input = len(X1)
    for i in range(0, len_input):
        total += (X1[i] - X2[i])**2
    result = math.sqrt(total)
    return result

def gaussian_formula(distance, bandwidth):
    euclidean_distance = np.sqrt(((distance)**2).sum(axis=1))
    result = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((euclidean_distance / bandwidth))**2)
    return result

# cluster all the points 
# look each point look their neighbours and create clusters 
def process_cluster_points(all_points):
    Clusters_list = []
    clusters = []
    cluster_id = 0
    # look each point
    for each_point in all_points:
        # get nearest cluster id
        nearest_cluster_id = get_nearest_cluster_id(each_point, clusters)
        if nearest_cluster_id == -1: # if there is no neighboring cluster
            # create new cluster
            clusters.append([each_point])
            Clusters_list.append(cluster_id)
            cluster_id += 1
        else:
            # add it to cluster
            Clusters_list.append(nearest_cluster_id)
            clusters[nearest_cluster_id].append(each_point)
    return np.array(Clusters_list)


def get_nearest_cluster_id(point, clusters):
    nearest_group_index = -1
    index = 0
    for each_cluster in clusters:
        distance_to_group = get_distance_to_cluster(point, each_cluster)
        if distance_to_group < CLUSTER_DISTANCE_CONDITION:
            nearest_group_index = index
        index += 1
    return nearest_group_index

# get minimum distance between points
def get_distance_to_cluster(point, group):
    min_distance = sys.float_info.max
    for each_grouppoint in group:
        distance = get_euclidean_distance(point, each_grouppoint)
        if distance < min_distance:
            min_distance = distance
    return min_distance

class MeanShift_Class(object):

    def cluster_points(self, points, kernel_bandwidth):

        shifted_points_result = np.array(points)
        boundary_distance = 1
        iteration_number = 0

        is_shifting_left = [True] * points.shape[0]
        while boundary_distance > MIN_DISTANCE:
            boundary_distance = 0
            iteration_number += 1
            for i in range(0, len(shifted_points_result)):
                if not is_shifting_left[i]:
                    continue
                current_P = shifted_points_result[i]
                Start_P = current_P
                current_P = self.Get_Shifted_Points(current_P, points, kernel_bandwidth)
                current_distance = get_euclidean_distance(current_P, Start_P)
                if current_distance > boundary_distance:
                    boundary_distance = current_distance
                if current_distance < MIN_DISTANCE:
                    is_shifting_left[i] = False
                shifted_points_result[i] = current_P
        Clusters_List = process_cluster_points(shifted_points_result.tolist())
        return MeanShiftResult(points, shifted_points_result, Clusters_List)

    def Get_Shifted_Points(self, point, points, kernel_bandwidth):
        points = np.array(points)
        gaussian_weights = gaussian_formula(point - points, kernel_bandwidth)
        tiled_weights = np.tile(gaussian_weights, [len(point), 1])

        # denominator
        Sum_weights = sum(gaussian_weights)
        shifted_points = np.multiply(tiled_weights.transpose(), points).sum(axis=0) / Sum_weights
        return shifted_points

def normalize_data(data):
    normalised = StandardScaler()
    temp = normalised.fit_transform(data)
    return(temp)

class MeanShiftResult:
    def __init__(self, original_points, shifted_points, cluster_ids):
        self.original_points = original_points
        self.shifted_points = shifted_points
        self.cluster_ids = cluster_ids
