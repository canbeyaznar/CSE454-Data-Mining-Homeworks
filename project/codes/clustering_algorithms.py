
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import dbscan
import MeanShift_Clustering
import agglomerative_hierarchical_clustering

def test_dbscan(Data, eps, min_pts, X, Y):
    temp = Data.iloc[:, [X, Y]].values

    DataTemp = temp
    temp = StandardScaler().fit_transform(temp)
    my_labels = dbscan.dbscan(temp, Epsilon=eps, Min_points=min_pts)

    print("Number of clusters : ", max(my_labels))
    dbscan.draw(my_labels, DataTemp)

def test_MeanShift(Data, X, Y, X_name='X', Y_name='Y', kernel_bandwidth=1):
    temp_data = Data.iloc[:, [X, Y]].values
    mean_shifter = MeanShift_Clustering.MeanShift_Class()
    mean_shift_result = mean_shifter.cluster_points(temp_data, kernel_bandwidth=kernel_bandwidth)

    data_original_points = mean_shift_result.original_points
    clusters_id_list = mean_shift_result.cluster_ids

    x = data_original_points[:, 0]
    y = data_original_points[:, 1]
    Clusters = clusters_id_list



    fig = plt.figure()
    #plt.title('Mean Shift Clustering')
    ax = fig.add_subplot(111)
    scatter = ax.scatter(x, y, c=Clusters, s=25)
    ax.set_xlabel(X_name)
    ax.set_ylabel(Y_name)
    plt.colorbar(scatter)
    plt.show()

def test_AGG_HC(Data,X,Y,Merge_Ratio,clusterCenterNumber):
    datas = Data.iloc[:, [X, Y]].values
    datas_to_points = agglomerative_hierarchical_clustering.load_data(datas)
    distanceMap = agglomerative_hierarchical_clustering.get_points_distance_list(datas_to_points)

    result = agglomerative_hierarchical_clustering.Agglomerative_Hierarchical_Clustering(datas_to_points, distanceMap, Merge_Ratio, clusterCenterNumber)
    agglomerative_hierarchical_clustering.draw(result)

def main():
    Data = pd.read_csv('world_happiness_report/2015.csv')
    Happiness_Score_index = 3
    Health_index = 7

    kernel_bandwidth=1

    clusterCenterNumber = 10
    Merge_Ratio = 0.3


    test_dbscan(Data,0.2,6,Happiness_Score_index,Health_index)
    test_MeanShift(Data, Happiness_Score_index, Health_index, X_name='Happiness Score', Y_name='Health', kernel_bandwidth=kernel_bandwidth)
    test_AGG_HC(Data, Happiness_Score_index, Health_index, Merge_Ratio, clusterCenterNumber)


if __name__ == "__main__":
    main()