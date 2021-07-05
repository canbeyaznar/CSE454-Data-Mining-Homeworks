"""
    Can BEYAZNAR
    161044038

"""
from sklearn.decomposition import PCA
import NaiveBayesClassifier as NBC



def train_model_PCA(dataset, n_folds):

    split_datasets = NBC.k_cross_validation(dataset, n_folds)
    prediction_scores = list()
    for each_dataset in split_datasets:
        train_set = list(split_datasets)
        train_set.remove(each_dataset)
        train_set = sum(train_set, [])
        test_set = list()
        for row in each_dataset:
            row_copy = list(row)
            test_set.append(row_copy)


        pca = PCA(n_components=n_folds)
        test_pca = pca.fit_transform(test_set)
        train_pca = pca.fit_transform(train_set)


        test_pca[-1] = None

        for i in range(len(test_pca)):
            test_pca[i][8] = test_set[i][8]

        for i in range(len(train_pca)):
            train_pca[i][8] = train_set[i][8]

        variable_statistics = NBC.get_statistics_of_dataset(train_pca)

        # test datasetini kullanarak prediction yap
        model_predictions = []
        for each_row in test_pca:
            prediction = NBC.make_prediction(variable_statistics, each_row)
            model_predictions.append(prediction)

        # prediction sonuclarini gercek verilerle karsilastir ve f1 skorunu yazdir
        true_dataset_results = [row[-1] for row in each_dataset]
        prediction_scores.append(NBC.print_confusion_matrix2(true_dataset_results, model_predictions))

    return prediction_scores

"""NBC.seed(1)
filename = 'diabetes.csv'
dataset = NBC.read_csv_and_preprocess_data(filename)

n_folds = 9
scores = train_model_PCA(dataset, n_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))"""
