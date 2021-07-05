"""
    Can BEYAZNAR
    161044038

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import NaiveBayesClassifier as NBC

def train_model_LDA(dataset, n_folds):

    split_datasets = NBC.k_cross_validation(dataset, n_folds)
    prediction_scores = []
    for each_dataset in split_datasets:
        train_set = list(split_datasets)
        train_set.remove(each_dataset)
        train_set = sum(train_set, [])
        test_set = list()
        for row in each_dataset:
            row_copy = list(row)
            test_set.append(row_copy)

        names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'Class']
        db = pd.read_csv('diabetes.csv', names=names)

        X = db.drop('Class', 1)
        y = db['Class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        lda = LDA(n_components=1)

        test_lda = lda.fit_transform(X_test,y_test)
        train_lda = lda.fit_transform(X_train, y_train)


        for i in range(len(test_set)):
            test_lda[i][0] = test_set[i][8]

        for i in range(len(test_set)):
            train_lda[i][0] = train_set[i][8]

        # egitime basla her variable icin agirliklari bul
        variable_statistics = NBC.get_statistics_of_dataset(train_lda)

        # test datasetini kullanarak prediction yap
        model_predictions = []
        for each_row in test_lda:
            prediction = NBC.make_prediction(variable_statistics, each_row)
            model_predictions.append(prediction)

        # prediction sonuclarini gercek verilerle karsilastir ve f1 skorunu yazdir
        true_dataset_results = [row[-1] for row in each_dataset]
        prediction_scores.append(NBC.print_confusion_matrix2(true_dataset_results, model_predictions))

    return prediction_scores


"""NBC.seed(1)
filename = 'diabetes.csv'
dataset = NBC.load_csv_dataset(filename)
for i in range(len(dataset[0]) - 1):
    NBC.str_column_to_float(dataset, i)
# convert class column to integers
NBC.str_column_to_int(dataset, len(dataset[0]) - 1)
# evaluate algorithm
n_folds = 9
scores = NBC.evaluate_algorith_3(dataset, NBC.naive_bayes, n_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))"""


"""NBC.seed(1)
filename = 'diabetes.csv'
dataset = NBC.read_csv_and_preprocess_data(filename)

# evaluate algorithm
n_folds = 9
scores = train_model_LDA(dataset, n_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))"""