"""
    Can BEYAZNAR
    161044038

"""

import NaiveBayesClassifier as NBC
import FilterFeatureSelection as FFS
import LDA
import PCA
import pandas as pd

filename = 'diabetes.csv'

def Part1():
    print("Naive-Bayes test")
    NBC.seed(1)
    dataset = NBC.read_csv_and_preprocess_data(filename)
    n_folds = 2
    NBC.train_model(dataset, n_folds)

def Part2():
    print("\nFilter test")
    # Part 2 test
    NBC.seed(1)
    names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'Class']

    data = pd.read_csv(filename, names=names)

    data2 = data['Class']
    correlation_results = []

    for column in data:
        if column != 'Class':
            data1 = data[column]
            correlation_results.append(FFS.PCC(data1, data2))
    dataset = NBC.read_csv_and_preprocess_data(filename)
    print("Correlation results : ")
    print(correlation_results,"\n\n")
    threshold = 0.2

    for each_row in range(len(dataset)):
        deleted_labels_count = 0
        for each_correlation in range(len(correlation_results)):
            if correlation_results[each_correlation] < threshold:
                dataset[each_row].remove(dataset[each_row][each_correlation - deleted_labels_count])
                deleted_labels_count += 1

    n_folds = 2
    NBC.train_model(dataset, n_folds)

def Part3():
    # Part 3 test
    print("\nWrapper test")
    NBC.seed(1)
    names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'Class']

    data = pd.read_csv(filename, names=names)

    data2 = data['Class']
    correlation_results = []

    for column in data:
        if column != 'Class':
            data1 = data[column]
            correlation_results.append(FFS.PCC(data1, data2))
    dataset = NBC.read_csv_and_preprocess_data(filename)

    threshold = 0.15

    for each_row in range(len(dataset)):
        deleted_labels_count = 0
        for each_correlation in range(len(correlation_results)):
            if correlation_results[each_correlation] < threshold:
                dataset[each_row].remove(dataset[each_row][each_correlation - deleted_labels_count])
                deleted_labels_count += 1

    n_folds = 2
    NBC.train_model(dataset, n_folds)

def Part4():
    NBC.seed(1)
    filename = 'diabetes.csv'
    dataset = NBC.read_csv_and_preprocess_data(filename)

    n_folds = 9
    scores = PCA.train_model_PCA(dataset, n_folds)
    print("\nPCA Test")
    print('F1 Scores: %s' % scores)
    print('Mean F1: %.3f%%' % ((sum(scores) / float(len(scores)))*100))

def Part5():
    NBC.seed(1)
    filename = 'diabetes.csv'
    dataset = NBC.read_csv_and_preprocess_data(filename)

    # evaluate algorithm
    n_folds = 9
    scores = LDA.train_model_LDA(dataset, n_folds)
    print("\nLDA Test")
    print('F1 Scores: %s' % scores)
    print('Mean F1: %.3f%%' % ((sum(scores) / float(len(scores)))*100))

def test():
    # Part 1 test
    Part1()

    # Part 2 test
    Part2()

    # Part 3 test
    Part3()

    # Part 4 test
    Part4()

    # Part 5 test
    Part5()



if __name__ == '__main__':
    test()

