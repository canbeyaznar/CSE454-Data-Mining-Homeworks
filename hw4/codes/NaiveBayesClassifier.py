'''

    CSE454 HW4 - Naive Bayes Classifier with k cross validation

    Can BEYAZNAR
    161044038


'''

import csv
from random import randrange
from random import seed
import math

PI = math.pi

# Listedeki sayilarin ortalamasini alir
def get_MeanOfList(val_list):
    total = 0
    list_size = len(val_list)
    for i in range(list_size):
        total += val_list[i]
    return total / float(list_size)


# listedeki sayilarin standart sapmasini alir
def get_StandardDeviation(val_list):
    mean = get_MeanOfList(val_list)
    size_list = len(val_list)
    total = 0
    for i in range(size_list):
        total += (val_list[i] - mean) ** 2
    standard_deviation = math.sqrt(total/float(size_list))
    return standard_deviation

# https://en.wikipedia.org/wiki/Gaussian_function
# Gaussian olasilik hesaplamasi
def Calculate_Gaussian_Probability(weight, mean, standard_deviation):
    if standard_deviation == 0.0:
        if weight == mean:
            return 1.0
        else:
            return 0.0
    exponent = math.exp(-((weight - mean) ** 2 / (2 * standard_deviation ** 2)))
    return (1 / (math.sqrt(2 * PI) * standard_deviation)) * exponent

# Split the dataset with using k_folds parameter
# k cross validation
def k_cross_validation(dataset, k_cross_splitCount):
    each_split_size = int(len(dataset) / k_cross_splitCount)
    result_split_dataset = []
    temp_dataset = list(dataset)

    # k kadar doungude git. Ve dataset'i bol (rastgele bir sekilde)
    # bolunen dataset'leri result_split_dataset listesine append et
    each_k = 0
    while each_k < k_cross_splitCount:
        each_split_dataset = []
        temp_size = 0
        while temp_size < each_split_size:
            index = randrange(len(temp_dataset))
            # random index i pop et
            val = temp_dataset.pop(index)
            # ve guncel bolunen dataset'e gonder
            each_split_dataset.append(val)
            temp_size += 1
        # bolunmus datasetleri result_split_dataset e gonder
        result_split_dataset.append(each_split_dataset)
        each_k += 1
    return result_split_dataset


# F1, precision, recall degerleri ekrana yazdirilir
# TP -> True positive
# FP -> False positive
# FN -> False negative
# TN -> True negative
def print_confusion_matrix(true_results, predicted_results):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(true_results)):
        if true_results[i] == 0 and predicted_results[i] == 0:
            TN += 1
        elif true_results[i] == 0 and predicted_results[i] == 1:
            FP += 1
        elif true_results[i] == 1 and predicted_results[i] == 0:
            FN += 1
        else:
            TP += 1

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = (2 * Precision * Recall) / (Precision + Recall)
    Accuracy = TP + TN / len(true_results)
    print("Accuracy : ",Accuracy," Precision : ", Precision, " Recall : ", Recall, " F1 : ", F1)

def print_confusion_matrix2(true_results, predicted_results):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(true_results)):
        if true_results[i] == 0 and predicted_results[i] == 0:
            TN += 1
        elif true_results[i] == 0 and predicted_results[i] == 1:
            FP += 1
        elif true_results[i] == 1 and predicted_results[i] == 0:
            FN += 1
        else:
            TP += 1
    F1 = (TP + TN) / len(true_results)
    return F1

# Kullanicidan gelen inputu predict et
def make_prediction(statistics, input_row):

    # Verilen satiri kullanarak tahmin yap ve olasiliklari hesapla
    all_probabilities = {}
    for (classValue, classModels) in statistics.items():
        all_probabilities[classValue] = 1
        for i in range(len(classModels)):
            (mean_numbers, standart_deviation, _) = classModels[i]
            result = input_row[i]
            all_probabilities[classValue] *= Calculate_Gaussian_Probability(result, mean_numbers, standart_deviation)

    # en iyi olasiligi bul ve return et
    best_label = None
    best_prediction = -1
    for class_val, probability in all_probabilities.items():
        if best_label is None or probability > best_prediction:
            best_prediction = probability
            best_label = class_val
    return best_label

# K cross validation ile dataset bolunur
# ardindan bolunmus her dataset ile naive bayes modeli egitilir
# Ve her bir egitimden sonra sonuclari ekrana yazdirir
def train_model(dataset, n_folds):
    split_datasets = k_cross_validation(dataset, n_folds)

    for each_dataset in split_datasets:
        # her dataset icin train ve test seklinde bol
        train_set = list(split_datasets)
        train_set.remove(each_dataset)
        train_set = sum(train_set, [])
        test_set = []
        for each_row in each_dataset:
            temp_row = list(each_row)
            test_set.append(temp_row)
            temp_row[-1] = None

        # egitime basla her variable icin agirliklari bul
        variable_statistics = get_statistics_of_dataset(train_set)

        # test datasetini kullanarak prediction yap
        model_predictions = []
        for each_row in test_set:
            prediction = make_prediction(variable_statistics, each_row)
            model_predictions.append(prediction)

        # prediction sonuclarini gercek verilerle karsilastir ve f1 skorunu yazdir
        true_dataset_results = [row[-1] for row in each_dataset]
        print_confusion_matrix(true_dataset_results, model_predictions)

# Dataset'i bol ve her bir satirin istatistiklerini hesapla
def get_statistics_of_dataset(dataset):
    split_dict = {}
    for i in range(len(dataset)):
        dataset_list = dataset[i]
        class_value = dataset_list[-1]
        # ayni class ta bulunan row'lari bir yerde topla
        # ve istatistiklerini ekleyebilmek icin liste olustur
        if (class_value not in split_dict):
            split_dict[class_value] = list()
        split_dict[class_value].append(dataset_list)

    # her bir veri icin istatistikleri hesapla ve result_statistics dict'e gonder
    result_statistics = {}
    for class_value, rows in split_dict.items():
        # ortalama, standart sapma ve dataset uzunlugunu al ve her bir degere bu bilgileri ekle
        each_row_statistic = []
        for each_column in zip(*rows):
            # tuple verilerinde (mean, standart deviation ve veri sayisi) her satirdaki verilerde bulunmakta
            each_row_statistic.append((get_MeanOfList(each_column),get_StandardDeviation(each_column),len(each_column)))
        del (each_row_statistic[-1])
        result_statistics[class_value] = each_row_statistic
    return result_statistics


# Reads the csv file and preprocess it
# converts variables to float
# Note : It does not convert the last variable
# The last variable will be class which we will use it in classifying
def read_csv_and_preprocess_data(filename):
    data = []
    with open(filename, 'r') as csv_file:
        FILE = csv.reader(csv_file)
        # read each line from csv file
        for each_line in FILE:
            data.append(each_line)

    for each_row in data:
        # convert variables to float
        for i in range(len(each_row) - 1):
            each_row[i] = float(each_row[i].strip())
        # convert class parameter to int
        # it must be 0 or 1
        each_row[len(each_row) - 1] = int(each_row[len(each_row) - 1])
    return data



'''
random.seed(1)
filename = 'diabetes.csv'
dataset = read_csv_and_preprocess_data(filename)

n_folds = 2
train_model(dataset, n_folds)
'''

