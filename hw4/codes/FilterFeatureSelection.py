'''
    Can BEYAZNAR
    161044038

    Filter Feature Selection icin
    Pearson's Correlation Coefficient yontemi kullanilmistir.

    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
'''

import math
import pandas as pd

# gonderilen listenin ortalamasini alir
def get_mean(input_list):
    total = 0
    list_size = len(input_list)
    for each_val in input_list:
        total += each_val
    return total / float(list_size)

# kovaryansi hesaplar
# https://en.wikipedia.org/wiki/Covariance
def get_cov(X, Y):

    if len(X) != len(Y):
        print("in function: get_cov() --> len(X) and len(Y) must be equal")
        return 0
    result = 0
    for i in range(len(X)):
        distanceX = X[i] - get_mean(X)
        distanceY = Y[i] - get_mean(Y)
        result += distanceX * distanceY
    return result

# Pearson algoritmasinda n-1'e bolme islemi yapilmaz
def get_standardDeviation(array):
    total = 0
    for num in range(len(array)):
        total += (array[num]-get_mean(array))**2
    result = math.sqrt(total)
    return result

def PCC(X, Y):
    cov = get_cov(X, Y)
    X_standarddev = (get_standardDeviation(X))
    Y_standarddev = (get_standardDeviation(Y))
    result = cov / ( X_standarddev * Y_standarddev )
    return result

'''
names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'Class']
filename = 'diabetes.csv'
data = pd.read_csv(filename, names=names)

data1 = data['a']
data2 = data['Class']

for column in data:
    if column != 'Class':
        data1 = data[column]
        print(PCC(data1, data2))
'''