import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def run_knn_classifier(K, x_train_classifer, y_train_classifer, x_train, y_train, x_test, y_test, ndim=28):

    y_pred = predict_classification(x_train_classifer.reshape(
        (len(x_train_classifer), ndim*ndim)), x_train_classifer.reshape(
        (len(x_train_classifer), ndim*ndim)), 11)
    clf_train_classifer_f1 = f1_score(y_train_classifer, y_pred, average='macro')
    
    y_pred = predict_classification(x_train_classifer.reshape(
        (len(x_train_classifer), ndim*ndim)), x_train.reshape(
        (len(x_train), ndim*ndim)), 11)
    clf_train_f1 = f1_score(y_train, y_pred, average='macro')
    
    y_pred = predict_classification(x_train_classifer.reshape(
        (len(x_train_classifer), ndim*ndim)), x_test.reshape(
        (len(x_test), ndim*ndim)), 11)
    clf_test_f1 = f1_score(y_test, y_pred, average='macro')

    return clf_train_classifer_f1, clf_train_f1, clf_test_f1

def ssl_knn(K, x_train_classifer, y_train_classifer, x_reconst_imgs, ndim=28):
    y_reconst_imgs = predict_classification(x_train_classifer.reshape(
        (len(x_train_classifer), ndim*ndim)), x_reconst_imgs.reshape(
        (len(x_reconst_imgs), ndim*ndim)), 11)
    return y_reconst_imgs

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)
 
# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors
 
# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction
 