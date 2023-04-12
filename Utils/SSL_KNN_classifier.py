import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def run_classifier(K, x_train_classifer, y_train_classifer, x_train, y_train, x_test, y_test, ndim=28):
    clf = KNeighborsClassifier(K)
    clf.fit(x_train_classifer.reshape(
        (len(x_train_classifer), ndim*ndim)), y_train_classifer)

    y_pred = clf.predict(x_train_classifer.reshape(
        (len(x_train_classifer), ndim*ndim)))
    clf_train_classifer_f1 = f1_score(y_train_classifer, y_pred, average='macro')
    
    y_pred = clf.predict(x_train.reshape(
        (len(x_train), ndim*ndim)))
    clf_train_f1 = f1_score(y_train, y_pred, average='macro')
    
    y_pred = clf.predict(x_test.reshape(
        (len(x_test), ndim*ndim)))
    clf_test_f1 = f1_score(y_test, y_pred, average='macro')

    return clf_train_classifer_f1, clf_train_f1, clf_test_f1

def ssl_knn(K, x_train_classifer, x_reconst_imgs, ndim=28):
    clf = KNeighborsClassifier(K)
    clf.fit(x_train_classifer.reshape(
        (len(x_train_classifer), ndim*ndim)), y_train_classifer)
    y_reconst_imgs = clf.predict(x_reconst_imgs.reshape(
        (len(x_reconst_imgs), ndim*ndim)))
    return y_reconst_imgs
