import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


def run_tknn_classifier(K, x_train_classifer, y_train_classifer, x_train, y_train, x_test, y_test, ndim=28):
    clf = KNeighborsClassifier(K)
    x_train_classifer=x_train_classifer.reshape((len(x_train_classifer), ndim*ndim))
    x_train=x_train.reshape((len(x_train), ndim*ndim))
    x_test=x_test.reshape((len(x_test), ndim*ndim))
    
    clf.fit(x_train_classifer, y_train_classifer)

    y_pred = clf.predict(x_train_classifer)
    clf_train_classifer_f1 = f1_score(y_train_classifer, y_pred, average='macro')
    
    y_pred = clf.predict(x_train)
    clf_train_f1 = f1_score(y_train, y_pred, average='macro')
    
    y_pred = clf.predict(x_test)
    clf_test_f1 = f1_score(y_test, y_pred, average='macro')

    return clf_train_classifer_f1, clf_train_f1, clf_test_f1

def ssl_tknn(K, x_train_classifer, y_train_classifer, x_reconst_imgs, ndim=28):
    x_train_classifer=x_train_classifer.reshape((len(x_train_classifer), ndim*ndim))
    x_reconst_imgs=x_reconst_imgs.reshape((len(x_reconst_imgs), ndim*ndim))
    
    dists=pairwise_distances(x_reconst_imgs, x_train_classifer)
    print(dists.shape)
    survived_indexes, y_reconst_imgs=[], []
    for i in range(len(x_reconst_imgs)):
        sorted_idx=np.argsort(dists[i])[:K]
        y_prob=np.zeros(10)
        print(y_train_classifer[sorted_idx])
        for j in range(10):
            
            y_prob[j]=np.count_nonzero(y_train_classifer[sorted_idx]==j)
        if y_prob.max() >= 9:
            survived_indexes.append(i)
            y_reconst_imgs.append(y_prob.argmax())
    
    return np.array(survived_indexes), np.array(y_reconst_imgs)