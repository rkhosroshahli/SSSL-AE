import numpy as np
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


def run_rf_classifier(_, x_train_classifer, y_train_classifer, x_train, y_train, x_test, y_test, ndim=28):
    clf = RandomForestClassifier()
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

def ssl_rf(_, x_train_classifer, y_train_classifer, x_reconst_imgs, ndim=28):
    x_train_classifer=x_train_classifer.reshape((len(x_train_classifer), ndim*ndim))
    x_reconst_imgs=x_reconst_imgs.reshape((len(x_reconst_imgs), ndim*ndim))
    
    clf = RandomForestClassifier()
    clf.fit(x_train_classifer, y_train_classifer)
    
    #y_reconst_imgs = clf.predict(x_reconst_imgs.reshape(
        #(len(x_reconst_imgs), ndim*ndim)))
    reconst_imgs_prob = clf.predict_proba(x_reconst_imgs)
    survived_indexes, y_reconst_imgs=[], []
    for i in range(len(x_reconst_imgs)):
        if reconst_imgs_prob[i].max() >= 0.7:
            y_reconst_imgs.append(reconst_imgs_prob[i].argmax())
            survived_indexes.append(i)
    
    return np.array(survived_indexes), np.array(y_reconst_imgs)