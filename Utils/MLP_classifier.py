import numpy as np
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


def run_mlp_classifier(_, x_train_classifer, y_train_classifer, x_train, y_train, x_test, y_test, ndim=28):
    clf = MLPClassifier(    
            hidden_layer_sizes=(128, ), max_iter=100, 
            solver='adam', verbose=False, random_state=1, learning_rate_init=0.001)
    
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

def ssl_mlp(_, x_train_classifer, y_train_classifer, x_reconst_imgs, ndim=28):
    clf = MLPClassifier(    
            hidden_layer_sizes=(128, ), max_iter=100,
            solver='adam', verbose=False, random_state=1, learning_rate_init=0.001)
    clf.fit(x_train_classifer.reshape(
        (len(x_train_classifer), ndim*ndim)), y_train_classifer)
    
    #y_reconst_imgs = clf.predict(x_reconst_imgs.reshape(
        #(len(x_reconst_imgs), ndim*ndim)))
    reconst_imgs_prob = clf.predict_proba(x_reconst_imgs.reshape((len(x_reconst_imgs), ndim*ndim)))
    survived_indexes, y_reconst_imgs=[], []
    for i in range(len(x_reconst_imgs)):
        if reconst_imgs_prob[i].max() >= 0.9:
            y_reconst_imgs.append(reconst_imgs_prob[i].argmax())
            survived_indexes.append(i)
    
    return np.array(survived_indexes), np.array(y_reconst_imgs)