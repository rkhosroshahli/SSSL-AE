import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import k_means


def random_firstever_samples(y_train, IMG_PER_CLASS):
    labeled_train_data_counter = np.zeros(10)
    labeled_train_data_indices = np.ones((10, IMG_PER_CLASS))*-1
    for i, y in enumerate(y_train[50000:]):
        if labeled_train_data_counter.sum() == IMG_PER_CLASS*10:
            break
        y_arg = y.argmax()
        counter = int(labeled_train_data_counter[y_arg])
        if counter >= IMG_PER_CLASS:
            continue
        if labeled_train_data_indices[y_arg, counter] == -1:
            labeled_train_data_counter[y_arg] += 1
            labeled_train_data_indices[y_arg, counter] = i

    labeled_train_data_indices = labeled_train_data_indices.astype(int)+50000
    return labeled_train_data_indices


def random_samples(y_train, IMG_PER_CLASS):
    labeled_train_data_counter = np.zeros(10)
    labeled_train_data_indices = np.ones((10, IMG_PER_CLASS))*-1
    rand_idx = np.random.randint(0, 10000, size=1000)

    for i, idx in enumerate(rand_idx):
        y = y_train[50000+idx]

        if labeled_train_data_counter.sum() == IMG_PER_CLASS*10:
            break
        y_arg = y.argmax()
        counter = int(labeled_train_data_counter[y_arg])
        if counter >= IMG_PER_CLASS:
            continue
        if labeled_train_data_indices[y_arg, counter] == -1:
            labeled_train_data_counter[y_arg] += 1
            labeled_train_data_indices[y_arg, counter] = idx+50000

    labeled_train_data_indices = labeled_train_data_indices.astype(int)
    return labeled_train_data_indices


def find_k_closest(centroids, data):
    nns = {}
    neighbors = NearestNeighbors(n_neighbors=10).fit(data)
    # for center in centroids:
    # print((center.reshape(1,-1)))
    nns = neighbors.kneighbors(centroids, return_distance=False)
    return nns

def knearest_samples_based_on_codes(vae, x_train, ndim):

    unlabeled_images = x_train[50000:].reshape((10000, ndim, ndim))

    unlabeled_code = vae.encoder.predict(unlabeled_images, verbose=0)
    # print(np.array(unlabeled_code[-1]).shape)

    unlabeled_code = unlabeled_code[-1]

    centroids, _, _ = k_means(unlabeled_code, 10)

    nns = find_k_closest(centroids, unlabeled_code)+50000
    labeled_train_data_indices = nns
    return labeled_train_data_indices

def cluster_indices_numpy(clustNum, labels_array): #numpy 
    return np.where(labels_array == clustNum)[0]

def knearest_samples_based_on_pixels(vae, x_train, ndim):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.cluster import KMeans
    unlabeled_code = x_train[50000:].reshape((10000, ndim*ndim))
    
    #standardized_data = StandardScaler().fit_transform(unlabeled_images)
    # unlabeled_code = vae.encoder.predict(unlabeled_images, verbose=0)
    # print(np.array(unlabeled_code[-1]).shape)
    unlabeled_code = TSNE(n_components=2).fit_transform(unlabeled_code)
    
    # km = KMeans(n_clusters=10).fit(x_train[50000:].reshape((10000, ndim*ndim)))
    km = KMeans(n_clusters=10).fit(unlabeled_code)
    nns=[]
    for i in range(10):
        samples_idx=cluster_indices_numpy(i, km.labels_)
        samples=unlabeled_code[samples_idx]
        distances=euclidean_distances(samples,samples)
        sum_dists=np.sum(distances, axis=1)
        arg_sorted=np.argsort(sum_dists)
        #arg_sorted=np.argsort(distances, axis=1)
        nns=np.concatenate([nns,samples_idx[arg_sorted[:10]]])
    labeled_train_data_indices = nns.astype(int)+50000
    return labeled_train_data_indices

"""def knearest_samples_based_on_pixels(vae, x_train, ndim):
    centroids, labels, _ = k_means(x_train[50000:].reshape((10000, ndim*ndim)), 10)

    nns = find_k_closest(centroids, x_train[50000:].reshape((10000, ndim*ndim)))+0
    labeled_train_data_indices = nns +50000
    return labeled_train_data_indices"""


def init_classifier_dataset(vae, x_train, y_train, x_test, y_test, IMG_PER_CLASS, sampling_method=""):
    ndim = x_train.shape[1]
    """labeled_train_data_counter=np.zeros(10)
    labeled_train_data_indices=np.ones((10,IMG_PER_CLASS))*-1
    for i, y in enumerate(y_train[50000:]):
        if labeled_train_data_counter.sum() == IMG_PER_CLASS*10:
            break
        y_arg = y.argmax()
        counter= int(labeled_train_data_counter[y_arg])
        if counter >= IMG_PER_CLASS:
            continue
        if labeled_train_data_indices[y_arg, counter] == -1:
            labeled_train_data_counter[y_arg] +=1
            labeled_train_data_indices[y_arg, counter] = i

    labeled_train_data_indices=labeled_train_data_indices.astype(int)+50000"""

    """from sklearn.cluster import k_means
    centroids, labels, _ = k_means(x_train[:].reshape((60000, ndim*ndim)), 10)
    
    nns=find_k_closest(centroids, x_train[:].reshape((60000, ndim*ndim)))+0
    labeled_train_data_indices=nns"""

    """ndim = x_train.shape[1]
    unlabeled_images = x_train[50000:].reshape((10000, ndim, ndim))

    unlabeled_code = vae.encoder.predict(unlabeled_images, verbose=0)
    # print(np.array(unlabeled_code[-1]).shape)

    unlabeled_code = unlabeled_code[-1]

    centroids, _, _ = k_means(unlabeled_code, 10)

    nns = find_k_closest(centroids, unlabeled_code)+50000
    labeled_train_data_indices = nns"""

    """labeled_train_data_counter=np.zeros(10)
    labeled_train_data_indices=np.ones((10,IMG_PER_CLASS))*-1
    rand_idx=np.random.randint(0, 10000, size=1000)
    
    for i, idx in enumerate(rand_idx):
        y=y_train[50000+idx]
        
        if labeled_train_data_counter.sum() == IMG_PER_CLASS*10:
            break
        y_arg = y.argmax()
        counter= int(labeled_train_data_counter[y_arg])
        if counter >= IMG_PER_CLASS:
            continue
        if labeled_train_data_indices[y_arg, counter] == -1:
            labeled_train_data_counter[y_arg] +=1
            labeled_train_data_indices[y_arg, counter] = idx+50000

    labeled_train_data_indices=labeled_train_data_indices.astype(int)"""
    labeled_train_data_indices = []
    if sampling_method == "knearest_samples_based_on_codes":
        labeled_train_data_indices = knearest_samples_based_on_codes(
            vae, x_train, ndim)
    elif sampling_method == "knearest_samples_based_on_pixels":
        labeled_train_data_indices = knearest_samples_based_on_pixels(
            vae, x_train, ndim)
    elif sampling_method == "random_samples":
        labeled_train_data_indices = random_samples(y_train, IMG_PER_CLASS)
    elif sampling_method == "random_firstever_samples":
        labeled_train_data_indices = random_firstever_samples(
            y_train, IMG_PER_CLASS)
    else:
        labeled_train_data_indices=sampling_method

    print(labeled_train_data_indices)

    indices = labeled_train_data_indices.flatten('C')
    x_train_classifer = x_train[indices].copy()
    y_train_classifer = y_train[indices].copy().argmax(axis=1)

    x_test_classifer = x_test.copy()
    y_test_classifer = y_test.copy().argmax(axis=1)

    print(x_train_classifer.shape)

    return x_train_classifer, y_train_classifer, x_test_classifer, y_test_classifer, labeled_train_data_indices


def init_hists(run_classifier, K, 
               x_train_classifer, y_train_classifer,
                x_train, y_train,
                   x_test, y_test):

    hist_clf_train_classifer_f1, hist_clf_train_f1, hist_clf_test_f1 = [], [], []

    clf_train_classifer_f1, clf_train_f1, clf_test_f1 = run_classifier(K, x_train_classifer, y_train_classifer, x_train, y_train, x_test, y_test, ndim=28)
    print(f"Classifier Train F1-score {(clf_train_classifer_f1):.3e}")
    print(f"Train F1-score= {(clf_train_f1):.3e}")
    print(f"Test F1-score= {(clf_test_f1):.3e}")
    print("-------------------------------------")
    hist_clf_train_classifer_f1.append(clf_train_classifer_f1)
    hist_clf_train_f1.append(clf_train_f1)
    hist_clf_test_f1.append(clf_test_f1)

    return hist_clf_train_classifer_f1, hist_clf_train_f1, hist_clf_test_f1
