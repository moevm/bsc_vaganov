import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import precision_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

#data = pd.read_csv('D:/Users/nikva/PycharmProjects/ml/train_data_tree.csv')
#X = data.drop(['num'], axis=1)
#y = data.num
#clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=1)
#clf.fit(X,y)

#print(clf.tree_.n_node_samples)
#print(clf.tree_.impurity)
#tree.plot_tree(clf, fontsize=10, feature_names=list(X), filled=True)
#plt.show()

rawdata = pd.read_json('/home/woghan/Desktop/ml/leninka_scrapper/rawdata.json')
freqdata = pd.read_json('/home/woghan/Desktop/ml/leninka_scrapper/freqdata.json')
data = pd.merge(rawdata, freqdata, on="paperPath")
X = data.drop(
    ['paperPath', 'paperUrl', 'paperTitle', 'journalName', 'journalViews'], axis=1)


def KMeansModel():
    model = KMeans(n_clusters=2)
    model.fit(X)
    all_prediction = model.predict(X)
    unique, counts = np.unique(all_prediction, return_counts=True)
    print(dict(zip(unique, counts)))


def linkageModel():
    mergings = linkage(X, method='complete')
    dendrogram(mergings)
    plt.show()


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def agglomerativeClusteringModel():
    model2 = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model2 = model2.fit(X)
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model2, truncate_mode='level', p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()


def TSNETest():
    # Метод визуализации путем понижения размерности
    model3 = TSNE(learning_rate=100, method='exact')
    transformed = model3.fit_transform(X)
    x_axis = transformed[:, 0]
    y_axis = transformed[:, 1]

    plt.scatter(x_axis, y_axis)
    plt.show()


def dbscanModel():
    dbscan = DBSCAN()
    dbscan.fit(X)
    pca = PCA(n_components=2).fit(X)
    pca_2d = pca.transform(X)
    for i in range(0, pca_2d.shape[0]):
        if dbscan.labels_[i] == 0:
            c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
        elif dbscan.labels_[i] == 1:
            c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
        elif dbscan.labels_[i] == -1:
            c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')
    plt.legend([c1, c2, c3], ['Кластер 1', 'Кластер 2', 'Шум'])
    plt.title('DBSCAN нашел 2 кластера и шум')
    plt.show()


TSNETest()
