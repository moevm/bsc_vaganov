# This Python file uses the following encoding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.cluster import KMeans, Birch, MiniBatchKMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from itertools import cycle
import matplotlib.colors as colors
from sklearn.ensemble import RandomForestClassifier
import pickle
# TODO
# 1. Метод генерации и отображения сравнительной таблицы моделей по метрикам


class ModelHandler:

    def __init__(self):
        self.rawdata = pd.read_json('bsc_vaganov/data_collect_and_preprocess/rawdata.json')
        self.freqdata = pd.read_json('bsc_vaganov/data_collect_and_preprocess/freqdata.json')
        self.data = pd.merge(self.rawdata, self.freqdata, on="paperPath")
        self.X = self.data.drop(['paperPath', 'paperUrl', 'paperTitle', 'journalName', 'journalViews', 'journalDownloads', 'journalHirch', 'isGood'], axis=1)
        self.y = self.data.isGood
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def decisionTreeTest(self, max_depth=9, min_samples_leaf=4, min_samples_split=4):
        clf = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
        #parameters = {'criterion': ['gini', 'entropy'],
        #              'max_depth': range(2, 25),
        #              'min_samples_leaf': range(100, 120),
        #              'min_samples_split': range(100, 120)}
        #grid_clf = GridSearchCV(clf, parameters, cv=5)
        #grid_clf.fit(X_train, y_train)
        #best_clf = grid_clf.best_estimator_
        #print(grid_clf.best_params_)
        clf.fit(self.X_train, self.y_train)
        print(clf.score(self.X_train, self.y_train))
        print(clf.score(self.X_test, self.y_test))
        tree.plot_tree(clf, fontsize=5, feature_names=list(self.X), filled=True)
        plt.show()
        crossval = cross_val_score(clf, self.X_test, self.y_test, cv=5).mean()
        print(crossval)

    def decisionTreeCreate(self, max_depth=9, min_samples_leaf=4, min_samples_split=4):
        clf = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
        clf.fit(self.X, self.y)
        pickle.dump(clf, open('bsc_vaganov/model/model.pkl', 'wb'))

    def randomForestTest(self, n_jobs=2, criterion='entropy', n_estimators=10, max_depth=14):
        clf = RandomForestClassifier(n_jobs=n_jobs, criterion=criterion, n_estimators=n_estimators, max_depth=max_depth)
        #parameters = {'n_estimators': range(99, 110),
        #              'max_depth': range(2, 10),
        #              'criterion': ['gini', 'entropy'],
        #              'n_jobs': range(1, 4)}
        #grid_clf = GridSearchCV(clf, parameters, cv=5)
        #grid_clf.fit(X_train, y_train)    #best_clf = grid_clf.best_estimator_
        #print(grid_clf.best_params_)
        #print(best_clf.score(X_train, y_train))
        #print(best_clf.score(X_test, y_test))

        clf.fit(self.X_train, self.y_train)
        print(clf.score(self.X_train, self.y_train))
        print(clf.score(self.X_test, self.y_test))
        crossval = cross_val_score(clf, self.X_test, self.y_test, cv=5).mean()
        print(crossval)
        print(str(clf.n_estimators) + " " + str(clf.max_depth))
        importances = clf.feature_importances_
        print(importances)
        #tree.plot_tree(clf, fontsize=5, feature_names=list(self.X), filled=True)
        #plt.show()

    def KMeansModel(self):
        model = KMeans(n_clusters=2)
        model.fit(self.X)
        all_prediction = model.predict(self.X)
        unique, counts = np.unique(all_prediction, return_counts=True)
        print(dict(zip(unique, counts)))
        for i in range(1, self.X.shape[0]):
            if model.predict(self.X)[i] == 1:
                print(i)

    def linkageModel(self):
        mergings = linkage(self.X, method='complete')
        dendrogram(mergings)
        plt.show()

    def plotDendrogram(self, model, **kwargs):
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

    def agglomerativeClusteringModel(self):
        model2 = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
        model2 = model2.fit(self.X)
        #print(model2.predict(X))
        plt.title('Hierarchical Clustering Dendrogram')
        # plot the top three levels of the dendrogram
        self.plotDendrogram(model2, truncate_mode='level', p=3)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()

    def TSNETest(self):
        # Метод визуализации путем понижения размерности
        model3 = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000, learning_rate=200)
        transformed = model3.fit_transform(self.X)
        x_axis = transformed[:, 0]
        y_axis = transformed[:, 1]
        plt.scatter(x_axis, y_axis)
        plt.show()

    def dbscanModel(self):
        dbscan = DBSCAN()
        dbscan.fit(self.X)
        pca = PCA(n_components=2).fit(self.X)
        pca_2d = pca.transform(self.X)
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

    def birchModel(self):
        birch_model = Birch()
        birch_model.fit(self.X)
        # Plot result
        labels = birch_model.labels_
        centroids = birch_model.subcluster_centers_
        n_clusters = np.unique(labels).size
        print("n_clusters : %d" % n_clusters)
        print(birch_model.predict(self.X))
        for i in range(1, self.X.shape[0]):
            if birch_model.predict(self.X)[i] == 1:
                print(i)


#KMeansModel()
#linkageModel()
#agglomerativeClusteringModel()
#TSNETest()
#birchModel()
#decisiontree()
#model_predict()
#randomforest()