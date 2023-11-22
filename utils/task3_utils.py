from sklearn.datasets import load_wine, load_iris, load_breast_cancer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

import pandas as pd
from mlxtend.frequent_patterns import apriori
from itertools import chain, combinations
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np 
from itertools import permutations

# from ucimlrepo import fetch_ucirepo
from sklearn.datasets import load_wine, load_iris, load_breast_cancer
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score, v_measure_score
from kmodes.kmodes import KModes

from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import v_measure_score

from ucimlrepo import fetch_ucirepo

def load_data(name):
    if name == 'breast_cancer':
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df = df.sample(frac=1)
        
        for col in data.feature_names:
            df[col] = df[col] > df[col].median()
        
        num_classes = 2
        
        return df.drop('target', axis=1), df['target'].values, num_classes
    
    if name == 'iris':
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df = df.sample(frac=1)

        for col in data.feature_names:
            df[col] = df[col] > df[col].median()

        num_classes = 3

        return df.drop('target', axis=1), df['target'].values, num_classes
    
    if name == 'wine':
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df = df.sample(frac=1)
        
        for col in data.feature_names:
            df[col] = df[col] > df[col].median()
        
        num_classes = 3
        
        return df.drop('target', axis=1), df['target'].values, num_classes

    if name == 'diabetes':
        df = pd.read_csv('./data/diabetes/diabetes.csv')
        df = df.sample(frac=1)
        target = df['Outcome']
        df = df.drop('Outcome', axis=1)
        for col in df.columns:
            df[col] = df[col] > df[col].median()

        num_classes = 2

        return df, target.values, num_classes

  
    if name == 'heart':
        df = pd.read_csv('./data/heart/heart.csv')
        df = df.sample(frac=1)
        target = df['output']
        df = df.drop('output', axis=1)
        for col in df.columns:
            df[col] = df[col] > df[col].median()
        
        num_classes = 2
        
        return df, target.values, num_classes

    if name == 'glass':
        df = pd.read_csv('./data/glass/glass.csv')
        df = df.sample(frac=1)
        target = df['type_glass']
        df = df.drop('type_glass', axis=1)
        for col in df.columns:
            df[col] = df[col] > df[col].median()
        
        num_classes = 6
        
        return df, target.values, num_classes
    

    else:
        raise ValueError("Invalid dataset name")


def get_frequent_itemsets(df, minsup):
    return apriori(df, min_support=minsup, use_colnames=True)

def get_subsets(s):
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def make_cluster_df(df, frequent_itemsets):
    new_list = set()
    for item in frequent_itemsets['itemsets']:
        if len(item) == 1:
            new_list.add(item)
        else:
            subsets = map(frozenset, get_subsets(item))
            new_list.update(subsets)
            
    new_list = list(new_list)
    new_list = [list(item) for item in new_list]

    univar_list, multivar_list = [], []

    for item in new_list:
        if len(item) == 1:
            univar_list.append(list(item)[0])
        else:
            multivar_list.append(list(item))
    
    df = df[univar_list]
    
    for item in multivar_list:
    # set it true if all the items in the itemset are true
        df[str(item)] = df[item].all(axis=1)
        
    return df.values


def do_clustering(df, algo, n_clusters):
    if algo == 'kmodes':
        kmode = KModes(n_clusters=n_clusters,
                       init='Huang', n_init=5, verbose=0)
        clusters = kmode.fit_predict(df)
        
    elif algo == 'agglomerative':
        aggloCluster = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = aggloCluster.fit_predict(df)
    
    elif algo == 'spectral':
        spectral = SpectralClustering(n_clusters=n_clusters)
        clusters = spectral.fit_predict(df)
    
    elif algo == 'gaussian':
        gauss = GaussianMixture(n_components=2, n_init=10)
        clusters = gauss.fit_predict(df)
    
    else:
        raise ValueError("Invalid algorithm name")
    
    return clusters

def evaluate_clusters(cluster, target, df=None):
    print(f"V measure Score:\t{v_measure_score(cluster, target):.4f}")
    print(f"Homogeneity Score:\t{homogeneity_score(cluster, target):.4f}")
    print(f"Completeness Score:\t{completeness_score(cluster, target):.4f}")
    print(f"Adj. Rand Score:\t{adjusted_rand_score(cluster, target):.4f}")
    print(f"Mutual Info Score:\t{mutual_info_score(cluster, target):.4f}")
    print(f"Adj. Mutual Info Score:\t{adjusted_mutual_info_score(cluster, target):.4f}")
    
    print(f"\nDavies Bouldin Score: {davies_bouldin_score(df, cluster)}")
    print(f"Silhouette Score: {silhouette_score(df, cluster)}")

def remap_labels(pred_labels, true_labels):
    pred_labels, true_labels = np.array(pred_labels), np.array(true_labels)
    assert pred_labels.ndim == 1 == true_labels.ndim
    assert len(pred_labels) == len(true_labels)
    cluster_names = np.unique(pred_labels)
    accuracy = 0
    perms = np.array(list(permutations(np.unique(true_labels))))
    remapped_labels = true_labels
    for perm in perms:
        flipped_labels = np.zeros(len(true_labels))
        for label_index, label in enumerate(cluster_names):
            flipped_labels[pred_labels == label] = perm[label_index]
        testAcc = np.sum(flipped_labels == true_labels) / len(true_labels)
        if testAcc > accuracy:
            accuracy = testAcc
            remapped_labels = flipped_labels
    return accuracy, remapped_labels

    
def plot_clusters(df_new, clusters, target):
    
    # Reduce dimensionality for visualization (adjust n_components as needed)
    pca = PCA(n_components=2)
    trans_data_pca = pca.fit_transform(df_new)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(trans_data_pca[:, 0],  trans_data_pca[:, 1],c=clusters, cmap='viridis')
    ax[0].set_title('Predicted clusters')
    ax[1].scatter(trans_data_pca[:, 0],  trans_data_pca[:, 1], c=target, cmap='viridis')
    ax[1].set_title('Actual clusters')
    plt.show()
    
def task3(df, target, num_classes, algo):
    print('Running Task 3 for ' + algo)
    clusters = do_clustering(df, algo, n_clusters=num_classes)
    acc, remapped_labels = remap_labels(clusters, target)
    print(f"Accuracy: {acc}")
    evaluate_clusters(remapped_labels, target, df)
    plot_clusters(df, remapped_labels, target)