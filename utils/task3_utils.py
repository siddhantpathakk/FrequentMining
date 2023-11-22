from sklearn.datasets import load_wine, load_iris, load_breast_cancer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

import pandas as pd
from mlxtend.frequent_patterns import apriori
from itertools import chain, combinations
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
    
    return clusters

def evalauate_clusters(cluster, target, mapping=None):
    if mapping:
        cluster = [mapping[i] for i in cluster]
    print(f"V measure Score: {v_measure_score(cluster,target)}")
    print(f"Homogeneity Score: {homogeneity_score(cluster,target)}")
    print(f"Completeness Score: {completeness_score(cluster,target)}")
    print(f"Adjusted Rand Score: {adjusted_rand_score(cluster,target)}")
    print(f"Adjusted Mutual Info Score: {adjusted_mutual_info_score(cluster, target)}")