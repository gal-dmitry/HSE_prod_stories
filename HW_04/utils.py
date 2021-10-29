import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from nltk.cluster import KMeansClusterer
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import rand_score

import umap
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA



def bar_chart(types, counts):
    plt.bar(types, counts)
    plt.show()
    
    
def get_dct(model_names, hist_path, count=None):
    
    dct = {}
    if count != None:
        model_names = model_names[:count]
        
    for name in model_names:
        with open(f"{hist_path}/{name}.json") as f:
            data = json.load(f)
            
            df_hist = pd.DataFrame(columns=[8, 16, 32, 64, 128], 
                                   index=['model_bounding_sphere_strict_outer',
                                          'model_bounding_sphere_strict_outer_absolute',
                                          'model_bounding_sphere_missed',
                                          'model_bounding_sphere_concentric_sphere',
                                          'hull_bounding_sphere_strict_outer',
                                          'hull_bounding_sphere_strict_outer_absolute',
                                          'hull_bounding_sphere_missed',
                                          'hull_bounding_sphere_concentric_sphere'])
            
            for _dct in data['histogram_data']:
                i, j, dt = _dct['type'], _dct['intervals'], _dct['data']
                df_hist.loc[i, j] = dt
                
            dct[name] = df_hist
        
    return dct



def form_dataset(dct, labels, _type='all', _intervals=8, standardize=False):

    keys, vectors = [], []
    for key, df in dct.items():      
        vector = []
        
        if _type == 'all':
            lst = df[_intervals].tolist()
            for hist in lst:
                vector.extend(hist)
        else:           
            lst = df[_intervals].tolist()
            
            if _type == 'analytic_model':
                lst = lst[:4]  
            elif _type == 'analytic_hull':    
                lst = lst[4:]
            elif _type == 'analytic':
                lst = lst
                
            elif _type == 'analytic_3':
                lst = [lst[0], lst[2], lst[3]]    
            
            elif _type == 'analytic_2':
                lst = [lst[0], lst[3]]
                
            for hist in lst:
                hist = np.array(hist)
                if standardize:
                    hist = (hist - hist.mean()) / hist.std()
                vector.append(hist)
                
            vector = np.array(vector)
                        
        vectors.append(vector)
        keys.append(key)
        
    sec_df = pd.DataFrame(data={"name": keys, "vectors": vectors})

    return pd.concat([labels.copy(), sec_df.drop(columns=["name"])], axis=1)


def get_cluster_stat(df):
    
    for num in range(df.cluster.unique().shape[0]):
    
        types, counts = np.unique(df[df.cluster == num].type, return_counts=True)
        
        print(f"--- cluster: {num} ---")
#         for _type, count in zip(types, counts):
#             print(f"{_type}: {count}")
#         print()
        
        bar_chart(types, counts)


    
"""
Kmeans sklearn
"""
def get_pred_labels(df):
    
    X = df.vectors.to_list()
    n_clusters = df.type.unique().shape[0]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    
    labels_pred = kmeans.labels_
    labels_true = df.type.to_list()
    rand_index = rand_score(labels_true, labels_pred)
    print(f"rand_index: {rand_index}" + "\n")
    
    df["cluster"] = labels_pred
    get_cluster_stat(df)
    
    return df 
    

    
"""
Kmeans NLTK
"""
def get_distance(h_x, h_y):    
    D = 0.5 * np.sum(np.abs(h_x - h_y)) / (np.sum(h_x) + np.sum(h_y)) + \
        0.5 * np.sqrt(np.sum((h_x - h_y)**2) / (np.sum(h_x)**2 + np.sum(h_y)**2))  
    return D


def analytic_distance(x, y):
    
    coefs = np.array([0.25, 0.25, 0.25, 0.25])
    distances = []
    
    for h_x, h_y in zip(x, y):
        D = get_distance(h_x, h_y)
        distances.append(D)      
        
    distances = np.array(distances)
    
    return np.sum(distances * coefs)


class AnalyticDistance:
    
    def __init__(self, coefs):
        self.coefs = coefs
        
    def __call__(self, x, y): 
        
        distances = []

        for h_x, h_y in zip(x, y):
            D = get_distance(h_x, h_y)
            distances.append(D)      

        distances = np.array(distances)

        return np.sum(distances * self.coefs)


def get_pred_labels_analytic(df, dist):
    
    X = df.vectors.to_list()
    n_clusters = df.type.unique().shape[0]

    clusterer = KMeansClusterer(n_clusters, dist)
    
    labels_pred = clusterer.cluster(X, True, trace=True)
    labels_true = df.type.to_list()
    rand_index = rand_score(labels_true, labels_pred)
    print(f"rand_index: {rand_index}" + "\n")
    
    df["cluster"] = labels_pred
    get_cluster_stat(df)
    
    return df 



# def get_pred_labels_analytic_DBSCAN(df, dist):
    
#     X = df.vectors.to_list()
#     n_clusters = df.type.unique().shape[0]

#     clusterer = DBSCAN(metric=dist).fit(X)
    
#     labels_pred = clusterer.labels_
#     labels_true = df.type.to_list()
#     rand_index = rand_score(labels_true, labels_pred)
#     print(f"rand_index: {rand_index}" + "\n")
    
#     df["cluster"] = labels_pred
#     get_cluster_stat(df)
    
#     return df 





"""
Visualize
"""
def reduce_fingerprints(fs, mode='TSNE'):
    if mode == 'UMAP':
        reducer = umap.UMAP()
    elif mode == 'TSNE':
        reducer = TSNE(n_components=2)
    elif mode == 'PCA':
        reducer = PCA(n_components=2)        
    else:
        raise ValueError("Unknown reducer")
        
    return reducer.fit_transform(fs)    
    
    
def show_clusters(df, mode="TSNE"):
    
    df["reduced_vectors"] = reduce_fingerprints(df.vectors.to_list(), mode=mode).tolist()
    
    assert len(df.cluster.unique()) == 5
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for cluster in sorted(df.cluster.unique()):
        points = df[df.cluster == cluster].reduced_vectors
        
        xs, ys = [], []
        for pair in points:
            x, y = pair
            xs.append(x)
            ys.append(y)
        
        color = colors[cluster]
        ax.scatter(xs, ys, c=color, label=f"cluster: {cluster}", alpha=0.3, edgecolors='none')
            
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    plt.show()

    
    



    