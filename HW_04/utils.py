import pandas as pd
import numpy as np
import json
import random
import itertools
import matplotlib.pyplot as plt
from nltk.cluster import KMeansClusterer
from sklearn.metrics.cluster import rand_score

import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


SEED = 42


"""
Itertools
"""
def C(n, k):
    lst_tpl = list(itertools.combinations(range(1, n), k))
    lst_lst = np.array([list(tpl) for tpl in lst_tpl]) - 1
    return lst_lst


"""
Datasets
"""   
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



def form_dataset(dct, labels, _type='concatenate_all', idx=None, _intervals=8, standardize=False):

    keys, vectors = [], []
    for key, df in dct.items():      
        
        lst = df[_intervals].tolist()
        vector = []
        
        if _type == 'concatenate_all':
            for hist in lst:
                vector.extend(hist)
                
        elif _type == 'analytic':                
            new_lst = lst if idx is None else [lst[i] for i in idx]
                
            for hist in new_lst:                
                if standardize:
                    hist = np.array(hist)
                    hist = (hist - hist.mean()) / hist.std()
                vector.append(hist)
                
        else:
            raise TypeError()
        
        vector = np.array(vector) 
        
        vectors.append(vector)
        keys.append(key)
        
    sec_df = pd.DataFrame(data={"name": keys, "vectors": vectors})

    return pd.concat([labels.copy(), sec_df.drop(columns=["name"])], axis=1)



"""
Statistics
"""
def bar_chart(types, counts):
    plt.bar(types, counts)
    plt.show()
    
    
def get_cluster_stat(df, mode='bar'):
    for num in range(df.cluster.unique().shape[0]):
        types, counts = np.unique(df[df.cluster == num].type, return_counts=True)
        print(f"--- cluster: {num} ---")
        if mode == 'bar':
            bar_chart(types, counts)
        else:
            for _type, count in zip(types, counts):
                print(f"{_type}: {count}")
            print()
        
        

"""
Analytic distance
"""
def get_distance(h_x, h_y):    
    D = 0.5 * np.sum(np.abs(h_x - h_y)) / (np.sum(h_x) + np.sum(h_y)) + \
        0.5 * np.sqrt(np.sum((h_x - h_y)**2) / (np.sum(h_x)**2 + np.sum(h_y)**2))  
    return D


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


def clusterize_analytic(df, dist, bar=False, return_score=False, trace=False):
    '''data'''
#     X = df.vectors.to_numpy()
    X = df.vectors.to_list()
    y_true = df.type.to_list()
    n_clusters = df.type.unique().shape[0]
    
    '''clusterer'''
    rng = random.Random()
    rng.seed(SEED)
    clusterer = KMeansClusterer(n_clusters, dist, rng=rng)
    y_pred = clusterer.cluster(X, True, trace=trace)
    df["cluster"] = y_pred
    
    '''score'''
    rand_index = rand_score(y_true, y_pred)

    if bar:
        get_cluster_stat(df)
    if return_score:
        return df, rand_index
    print(f"Rand index: {rand_index}" + "\n")
    return df


def best_search_analytic(intervals, idx, dct, labels):
    
    for i in idx:        
        for interval in intervals:
            train_df = form_dataset(dct, labels, _type='analytic', idx=i, _intervals=interval)
            
            size = len(i) 
            coef = [1/size for i in range(size)]
            
            print(f"idx: {i} | histogram intervals: {interval} | coeffs: {coef}")
            dist = AnalyticDistance(coef)
            _ = clusterize_analytic(train_df, dist)
        print()



"""
Distance metric learning
"""
def clusterize_dist_learn(df, learner, bar=False, return_score=False, trace=False):
    '''data'''
    X = df.vectors.to_list()
    y_true = df.type.to_numpy()
    n_clusters = df.type.unique().shape[0]
    
    _dct = {}
    for i, _str in enumerate(np.unique(y_true)):
        _dct[_str] = i
    y_true = np.array([_dct[_str] for _str in y_true])
    
    '''distance'''
    try:
        dist_learn = learner(random_state=SEED)        
    except:
        dist_learn = learner() 
    dist_learn.fit(X, y_true)
    dist = dist_learn.get_metric()
    
    '''clusterer'''
    rng = random.Random()
    rng.seed(SEED)
    clusterer = KMeansClusterer(n_clusters, dist, rng=rng)
    y_pred = clusterer.cluster(X, True, trace=trace)
    df["cluster"] = y_pred
    
    '''score'''
    rand_index = rand_score(y_true, y_pred)

    if bar:
        get_cluster_stat(df)
    if return_score:
        return df, rand_index
    print(f"Rand index: {rand_index}" + "\n")
    return df
    
        
def best_search_dist_learn(intervals, dist_learners, dct, labels):
    
    for learner in dist_learners:
        learner_name = str(learner).split('.')[-1][:-2]
        
        for interval in intervals:
            print(f"learner: {learner_name} | histogram intervals: {interval}")
            
            train_df = form_dataset(dct, labels, _type='concatenate_all', _intervals=interval)
            _ = clusterize_dist_learn(train_df, learner)
        print()
        
            
            
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

    
