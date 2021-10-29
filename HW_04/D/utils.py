import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import rand_score

# import umap
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



# def form_dataset(dct, labels, _type='all', _intervals=8):
    
#     vector_df = pd.DataFrame(data={"vector": [[np.nan]*8*_intervals] * len(labels)})
#     new_df = pd.concat([labels.copy(), vector_df], axis=1)

# #     new_df = pd.concat([labels.copy(), vector_df], axis=1)  
# #     return new_df    
        
        
#     for key, df in dct.items():
        
#         vector = []
        
#         if _type == 'all':
#             lst = df[_intervals].tolist()
#             for hist in lst:
#                 vector.extend(hist)
        
# #         print(new_df)
# #         print(key)
# #         print(vector)
# #         new_df.model == key
#         print(new_df.loc[new_df.model == key, "vector"])
#         new_df.loc[new_df.model == key, "vector"] = vector
# #         print(new_df.loc[new_df.model == key, "vector"])
# #         new_df.loc[new_df.model == key]["vector"] = vector
    
#     return new_df



def form_dataset(dct, labels, _type='all', _intervals=8):

    keys, vectors = [], []
    for key, df in dct.items():      
        vector = []
        
        if _type == 'all':
            lst = df[_intervals].tolist()
            for hist in lst:
                vector.extend(hist)
        
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
            

def reduce_fingerprints(fingerprints, mode='TSNE'):
    '''
    takes fingerprints (and optionally rationale_fingerprints) and reduce it for 2D scatter plot
    '''   
    if mode == 'UMAP':
        reducer = umap.UMAP()
    elif mode == 'TSNE':
        reducer = TSNE(n_components=2)
    elif mode == 'PCA':
        reducer = PCA(n_components=2)        
    else:
        raise ValueError("Unknown reducer")
        
    return reducer.fit_transform(fingerprints)
    
    
def get_pred_labels(df):
    
    X = df.vectors.to_list()
    n_clusters = df.type.unique().shape[0]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    
    labels_pred = kmeans.labels_
    labels_true = df.type.to_list()
    rand_index = rand_score(labels_true, labels_pred)
    print(f"rand_index: {rand_index}" + "\n")
    
    df["cluster"] = labels_pred
    df["reduced_vectors"] = reduce_fingerprints(df.vectors.to_list()).tolist()
    get_cluster_stat(df)
    
    return df 
    
    
def show_clusters(df):
    
    assert len(df.cluster.unique()) == 5
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    fig, ax = plt.subplots(figsize=(12, 12))
    
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
    ax.legend()
    ax.grid(True)
    plt.show()



    