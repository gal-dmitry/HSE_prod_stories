import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt


def update_writer_training(writer, result, n, _print=False):
    
    if _print:
        print(result.keys())
        for key, value in result.items():
            print()
            print()
            print(key)
            print()
            print(value)
    
    n +=1        
         
    lst = ["episode_reward_min",
           "episode_reward_mean",
           "episode_reward_max",
           "episode_len_mean",
           "episodes_this_iter",
           "episodes_total",
           "training_iteration"]    
    
    '''print'''
    s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f}"
    print(s.format(n + 1, result[lst[0]], result[lst[1]], result[lst[2]], result[lst[3]]))
    
    '''tensorboard'''
    for key in lst:
        writer.add_scalar(key, result[key], n)

    dct = result['info']['learner']['default_policy']['learner_stats']
    for key, value in dct.items():
        writer.add_scalar(key, value, n)

    return writer



def update_writer_trajectory(writer_trajectory, info, reward, _):

    info["total_unexplored"] = info["total_cells"] - info["total_explored"]
    info["total_explored_%"] = info["total_explored"] / info["total_cells"] 
    info["new_explored_abs_%"] = info["new_explored"] / info["total_cells"]
    info["new_explored_relative_%"] = info["new_explored"] / (info["total_unexplored"] + info["new_explored"])
    
    writer_trajectory.add_scalar("reward", reward, _)
    
    for key, value in info.items():
        if type(value) == bool:
            value = int(value)
        writer_trajectory.add_scalar(key, value, _)
    
    return writer_trajectory
   
    
    
def seed_everything(seed, env=None):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    

    
    
    
    
"""
IMG
"""
class Pictures(Dataset):
    """
    Dataset with pictures
    """
    def __init__(self, path_to_dataset):
        self.names = os.listdir(path_to_dataset)
        self.names.sort()
        if '.ipynb_checkpoints' in self.names:
            self.names.remove('.ipynb_checkpoints')
        self.path = path_to_dataset
    
    def __getitem__(self, index):
        full_path = self.path + "/" + self.names[index]
        if not os.path.exists(full_path):
            return None
        img = plt.imread(full_path)
        return np.array(img)

    def __len__(self):
        return len(self.names)

    
    
def show_pictures(path):
    
    dataset = Pictures(path)
    size = len(dataset)
    indx = range(size)
    imgs = [dataset[i] for i in indx]
    
    fig = plt.figure(figsize=(1.5*size, 3*size))
    columns = 1
    rows = size
    ax = []
    
    for num, img in enumerate(imgs):                 
        ax.append(fig.add_subplot(rows, columns, num+1))
        plt.imshow(img)
        plt.axis('off') 

    plt.tight_layout()