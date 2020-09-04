import numpy as np
import pandas as pd
from collections import deque, defaultdict
from tqdm import tqdm 
#import utils.standard_algo as standard_algo
from utils.standard_algo import LRU, LFU, Belady, FIFO, LIFO, Arc, Lecar
import os
from glob import glob


class NewDict(dict) :
    def __getitem__(self, key):
        val = super().__getitem__(key)
        if callable(val) :
            return val
    
function_dict = NewDict({"LRU" : LRU,
                         "LFU" : LFU,
                         "Belady" : Belady,
                         "FIFO" : FIFO,
                         "LIFO" : LIFO,
                         "ARC" : Arc,
                         "LECAR" : Lecar})


def get_hit_rate_across_datasets(algo_name,cache_size):
    PATH = "data/csv_data"
    EXT = "*.csv"
    all_csv_files = [file
                    for path, subdir, files in os.walk(PATH)
                    for file in glob(os.path.join(path, EXT))]
                    
    non_miss_scores = []
    miss_scores  = []
    overall_scores = []
    

 
    print(f'\n\n---Running {algo_name}---')
    for i, path in enumerate(all_csv_files) :
        # print("File {} / {}".format(i+1, len(all_csv_files)))
        df = pd.read_csv(path)
        print(path)

        if all(x in path for x in ['misses', 'lru']) :
                trace = df['LRU Miss Address'].tolist()
                mode = 'misses'

        elif all(x in path for x in ['misses', 'lfu']) :
                trace = df['LFU Miss Address'].tolist()
                mode = 'misses'
        else : 
            trace = df['Address'].tolist()
            mode = 'normal'

        if  mode == 'misses':
            miss_scores.append(function_dict[algo_name](trace, cache_size))
        else : 
            non_miss_scores.append(function_dict[algo_name](trace, cache_size))
        

    miss_scores, non_miss_scores = np.mean(miss_scores), np.mean(non_miss_scores)
    overall_scores = ( miss_scores + non_miss_scores ) / 2

    #return scores, np.mean(scores)
    return miss_scores, non_miss_scores, overall_scores

# def get_hit_rate_across_size(algo_name ,data_path, size_min, size_max, sample_rate , csv_name):
#     scores = []
#     size_list = list(range(size_min,size_max,sample_rate))

#     df = pd.read_csv(data_path)

#     # if 'lru' in path :
#     #         trace = df['LRU Miss Address'].tolist()
#     # elif 'lfu' in path :
#     #     trace = df['LFU Miss Address'].tolist()
#     # else : 
#     trace = df['Address'].tolist()  

#     for i in range(len(size_list)):
#         size = size_list[i]
#         if algo_name == 'LRU':
#             scores.append(standard_algo.LRU(trace,size))

#     df = pd.DataFrame(list(zip(size_list, scores)), 
#                 columns =['size', 'hit_rate']) 

#     df.to_csv(csv_name,index=False)