import numpy as np
import pandas as pd
from collections import deque, defaultdict
from tqdm import tqdm 
import utils.standard_algo as standard_algo
import os
from glob import glob

def get_hit_rate_across_datasets(algo_name,cache_size):
    PATH = "csv_data"
    EXT = "*.csv"
    all_csv_files = [file
                    for path, subdir, files in os.walk(PATH)
                    for file in glob(os.path.join(path, EXT))]

    scores = []

    for path in all_csv_files:
        df = pd.read_csv(path)
        trace = df['Address'].tolist()

        if algo_name == 'LRU':
            scores.append(standard_algo.LRU(trace,cache_size))

        if algo_name == 'Belady':
            scores.append(standard_algo.Belady(trace,cache_size))

        if algo_name == 'LFU':
            scores.append(standard_algo.LFU(trace,cache_size))

        if algo_name == 'FIFO':
            scores.append(standard_algo.FIFO(trace,cache_size))

        if algo_name == 'LIFO':
            scores.append(standard_algo.LIFO(trace,cache_size))

    return scores, np.mean(scores)

def get_hit_rate_across_size(algo_name ,data_path, size_min, size_max, sample_rate , csv_name):
    scores = []
    size_list = list(range(size_min,size_max,sample_rate))

    df = pd.read_csv(data_path)
    trace = df['Address'].tolist()    

    for i in range(len(size_list)):
        size = size_list[i]
        if algo_name == 'LRU':
            scores.append(standard_algo.LRU(trace,size))

    df = pd.DataFrame(list(zip(size_list, scores)), 
                columns =['size', 'hit_rate']) 

    df.to_csv(csv_name,index=False)