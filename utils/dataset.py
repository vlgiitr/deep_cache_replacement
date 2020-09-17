import os
from glob import glob
import numpy as np
import pandas as pd
from collections import Counter, deque, defaultdict
from tqdm import tqdm as tqdm

def get_complete_data_padded():
    PATH = "dataset/"
    EXT = "*.csv"
    all_csv_files = [file
                    for path, subdir, files in os.walk(PATH)
                    for file in glob(os.path.join(path, EXT))]
    len_list = []
    
    for path in all_csv_files:
        df = pd.read_csv(path)
        len_list.append(df.shape[0])

    max_len = np.max(len_list)
    num_datasets = len(len_list)

    dataset = np.zeros((num_datasets,max_len,2), dtype = "int")
    print(dataset.shape)


    for i,path in enumerate(all_csv_files):
        df = pd.read_csv(path)
        pc = (df['PC'].apply(int, base=16)).zfill(35)
        address = df['Address'].apply(int, base=16)
        dataset[i,:len(address),0]  = pc
        dataset[i,:len(address),0]  = address
    
    return dataset


def create_inout_sequences(input_data, labels,tw):
    inout_seq = []
    L = len(input_data)
    x = []
    y = []
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = (input_data[i+tw:i+tw+1] , labels[i+tw:i+tw+1 , 1] ,labels[i+tw:i+tw+1 , 2])
        x.append(train_seq)
        y.append(train_label)
    return x,y

