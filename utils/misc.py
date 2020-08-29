import os
from glob import glob
import numpy as np
import pandas as pd
    

def get_complete_data_padded():
    PATH = "csv_data"
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
        pc = df['PC'].apply(int, base=16)
        address = df['Address'].apply(int, base=16)
        dataset[i,:len(address),0]  = pc
        dataset[i,:len(address),0]  = address
    
    return dataset