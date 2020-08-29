import numpy as np
import pandas as pd
from collections import deque, defaultdict
from tqdm import tqdm 
  
df = pd.read_csv('csv_data/cse240_project_ucsd/g++.csv')
print(df.head())


def LRU(blocktrace, frame):
    
    cache = set()
    recency = deque()
    hit, miss = 0, 0
    
    for i in tqdm(range(len(blocktrace))):
        block = blocktrace[i]
        if block in cache:
            recency.remove(block)
            recency.append(block)
            hit += 1
            
        elif len(cache) < frame:
            cache.add(block)
            recency.append(block)
            miss += 1
            
        else:
            cache.remove(recency[0])
            recency.popleft()
            cache.add(block)
            recency.append(block)
            miss += 1
    
    hitrate = hit / (hit + miss)
    return hitrate



trace = df['Address'].tolist()

scores = []

size_list = list(range(50,5000,50))
# size_list = [50,100]


for i in range(len(size_list)):
    print("I: ",i)
    size = size_list[i]
    scores.append(LRU(trace,size))

df = pd.DataFrame(list(zip(size_list, scores)), 
               columns =['size', 'hit_rate']) 

df.to_csv('LRU.csv',index=False)