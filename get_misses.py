import argparse
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import codecs
from collections import deque, defaultdict
import os
from glob import glob

def get_files(p):
    
    PATH = p
    csvs = []
    files_list = [files for path, subdir, files in os.walk(PATH)]
    for files in files_list:
        for file in files:
            csvs.append(file)
    csv_files = [p+ '/' +file.split('.csv')[0] for file in csvs]
    
    return csv_files

def LRU(blocktrace, frame):
    
    cache = set()
    recency = deque()
    hit, miss = 0, 0
    miss_addresses = []
    
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
            miss_addresses.append(block)
            
        else:
            cache.remove(recency[0])
            recency.popleft()
            cache.add(block)
            recency.append(block)
            miss_addresses.append(block)
            miss += 1
    
    hitrate = hit / (hit + miss)
    print('---------------------------')
    print('LRU')
    print('HitRate: {}'.format(hitrate))
    print('Miss_length: {}'.format(len(miss_addresses)))
    print('---------------------------')
    return miss_addresses

def LFU(blocktrace, frame):
    
    cache = set()
    cache_frequency = defaultdict(int)
    frequency = defaultdict(int)
    
    hit, miss = 0, 0
    miss_addresses = []
    
    for block in tqdm(blocktrace):
        frequency[block] += 1
        
        if block in cache:
            hit += 1
            cache_frequency[block] += 1
        
        elif len(cache) < frame:
            cache.add(block)
            cache_frequency[block] += 1
            miss_addresses.append(block)
            miss += 1

        else:
            e, f = min(cache_frequency.items(), key=lambda a: a[1])
            cache_frequency.pop(e)
            cache.remove(e)
            cache.add(block)
            cache_frequency[block] = frequency[block]
            miss_addresses.append(block)
            miss += 1
    
    hitrate = hit / ( hit + miss )
    print('---------------------------')
    print('LFU')
    print('HitRate: {}'.format(hitrate))
    print('Miss_length: {}'.format(len(miss_addresses)))
    print('---------------------------')
    return miss_addresses


def main(args):
    
    files = get_files(args.r)

    for f in files:
        count = 0
        addresses = []
        lru_misses = []
        lfu_misses = []

    # # For data from txt file    
    #     with codecs.open(args.r, 'r', encoding='utf-8',errors='ignore') as file:
    #         inputFile=file.readlines()
    #     for line in tqdm(inputFile):
    #         item = line.split(" ")
    #         if len(item) is 3:
    #             page_counters.append(item[0].split(':')[0])
    #             addresses.append(item[2])
    #         else:
    #             print('---------------------------')
    #             print(len(item))
    #             print(item)
    #             print('---------------------------')
    #         count+=1
    #     print('---------------------------')
    #     print('Count: {}'.format(count))
    #     print('---------------------------')

    # For data fri]om csv file
        with open(f+'.csv','r') as file:
            reader = csv.reader(file)
            for row in reader:
                count+=1
                if count == 1:
                    print(row)
                    continue
                else:
                    if count == 2:
                        print(row)
                    addresses.append(row[2])
        print('---------------------------')
        print('Count: {}'.format(count))
        print('---------------------------')

        lru_misses = LRU(addresses,32)
        lfu_misses = LFU(addresses,32)

        data_lru = {'LRU Miss Address': lru_misses}

        new_df_lru = pd.DataFrame(data_lru,columns=['LRU Miss Address'])
        new_df_lru.to_csv(args.r+'_lru_misses.csv')

        data_lfu = {'LFU Miss Address': lfu_misses}

        new_df_lfu = pd.DataFrame(data_lfu,columns=['LFU Miss Address'])
        new_df_lfu.to_csv(f+'_lfu_misses.csv')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train_overexposure")
    parser.add_argument("--r", required=True,
    help="path to directory containing the files")
    args =  parser.parse_args()

    main(args)


