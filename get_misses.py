import argparse
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import codecs
from collections import deque, defaultdict
import os
from glob import glob
from pathlib import Path

def get_files(p):
    
    PATH = p
    csvs = []
    files_list = [files for path, subdir, files in os.walk(PATH)]
    for files in files_list:
        for file in files:
            csvs.append(file)
    csv_files = [p +file.split('.csv')[0] for file in csvs]
    
    return csv_files

def LRU(blocktrace,pcs,recencies,frequencies,frame):
    
    cache = set()
    recency = deque()
    hit, miss = 0, 0
    miss_addresses = []
    pc_misses = []
    recencies_misses = []
    frequencies_misses = []
    
    for i in tqdm(range(len(blocktrace))):
        block = blocktrace[i]
        pc = pcs[i]
        rec = recencies[i]
        freq = frequencies[i]
        if block in cache:
            recency.remove(block)
            recency.append(block)
            hit += 1
            
        elif len(cache) < frame:
            cache.add(block)
            recency.append(block)
            miss += 1
            miss_addresses.append(block)
            pc_misses.append(pc)
            recencies_misses.append(rec)
            frequencies_misses.append(freq)
            
        else:
            cache.remove(recency[0])
            recency.popleft()
            cache.add(block)
            recency.append(block)
            miss_addresses.append(block)
            pc_misses.append(pc)
            recencies_misses.append(rec)
            frequencies_misses.append(freq)
            miss += 1
    
    hitrate = hit / (hit + miss)
    print('---------------------------')
    print('LRU')
    print('HitRate: {}'.format(hitrate))
    print('Miss_length: {}'.format(len(miss_addresses)))
    print('---------------------------')
    return miss_addresses,pc_misses,recencies_misses,frequencies_misses

def LFU(blocktrace,pcs,recencies,frequencies,frame):
    
    cache = set()
    cache_frequency = defaultdict(int)
    frequency = defaultdict(int)
    
    hit, miss = 0, 0
    miss_addresses = []
    pc_misses = []
    recencies_misses = []
    frequencies_misses = []
    
    for i in tqdm(range(len(blocktrace))):
        block = blocktrace[i]
        frequency[block] += 1
        rec = recencies[i]
        freq = frequencies[i]
        
        if block in cache:
            hit += 1
            cache_frequency[block] += 1
        
        elif len(cache) < frame:
            cache.add(block)
            cache_frequency[block] += 1
            miss_addresses.append(block)
            pc_misses.append(pcs[i])
            recencies_misses.append(rec)
            frequencies_misses.append(freq)
            miss += 1

        else:
            e, f = min(cache_frequency.items(), key=lambda a: a[1])
            cache_frequency.pop(e)
            cache.remove(e)
            cache.add(block)
            cache_frequency[block] = frequency[block]
            miss_addresses.append(block)
            pc_misses.append(pcs[i])
            recencies_misses.append(rec)
            frequencies_misses.append(freq)
            miss += 1
    
    hitrate = hit / ( hit + miss )
    print('---------------------------')
    print('LFU')
    print('HitRate: {}'.format(hitrate))
    print('Miss_length: {}'.format(len(miss_addresses)))
    print('---------------------------')
    return miss_addresses,pc_misses,recencies_misses,frequencies_misses


def main(args):
    
    files = get_files(args.r)

    for f in files:
        count = 0
        addresses = []
        pcs= []
        recencies = []
        frequencies = []
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
                    frequencies.append(row[3])
                    recencies.append(row[4])
                    addresses.append(row[2])
                    pcs.append(row[1])
        print('---------------------------')
        print('Count: {}'.format(count))
        print('---------------------------')

        lru_misses,pcs_lru,recencies_lru,frequencies_lru = LRU(addresses,pcs,recencies,frequencies,32)
        lfu_misses,pcs_lfu,recencies_lfu,frequencies_lfu = LFU(addresses,pcs,recencies,frequencies,32)

        data_lru = {'PC': pcs_lru,'Address': lru_misses, 'Recency': recencies_lru, 'Frequency': frequencies_lru}

        new_df_lru = pd.DataFrame(data_lru,columns=['PC','Address','Recency','Frequency'])
        new_df_lru.to_csv(Path(f+'.csv').resolve().parents[1].joinpath('misses').joinpath(f.split(args.r)[1] +'_lru_misses.csv'))

        data_lfu = {'PC': pcs_lfu,'Address': lfu_misses, 'Recency': recencies_lfu, 'Frequency': frequencies_lfu}

        new_df_lfu = pd.DataFrame(data_lfu,columns=['PC','Address','Recency','Frequency'])
        new_df_lfu.to_csv(Path(f+'.csv').resolve().parents[1].joinpath('misses').joinpath(f.split(args.r)[1] +'_lfu_misses.csv'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train_overexposure")
    parser.add_argument("--r", required=True,
    help="path to directory containing the files")
    args =  parser.parse_args()

    main(args)

