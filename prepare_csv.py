import argparse
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import codecs

def minDistance(arr): 
      
    mp = {}
    indexes = {}
    distances = [] 
  
    minDistance = float('inf')
    for i in range(len(arr)): 

        if arr[i] not in indexes.keys(): 
            mp[arr[i]] = [-1,i]
            indexes[arr[i]] = [i]
        else: 
            indexes[arr[i]].append(i)
            mp[arr[i]] = [mp[arr[i]][1],i]
            dist = i - mp[arr[i]][0]
            distances[mp[arr[i]][0]] = dist
        distances.append(minDistance)
    return distances

def Freq(arr): 
       
    indexes = {}
    frequencies = [1]*len(arr)

    for i in range(len(arr)): 

        if arr[i] not in indexes.keys(): 
            indexes[arr[i]] = [i]
        else: 
            indexes[arr[i]].append(i)
    for address in arr:
        if len(indexes[address]) >=1:
            freq = len(indexes[address]) - 1
            frequencies[indexes[address][0]] = freq
            indexes[address].pop(0)

    return frequencies

def main(args):
    count = 0
    page_counters = []
    addresses = []
    frequencies = []
    distances = []
    
# For data from txt file    
    with codecs.open(args.r, 'r', encoding='utf-8',errors='ignore') as file:
        inputFile=file.readlines()
    for line in tqdm(inputFile):
        item = line.split(" ")
        if len(item) is 3:
            page_counters.append(item[0].split(':')[0])
            addresses.append(item[2])
        else:
            print('---------------------------')
            print(len(item))
            print(item)
            print('---------------------------')
        count+=1
    print('---------------------------')
    print('Count: {}'.format(count))
    print('---------------------------')

# # For data fri]om csv file
#     with open(args.r,'r') as file:
#         reader = csv.reader(file)
#         for row in reader:
#             count+=1
#             if count == 1:
#                 continue
#             else:
#                 page_counters.append(row[1])
#                 addresses.append(row[2])
#     print('---------------------------')
#     print('Count: {}'.format(count))
#     print('---------------------------')

    frequencies = Freq(addresses)
    distances = minDistance(addresses)
    
    data = {'PC': page_counters, 'Address': addresses, 'Frequency': frequencies, 'Recency': distances}

    new_df = pd.DataFrame(data,columns=['PC','Address', 'Frequency', 'Recency'])
    new_df.to_csv(args.w)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train_overexposure")
    parser.add_argument("--r", required=True,
    help="txt file")
    parser.add_argument("--w", required=True,
    help="csv to write to")
    args =  parser.parse_args()

    main(args)

