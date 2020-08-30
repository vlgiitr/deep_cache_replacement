import argparse
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import codecs

def minDistance(arr): 
      
    mp = {}
    dist = {} 
  

    for i in range(len(arr)): 

        if arr[i] not in mp.keys(): 
            mp[arr[i]] = i
            dist[arr[i]] = float('inf') 
        else:
            dist[arr[i]] = min(dist[arr[i]],i-mp[arr[i]])
            mp[arr[i]] = i

    return dist

def Freq(arr): 
       
    mp = {} 

    for i in range(len(arr)): 
  

        if arr[i] not in mp.keys(): 
            mp[arr[i]] = 1 
  
        else: 
            mp[arr[i]] += 1 
  
    return mp  

def main(args):
    count = 0
    page_counters = []
    addresses = []
    frequencies = []
    distances = []
    
# For data from txt file    
    # with codecs.open(args.r, 'r', encoding='utf-8',errors='ignore') as file:
    #     inputFile=file.readlines()
    # for line in tqdm(inputFile):
    #     item = line.split(" ")
    #     if len(item) is 4:
    #         page_counters.append(item[1])
    #         addresses.append(item[2])
    #     else:
    #         print('---------------------------')
    #         print(len(item))
    #         print(item)
    #         print('---------------------------')
    #     count+=1
    # print('---------------------------')
    # print('Count: {}'.format(count))
    # print('---------------------------')

# For data fri]om csv file
    with open(args.r,'r') as file:
        reader = csv.reader(file)
        for row in reader:
            count+=1
            if count == 1:
                continue
            else:
                page_counters.append(row[1])
                addresses.append(row[2])
    print('---------------------------')
    print('Count: {}'.format(count))
    print('---------------------------')

    address_freq = Freq(addresses)
    address_dist = minDistance(addresses)

    for address in addresses:
        frequencies.append(address_freq[address])
        distances.append(address_dist[address])
    
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

