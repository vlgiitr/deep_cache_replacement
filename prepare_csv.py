import argparse
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import codecs
import os
from glob import glob

def get_files(p):

    PATH = p
    txts = []
    files_list = [files for path, subdir, files in os.walk(PATH)]
    for files in files_list:
        for file in files:
            txts.append(file)
    txt_files = [p+ '/' +file.split('.txt')[0] for file in txts]
    
    return txt_files

def minDistance(arr): 
    """
Function to calculate future distance between equal elements
input: A list arr
output: A list distances with distance corresponding to each element of arr in the same order 
"""      
    mp = {} # dict to store the most recent and 2nd most recent index of an element
    indexes = {} # dict to store all the indexes for an element
    distances = [] 
  
    minDistance = int(len(arr))
    for i in range(len(arr)): 

        if arr[i] not in indexes.keys(): 
            mp[arr[i]] = [int(len(arr)),i] # if we see the element for the first time initialize the list as -inf and current index 
            indexes[arr[i]] = [i] # if we see the element for the first time initialize the list with the current index
        else: 
            indexes[arr[i]].append(i) # add the index to the corresponding list
            mp[arr[i]] = [mp[arr[i]][1],i] # update the most recent and 2nd most recent index 
            dist = i - mp[arr[i]][0] # calculate distance for the 2nd most recent index
            if dist < minDistance:
                distances[mp[arr[i]][0]] = dist # update the distance 
        distances.append(minDistance) # add inf as the distance for the current element which will be updated later
    return distances

def Freq(arr): 
    """
Function to calculate future frequencies 
input: A list arr
output: A list frequencies with frequency corresponding to each element of arr in the same order 
"""      
    indexes = {} # dict to store all the indexes for an element
    frequencies = [1]*len(arr) # initialize frequency list

    for i in range(len(arr)): 

        if arr[i] not in indexes.keys(): 
            indexes[arr[i]] = [i] # if we see the element for the first time initialize the list with the current index
        else: 
            indexes[arr[i]].append(i)
    for address in arr:
        if len(indexes[address]) >=1: #check if the element is repeated
            freq = len(indexes[address]) - 1 # freq is the number of entries in the indexes list except the current element (which is always the 1st one)
            frequencies[indexes[address][0]] = freq #update the frequency
            indexes[address].pop(0) # remove the element for which frequency has been calculated from the indexes list
    
    return frequencies

def main(args):
    
    files = get_files(args.r)

    for f in files:
        count = 0
        page_counters = []
        addresses = []
        frequencies = []
        distances = []
    
    # For data from txt file    
        with codecs.open(f+'.txt', 'r', encoding='utf-8',errors='ignore') as file:
            inputFile=file.readlines()
        for line in tqdm(inputFile):
            item = line.split(" ")
            if len(item) is 4:
                page_counters.append(item[1])
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
    
    #  For data fri]om csv file
        #  with open(args.r,'r') as file:
            #  reader = csv.reader(file)
            #  for row in reader:
                #  count+=1
                #  if count == 1:
                    #  continue
                #  else:
                    #  page_counters.append(row[1])
                    #  addresses.append(row[2])
        #  print('---------------------------')
        #  print('Count: {}'.format(count))
        #  print('---------------------------')
    
        frequencies = Freq(addresses)
        distances = minDistance(addresses)
        
        data = {'PC': page_counters, 'Address': addresses, 'Frequency': frequencies, 'Recency': distances}
    
        new_df = pd.DataFrame(data,columns=['PC','Address', 'Frequency', 'Recency'])
        new_df.to_csv(f+'.csv')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train_overexposure")
    parser.add_argument("--r", required=True,
    help="path to directory containing the files")
    args =  parser.parse_args()

    main(args)


