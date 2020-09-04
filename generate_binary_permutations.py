import csv
import pandas as pd
 
def save_csv(save_arr,arr, n):  

    byte = ''
    for i in range(0, n):  
        byte+=str(arr[i]) 
    save_arr.append(byte)
    return save_arr 
  
# Function to generate all binary strings  
def generateAllBinaryStrings(n, arr, i,save_arr):  
  
    if i == n: 
        save_arr = save_csv(save_arr,arr, n)  
        return
      
    # First assign "0" at ith position  
    # and try for all other permutations  
    # for remaining positions  
    arr[i] = 0
    generateAllBinaryStrings(n, arr, i + 1,save_arr)  
  
    # And then assign "1" at ith position  
    # and try for all other permutations  
    # for remaining positions  
    arr[i] = 1
    generateAllBinaryStrings(n, arr, i + 1,save_arr)  
  
# Driver Code  
if __name__ == "__main__":  
  
    n = 8
    arr = [None] * 8
    save_arr = []  
 
    generateAllBinaryStrings(n, arr, 0,save_arr)
    print(len(save_arr))
    data = {'Bytes': save_arr}

    new_df = pd.DataFrame(data,columns=['Bytes'])
    new_df.to_csv('bytes.csv') 
