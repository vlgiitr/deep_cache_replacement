import numpy as np
import pandas as pd
import utils.standard_algo as standard_algo
import utils.benchmark as benchmark

scroes_lru, mean_score_lru = benchmark.get_hit_rate_across_datasets('LRU',50)
scroes_belady, mean_score_belady = benchmark.get_hit_rate_across_datasets('Belady',50)
scroes_lfu, mean_score_lfu = benchmark.get_hit_rate_across_datasets('LFU',50)
scroes_fifo, mean_score_fifo = benchmark.get_hit_rate_across_datasets('FIFO',50)
scroes_lifo, mean_score_lifo = benchmark.get_hit_rate_across_datasets('LIFO',50)

print("Belady: ",mean_score_belady)

# print(scroes_lru)
print("LRU: ",mean_score_lru)

# print(scroes_belady)

# print(scroes_lfu)
print("LFU: ",mean_score_lfu)

# print(scroes_fifo)
print("FIFO: ",mean_score_fifo)

# print(scroes_lifo)
print("LIFO: ",mean_score_lifo)
