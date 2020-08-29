import numpy as np
import pandas as pd
import utils.standard_algo as standard_algo
import utils.benchmark as benchmark

scroes_lru, mean_score_lru = benchmark.get_hit_rate_across_datasets('LRU',50)
print(scroes_lru)
print(mean_score_lru)

scroes_belady, mean_score_belady = benchmark.get_hit_rate_across_datasets('Belady',50)
print(scroes_belady)
print(mean_score_belady)
