import numpy as np
import pandas as pd
import utils.new_standard_algo as standard_algo
import utils.benchmark as benchmark

scores = [benchmark.get_hit_rate_across_datasets('LRU',50),
          benchmark.get_hit_rate_across_datasets('LFU',50),
          benchmark.get_hit_rate_across_datasets('FIFO',50),
          benchmark.get_hit_rate_across_datasets('LIFO',50),
          benchmark.get_hit_rate_across_datasets('Belady',50),
          benchmark.get_hit_rate_across_datasets('ARC',50),
          benchmark.get_hit_rate_across_datasets('LECAR',50)]


table = pd.DataFrame(scores, columns = ['Mean Miss Scores',
                                         'Mean Non Miss Scores',
                                         'Mean Overall Scores'])
table['ALGOs'] = ['LRU', 'LFU', 'FIFO', 'LIFO', 'BELADY', 'ARC', 'LECAR']

table.set_index(['ALGOs'], inplace = True)
print(table)

