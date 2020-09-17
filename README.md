# DEAP Cache: Deep Eviction Admission and Prefetching for Cache
Ayush Mangal, Jitesh Jain, Keerat Kaur Guliani, Omkar Sunil Bhalerao

## Contents
1. [Overview](#1-overview)
2. [Setup Instructions](#2-setup-instructions)
3. [Repository Overview](#3-repository-overview)

## Overview

This repo contains the code for the paper **DEAP Cache: Deep Eviction Admission and Prefetching for Cache** submitted at **AAAI 2021** as a *Student Abstract Paper*.

## 2. Setup Instructions

You can setup the repo by running the following commands:
```
$ git clone https://github.com/vlgiitr/deep_cache_replacement.git

$ pip install -r requirements.txt
```

## 3. Repository Overview

The repository contains the following modules:

- `checkpoints/` - Contains the pretrained embeddings and a trained version of the DeepCache model.
- `dataset/` - Dataset folder
    - `address_pc_files/` - Contains csv files with addresses and PCs with their corresponding future frequency and recency
    - `misses/` - Contains csv files with the missed (separately calculated for LRU and LFU) addresses and PCs with their corresponding future frequency and recency
- `runs/`  - Contains the tensorboard logs stored during DeepCache's training
- `utils/` - Contains various utility files such as `.py` scripts for various baselines, etc. 
- `cache_lecar.py` - Script for the modified LeCaR that evicts based on the future frequencies and recencies 
- `cache_model_train.py` - Script for training the DeepCache model.
- `create_train_dataset.py` - Script for creating the dataloader for training DeepCache
- `embed_32.py` - Script for training the byte embeddings 
- `generate_binary_permutations.py` - Script for generating a csv file with all the binary representations of numbers till 255 for the global vocabulary
- `get_misses.py` - Script for storing the missed addresses and PCs in csv files
- `requirements.txt` - Contains all the dependencies required for running the code
- `standard_algo_benchmark.py` - Script for caclculating hitrates on the dataset using all the baselines algorithms
- `test_sim.py` - Script for running the **online test simulation**