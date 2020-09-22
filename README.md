# DEAP Cache: Deep Eviction Admission and Prefetching for Cache
Ayush Mangal*, Jitesh Jain*, Keerat Kaur Guliani*, Omkar Bhalerao*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

## Contents
1. [Overview](#1-overview)
2. [Setup Instructions](#2-setup-instructions)
3. [Repository Overview](#3-repository-overview)
4. [Training and Testing](#4-training-and-testing)
5. [Results](#5-results)
6. [License](#6-license)

## 1. Overview

This repo contains the code for the paper [DEAP Cache: Deep Eviction Admission and Prefetching for Cache](https://arxiv.org/abs/2009.09206).

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
- `embed_lstm_32.py` - Script for training the byte embeddings 
- `generate_binary_permutations.py` - Script for generating a csv file with all the binary representations of numbers till 255 for the global vocabulary
- `get_misses.py` - Script for storing the missed addresses and PCs in csv files
- `requirements.txt` - Contains all the dependencies required for running the code
- `standard_algo_benchmark.py` - Script for caclculating hitrates on the dataset using all the baselines algorithms
- `test_sim.py` - Script for running the **online test simulation**

## 4. Training and Testing

- To train the byte-embeddings, run the following command:
```
$ python embed_lstm_32.py 
```
- To train DeepCache, run the following command:
```
$ python cache_model_train.py
```
- To run the online test simulation, run the following command
```
$ python test_sim.py
```

## 5. Results

The hit-rates for various baselines and **our** approach are given in the table below:

| Method  | Mean Hit-Rate |
| ------- | ------------- |
| LRU     | 0.42          |
| LFU     | 0.43          |
| FIFO    | 0.36          |
| LIFO    | 0.03          |
| BELADY  | 0.54          |
| **Ours**| **0.48**      |

It can be observed that our method comes the closest in performance to the optimal figure obtained from BELADY’s algorithm (Oracle), thus demonstrating the validity of our approach.

## 6. License

The code is released under **MIT** License.
