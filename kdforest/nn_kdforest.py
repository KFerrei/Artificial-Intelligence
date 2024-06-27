import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
from nn_kdtree import *

def kdForest(data, d_list, rand_seed):
    forest = []
    n_trees = len(d_list)
    N = data.shape[0]

    random.seed(rand_seed)
    sample_indexes = sample_indexes = np.array([[random.randint(0, N-1) for i in range(N)] for j in range(n_trees)])
    for count in range(n_trees):
        sampled_data = np.array([data[j] for j in sample_indexes[count]])
        
        seen = {}
        for p in sampled_data:
            key = tuple(p)
            if key not in seen:
                seen[key] = p
        sampled_data = np.array(list(seen.values()))

        tree = build_KdTree(sampled_data, d_list[count]) 
        forest.append(tree)
    return forest

def predictKdForest(forest, data, data_y, data_test, d_list):
    labels = []
    n = len(forest)
    for i in range(n):
        tree = forest[i]
        best = nearest_neighbor(tree, data_test,d_list[i])
        index = np.where(np.all(data == best[0].point, axis=1))
        labels.append(int(data_y[index[0]]))
    return max(set(labels), key=labels.count)

def main():
    p, train, test, rand_seed, d_list = sys.argv
    d_list = [int(d) for d in d_list.strip('[]').split(',')]
    train_data, train_quality, test_data = read_files(train, test)
    forest = kdForest(train_data, d_list, int(rand_seed))
    for p in range(len(test_data)):
        res = predictKdForest(forest, train_data, train_quality, test_data[p], d_list)
        print(res)

if __name__ == "__main__":
    main()