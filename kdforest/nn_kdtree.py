import sys
import pandas as pd
import numpy as np

def read_files(train, test):
    train_data = pd.read_csv(train, delimiter=r'\s+')
    train_quality = train_data["quality"]
    train_data = train_data.drop(columns="quality")
    test_data = pd.read_csv(test, delimiter=r'\s+')
    return np.array(train_data), np.array(train_quality), np.array(test_data)

class Node:
    def __init__(self, d, val, point, parent):
        self.d = d
        self.val = val
        self.point = point
        self.parent = parent

def build_KdTree(P, D = 0, parent = None):
    S = P.shape[0]
    # if P is empty then return null
    if S == 0:
        return None
    
    else:
        M = P.shape[1]
        d = D % M
        sorted_P = np.array(sorted(P, key=lambda x: x[d]))
        
        if S%2 == 0:
            val = (sorted_P[S//2 - 1][d]+sorted_P[S//2][d])/2 
            point = (sorted_P[S//2 - 1]+sorted_P[S//2])/2
        else:
            val = sorted_P[S//2][d]
            point = sorted_P[S//2]
            
        node = Node(d, val, point, parent)
        if S == 1:
            node.left  = None
            node.right  = None
            return node
        node.left  = build_KdTree(sorted_P[sorted_P[:, d] <= val], D + 1, node)
        node.right = build_KdTree(sorted_P[sorted_P[:, d] > val], D + 1, node)
        return node

def search_leaf(tree, point):
    if tree.right is None and tree.left is None:
        return tree
    elif point[tree.d] <= tree.val or tree.right is None:
        return search_leaf(tree.left, point)
    elif point[tree.d] > tree.val or tree.left is None:
        return search_leaf(tree.right, point)

def distance(p1, p2):
    return (np.sum((p1-p2)**2))**0.5

def nearest_neighbor(tree, point, i=0):

    leaf = search_leaf(tree, point)
    best_node = leaf
    best_distance = distance(leaf.point, point)
    node = leaf

    while node is not tree.parent:
        i +=1
        parent = node.parent
        if parent is not tree.parent:
            other_child = parent.right if node is parent.left else parent.left
            if other_child is not None:
                if abs(parent.point[parent.d] - point[parent.d]) < best_distance:
                    lower_node, lower_distance, i = nearest_neighbor(other_child, point, i)
                    if lower_distance < best_distance:
                        best_node, best_distance = lower_node, lower_distance
                        node = best_node
        node = node.parent
    return best_node, best_distance, i

def main():
    p, train, test, dimension = sys.argv
    train_data, train_quality, test_data = read_files(train, test)
    tree = build_KdTree(train_data, int(dimension))
    res = []
    for p in range(len(test_data)):
        best = nearest_neighbor(tree, test_data[p], int(dimension))
        index = np.where(np.all(train_data == best[0].point, axis=1))
        print(int(train_quality[index[0]]))

if __name__ == "__main__":
    main()
