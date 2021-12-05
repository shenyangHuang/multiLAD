import sys
sys.path.append('../')

import numpy as np
import tensorly as tl
import networkx as nx
import scipy
import sparse
import tensorly
from tensorly.contrib.sparse.decomposition import parafac as sparse_parafac
from tensorly.decomposition import parafac, non_negative_parafac, randomised_parafac
from datasets import SBM_loader
from util import normal_util
import re
import pickle
from math import sqrt
from time import process_time 


def compute_accuracy(anomalies, real_events):

    correct = 0
    for anomaly in anomalies:
        if anomaly in real_events:
            correct = correct + 1

    return (correct/len(real_events))


'''
pad to num nodes * num nodes
'''

def toArray(G_times, num_nodes, directed=False, normalized_laplacian=False, normalized_adjacency=False):
    view_arr = []
    for G in G_times:
        if (len(G) < num_nodes):
            for i in range(len(G), num_nodes):
                G.add_node(-1 * i)      #add empty node with no connectivity (zero padding)
        
        A = nx.to_numpy_matrix(G)
        
        if (normalized_laplacian):
            A = nx.linalg.normalized_laplacian_matrix(G, weight='weight')

        if (normalized_adjacency):
            
            degrees = [val for (node, val) in G.degree(weight='weight')]
            D = np.zeros(A.shape)  #D^{-1/2}
            for i in range(len(degrees)):
                if degrees[i] == 0:
                    D[i,i] = 0
                else:
                    D[i,i] =(1 / sqrt(degrees[i]))
            A = np.dot(D,A)
            A = np.dot(A,D)

        A = np.asarray(A)
        A.astype(float)
        view_arr.append(A)
    return np.asarray(view_arr)



'''
G_times: is a temporal graph where each element in the list is a networkx graph
return a third order tensor T
'''
def toTensor(arr, Sparse=False):
    
    if (Sparse):
        #sparse tensor
        T = tensorly.contrib.sparse.tensor(arr)
    else:
        #dense tensor
        T = tl.tensor(arr)
    print (type(T))

    return T 

'''
apply parafac decomposition on tensor
'''
def apply_parafac(T, rank=10, normalize=False, Sparse=False):
    #print (T.shape)
    if (Sparse):
        factors = sparse_parafac(T, rank, init='random', normalize_factors=normalize)
    else:
        factors = parafac(T, rank, init='svd', normalize_factors=normalize)
    return factors




'''
check if all elements in this array is sparse array
'''
def check_type(T):
    for i in range(len(T)):
        if (type(T[i]) is not sparse._coo.core.COO):
            print ("shape of T " + str(i) +  "is")
            print (type(T[i]))
        for j in range(len(T[0])):
            if (type(T[i][j]) is not sparse._coo.core.COO):
                print ("shape of T " + str(i) + str(j)  + "is")
                print (type(T[i][j]))
            for k in range(len(T[0][0])):
                if (type(T[i][j][k]) is not sparse._coo.core.COO):
                    print ("shape of T " + str(i) + str(j) + str(k) + "is")
                    print (type(T[i][j][k]))



def find_synthetic_factors(fnames, outName, rank, Sparse=False, normalized_laplacian=False, normalized_adjacency=False):
    num_nodes = 500

    T_arr = []
    for fname in fnames:
        G_times = SBM_loader.load_temporarl_edgelist(fname + ".txt")
        view = toArray(G_times, num_nodes, normalized_laplacian=normalized_laplacian, normalized_adjacency=False)
        T_arr.append(view)

    T_arr = np.asarray(T_arr)
    if (Sparse):
        T_arr = sparse.COO.from_numpy(T_arr)
    T = toTensor(T_arr, Sparse=Sparse)
    print ("CPD starts")
    print (process_time())
    factors = apply_parafac(T, rank=rank, Sparse=Sparse)
    print (process_time())
    print ("CPD ends")
    normal_util.save_object(factors[1][1], outName + str(rank) +".pkl")
    print("factors saved")






def main():
    print ("hi")
    
if __name__ == "__main__":
    main()
