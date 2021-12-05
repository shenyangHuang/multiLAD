import sys
sys.path.append('../')

import numpy as np
import networkx as nx
from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import svds, eigsh
from scipy import sparse
from numpy import linalg as LA
from datasets import SBM_loader
from util import normal_util
import math
import os

def random_SVD(G_times, max_size, directed=True, num_eigen=6, top=True):
    Temporal_eigenvalues = []
    activity_vecs = []  #eigenvector of the largest eigenvalue
    counter = 0


    for G in G_times:

        A = normalized_laplacian_matrix(G, weight='weight')

        if (top):
            which="LM"
        else:
            which="SM"

        u, s, vh = randomized_svd(A, num_eigen)
        vals = s
        vecs = u
        
        #eigenvector of the largest eigenvalue
        max_index = list(vals).index(max(list(vals)))
        activity_vecs.append(np.asarray(vecs[max_index]))
        Temporal_eigenvalues.append(np.asarray(vals))

        print ("processing " + str(counter), end="\r")
        counter = counter + 1

    return (Temporal_eigenvalues, activity_vecs)


'''
compute the eigenvalues for square laplacian matrix per time slice 
input: list of networkx Graphs
output: list of 1d numpy array of diagonal entries computed from SVD
randomized will use random svd
'''
def SVD_perSlice(G_times, max_size, directed=True, num_eigen=6, top=True, add_shift=False, weighted=False):
    Temporal_eigenvalues = []
    activity_vecs = []  #eigenvector of the largest eigenvalue
    counter = 0
    for G in G_times:
        if (len(G) < max_size):
            for i in range(len(G), max_size):
                G.add_node(-1 * i)      #add empty node with no connectivity (zero padding)

        if (directed):
            if (weighted):
                L = nx.directed_laplacian_matrix(G, weight='weight')
            else:
                L = nx.directed_laplacian_matrix(G)

        else:
            if (weighted):
                L = nx.laplacian_matrix(G, weight='weight')
            else:
                L = nx.laplacian_matrix(G)
                L = L.asfptype()

        if (add_shift):
            L = add_diagonal_shift(L, -10)
            L = sparse.csr_matrix(L)
            L = L.asfptype()

        if (top):
            which="LM"
        else:
            which="SM"

        u, s, vh = svds(L,k=num_eigen, which=which)
        vals = s
        vecs = u
        #second smallest eigenvalue's eigenvector fielder vector
        min_index = list(vals).index(min(list(vals)))
        activity_vecs.append(np.asarray(vecs[min_index]))
        Temporal_eigenvalues.append(np.asarray(vals))

        print ("processing " + str(counter), end="\r")
        counter = counter + 1

    return (Temporal_eigenvalues, activity_vecs)


def add_diagonal_shift(A, p=-10):
    #add a small shift for 0
    if (p==0):
        shift = 10.0**6
        np.fill_diagonal(A, A.diagonal() + shift)


    #do not shift for positive power
    elif (p>0):
        return A

    #shift for negative power
    else:
        shift = math.log(1 + abs(p))
        np.fill_diagonal(A, A.diagonal()+ shift)

    return A


'''
take the matrix power and find the eigenvalue
for the normalized laplacian matrix
'''
def find_NL_eigenvalues(G_times, max_size, num_eigen, p, add_shift=True, directed=True, top=True):
    Temporal_eigenvalues = []
    activity_vecs = []  #eigenvector of the largest eigenvalue
    counter = 0


    for G in G_times:
        if (len(G) < max_size):
            for i in range(len(G), max_size):
                G.add_node(-1 * i)      #add empty node with no connectivity (zero padding)

        L = nx.linalg.normalized_laplacian_matrix(G, weight='weight')
        L = L.todense()


        if (add_shift):
            L = add_diagonal_shift(L, p)
            L = sparse.csr_matrix(L)
            L = L.asfptype()

        
        if (top):
            which="LM"
        else:
            which="SM"

        u, s, vh = svds(L,k=num_eigen, which=which)
        vals = s
        vecs = u
        max_index = list(vals).index(max(list(vals)))
        activity_vecs.append(np.asarray(vecs[max_index]))
        Temporal_eigenvalues.append(np.asarray(vals))

        print ("processing " + str(counter), end="\r")
        counter = counter + 1

    return (Temporal_eigenvalues, activity_vecs)





    
def compute_multiSBM_SVD(fname, num_eigen=499, top=True, adj=False, normalized=False):
    edgefile = "datasets/multi_SBM/" + fname + ".txt"
    if (not os.path.isfile(edgefile)):
        edgefile = fname + ".txt"
    max_size = 500
    add_shift = True
    directed = False
    draw = False
    
    G_times = SBM_loader.load_temporarl_edgelist(edgefile, draw=draw)
    print (len(G_times))


    if (normalized):
        p=-10 
        print("compute normalized Laplacian")
        Temporal_eigenvalues, activity_vecs = find_NL_eigenvalues(G_times, max_size, num_eigen, p, add_shift=add_shift, directed=directed, top=True)
    else:
        print("compute unnormalized Laplacian")
        (Temporal_eigenvalues, activity_vecs) = SVD_perSlice(G_times, max_size, directed=directed, num_eigen=num_eigen, top=top)
    print (len(Temporal_eigenvalues))
    normal_util.save_object(Temporal_eigenvalues, fname + ".pkl")





def main():
    print ("hi")



if __name__ == "__main__":
    main()
