import sys
sys.path.append('../')
from os import path
import os
import networkx as nx
import math
import numpy as np
from numpy import linalg as LA
from subroutines import compute_SVD, Anomaly_Detection
from scipy.sparse.linalg import svds, eigsh
from scipy.linalg import fractional_matrix_power
from scipy import sparse
from datasets import SBM_loader
from scipy.spatial.distance import euclidean

import matplotlib.pyplot as plt
from util import normal_util


def compute_accuracy(anomalies, real_events):

    correct = 0
    for anomaly in anomalies:
        if anomaly in real_events:
            correct = correct + 1

    return (correct/len(real_events))


def add_diagonal_shift(A, p):
    #add a small shift for 0
    if (p==0):
        shift = 10.0**(-6)
        np.fill_diagonal(A, A.diagonal() + shift)


    #do not shift for positive power
    elif (p>0):
        return A

    #shift for negative power
    else:
        shift = math.log(1 + abs(p))
        np.fill_diagonal(A, A.diagonal()+ shift)

    return A




def scalar_power_mean(v_s, p):
    v_sum = 0
    for v in v_s:
        v_sum = v_sum + v**p
    v_sum = v_sum / len(v_s)
    v_sum = v_sum**(1/p)
    return v_sum


'''
take the matrix power and find the eigenvalue
for the normalized laplacian matrix
'''
def find_eigenvalues(G_times, max_size, num_eigen, p, add_shift=True, directed=True, top=True):
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


def find_power_mean_eigenvalues(fnames, outName, num_eigen, p):
    num_nodes = 500
    num_times = 0
    directed = False
    add_shift = True
    all_eigenvals = []  #3 views
    for fname in fnames:
        edgefile = "datasets/multi_SBM/" + fname + ".txt"
        if (not os.path.isfile(edgefile)):
            edgefile = fname + ".txt"
        G_times = SBM_loader.load_temporarl_edgelist(edgefile)
        num_times = len(G_times)
        Temporal_eigenvalues, activity_vecs = find_eigenvalues(G_times, num_nodes, num_eigen, p, add_shift=add_shift, directed=directed, top=True)
        normal_util.save_object(Temporal_eigenvalues, fname + "NL.pkl")
        all_eigenvals.append(Temporal_eigenvalues)

    power_vals = []

    for j in range(len(all_eigenvals[0])):
        v_s = []
        for i in range(len(all_eigenvals)):
            v_s.append(all_eigenvals[i][j])
        power_vals.append(scalar_power_mean(v_s, p))

    normal_util.save_object(power_vals, outName + str(num_eigen) + "power" + str(p) + ".pkl")

def naive_powermean(views, max_size, num_eigen, p, add_shift=True, directed=True, top=True):
    Temporal_eigenvalues = []
    activity_vecs = []  #eigenvector of the largest eigenvalue

    FirstView = True
    Lp = 0

    for G in views:
        if (len(G) < max_size):
            for i in range(len(G), max_size):
                G.add_node(-1 * i)      #add empty node with no connectivity (zero padding)

        L = nx.linalg.normalized_laplacian_matrix(G, weight='weight')
        L = L.todense()

        if (add_shift):
            L = add_diagonal_shift(L, p)

        if (FirstView):
            Lp = LA.matrix_power(L, p)
            FirstView = False
        
        else:
            Lp = Lp + LA.matrix_power(L, p)


    Lp = Lp / len(views)
    Lp = fractional_matrix_power(L, 1 / p)
    Lp = sparse.csr_matrix(Lp)
    Lp = Lp.asfptype()

    if (top):
        which="LM"
    else:
        which="SM"

    u, s, vh = svds(Lp, k=num_eigen, which=which)
    #u, s, vh = svds(L, k=num_eigen, which=which)
    vals = s
    vecs = u
    max_index = list(vals).index(max(list(vals)))
    activity_vecs.append(np.asarray(vecs[max_index]))
    Temporal_eigenvalues.append(np.asarray(vals))

    return (Temporal_eigenvalues, activity_vecs)


'''
actually computing the power mean laplacian, direct implementation
'''
def direct_power_mean_eigenvalues(fnames, outName, num_eigen, p):
    num_nodes = 500
    directed = False
    Lp_eigenvals = []
    times_views = []
    for fname in fnames:
        G_times = SBM_loader.load_temporarl_edgelist("datasets/multi_SBM/" + fname + ".txt")
        times_views.append(G_times)

    counter = 0
    for i in range(len(times_views[0])):
        views = []
        for j in range(len(times_views)):
            views.append(times_views[j][i])
        Temporal_eigenvalues, activity_vecs = naive_powermean(views, num_nodes, num_eigen, p, add_shift=True, directed=False, top=True)
        Lp_eigenvals.append(Temporal_eigenvalues)
        print ("processing " + str(counter), end="\r")
        counter = counter + 1

    normal_util.save_object(Lp_eigenvals, outName + str(num_eigen) + "power" + str(p) + ".pkl")

def power_of_a_matrix(H,p):
    w, v = LA.eig(H)
    order = np.argsort(w)
    vals = w[order]
    vecs = v[order]
    zs = np.zeros(vecs.shape)
    D = np.fill_diagonal(zs, vals)
    V = vecs

    Dp = LA.matrix_power(D, p)
    out = V * np.diag(Dp) * V.T
    out = 0.5 * (out + out.T)
    return out


def main():
    print ("hi")
    
if __name__ == "__main__":
    main()
