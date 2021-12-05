import numpy as np
import pylab as plt
import random
import collections
import scipy.stats as stats
import dateutil.parser as dparser
import re
import copy
from scipy.stats import powerlaw
from networkx.algorithms.community.quality import modularity
import networkx as nx
from networkx.utils import *
from networkx import generators
import random

def to_edgelist(G_times, outfile):

    outfile = open(outfile,"w")
    tdx = 0
    for G in G_times:
        
        for (u,v) in G.edges:
            outfile.write(str(tdx) + "," + str(u) + "," + str(v) + "\n")
        tdx = tdx + 1
    outfile.close()
    print("write successful")

'''
blocks is an array of sizes
inter is the inter community probability
intra is the intra community probability
'''
def construct_SBM_block(blocks, inter, intra):
    probs = []
    for i in range(len(blocks)):
        prob = [inter]*len(blocks)
        prob[i] = intra
        probs.append(prob)
    return probs


def SBM_snapshot(G_prev, alpha, sizes, probs):

    G_t = G_prev.copy()
    nodelist = list(range(0,sum(sizes)))
    G_new = nx.stochastic_block_model(sizes, probs, nodelist=nodelist)
    n = len(G_t)
    if (alpha == 1.0):
        return G_new

    for i in range(0,n):
        for j in range(i+1,n):
            prob = random.uniform(0, 1)
            if (prob <= alpha):
                if (G_new.has_edge(i,j) and not G_t.has_edge(i, j)):
                    G_t.add_edge(i,j)
                if (not G_new.has_edge(i,j) and G_t.has_edge(i, j)):
                    G_t.remove_edge(i,j)
    return G_t


def add_noise(p, G_t):
    n = len(G_t)
    for i in range(0,n):
        for j in range(i+1,n):
            prob = random.uniform(0, 1)
            if (prob <= p):
                if (G_t.has_edge(i,j)):
                    G_t.remove_edge(i, j)
                else:
                    G_t.add_edge(i,j)
    return G_t



'''
generate just change points
'''

def generate_change_point(inter_prob, intra_prob, alpha, noise_p=0, seed=1, outname="none"):

    random.seed(seed)       
    np.random.seed(seed)

    if (outname != "none"):
        fname = outname+".txt"
    else:
        fname = "CP_"+ str(inter_prob)+ "_"+ str(intra_prob) + "_" + str(noise_p) + "_" + str(alpha) + ".txt"

    cps_sizes = []
    cps_probs = []


    sizes_1 = [250,250]
    probs_1 = construct_SBM_block(sizes_1, inter_prob, intra_prob)

    sizes_2 = [125,125,125,125]
    probs_2 = construct_SBM_block(sizes_2, inter_prob, intra_prob)

    sizes_3 = [83,83,83,83,83,85]
    probs_3 = construct_SBM_block(sizes_3, inter_prob, intra_prob)

    sizes_4 = [50]*10
    probs_4 = construct_SBM_block(sizes_4, inter_prob, intra_prob)

    sizes_5 = [25]*20
    probs_5 = construct_SBM_block(sizes_5, inter_prob, intra_prob)

    com_split=True





    list_sizes = []
    list_sizes.append(sizes_1)
    list_sizes.append(sizes_2)
    list_sizes.append(sizes_3)
    list_sizes.append(sizes_4)
    list_sizes.append(sizes_5)


    list_probs = []
    list_probs.append(probs_1)
    list_probs.append(probs_2)
    list_probs.append(probs_3)
    list_probs.append(probs_4)
    list_probs.append(probs_5)

    
    cps=[15,30,60,75,90,105,135]
    sizes = sizes_1
    probs = probs_1
    list_idx = 0

    
    maxt = 150
    G_0=nx.stochastic_block_model(sizes, probs)
    G_0 = nx.Graph(G_0)
    G_t = G_0
    G_times = []
    if (noise_p > 0):
        G_t = add_noise(noise_p, G_t)
    G_times.append(G_t)


    for t in range(maxt):
        if (t in cps):
            if ((list_idx+1) > len(list_sizes)-1):
                com_split = False   #now merge communities
            if ((list_idx-1) < 0):
                com_split = True   #now split communities
            if (com_split):
                list_idx = list_idx + 1
            else:
                list_idx = list_idx - 1
            sizes = list_sizes[list_idx]
            #print (sizes)
            probs = list_probs[list_idx]
            G_t = SBM_snapshot(G_t, alpha, sizes, probs)
            if (noise_p > 0):
                G_t = add_noise(noise_p, G_t)
            G_times.append(G_t)
            print ("generating " + str(t), end="\r")

        else:
            G_t = SBM_snapshot(G_t, alpha, sizes, probs)
            if (noise_p > 0):
                G_t = add_noise(noise_p, G_t)
            G_times.append(G_t)
            print ("generating " + str(t), end="\r")

    #write the entire history of snapshots
    to_edgelist(G_times, fname)


def main():
    print ("hi")


    


if __name__ == "__main__":
    main()
