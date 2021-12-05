'''
case2
'''
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
            #randomly decide if remain the same or resample
            #remain the same if prob > alpha
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
            #flip the existence of an edge
            prob = random.uniform(0, 1)
            if (prob <= p):
                if (G_t.has_edge(i,j)):
                    G_t.remove_edge(i, j)
                else:
                    G_t.add_edge(i,j)
    return G_t


def generate_event_change(inter_prob, intra_prob, alpha, increment, noise_p=0, seed=1, outname="none"):

    random.seed(seed)       
    np.random.seed(seed)

    if (outname != "none"):
        fname = outname+".txt"
    else:
        fname = "eventChange_"+ str(inter_prob)+ "_"+ str(intra_prob) + "_" + str(noise_p) + "_" + str(alpha) + ".txt"

    cps_sizes = []
    cps_probs = []


    sizes_1 = [250,250]
    probs_1 = construct_SBM_block(sizes_1, inter_prob, intra_prob)

    sizes_2 = [125,125,125,125]
    probs_2 = construct_SBM_block(sizes_2, inter_prob, intra_prob)


    sizes_3 = [50]*10
    probs_3 = construct_SBM_block(sizes_3, inter_prob, intra_prob)

    list_sizes = []
    list_sizes.append(sizes_1)
    list_sizes.append(sizes_2)
    list_sizes.append(sizes_3)


    list_probs = []
    list_probs.append(probs_1)
    list_probs.append(probs_2)
    list_probs.append(probs_3)

    list_idx = 1
    isEvent = True 


    cps=[15,30,60,75,90,105,135]
    sizes = sizes_2
    probs = probs_2

    
    maxt = 150
    G_0=nx.stochastic_block_model(sizes, probs)
    G_0 = nx.Graph(G_0)
    G_t = G_0
    G_times = []
    G_times.append(G_t)


 
    for t in range(maxt):
        if (t in cps):

            if (isEvent):

                copy_probs = copy.deepcopy(probs)
                for i in range(len(copy_probs)):
                    for j in range(len(copy_probs[0])):
                        if (copy_probs[i][j] < intra_prob):
                            copy_probs[i][j] = inter_prob + increment

                G_t = SBM_snapshot(G_t, alpha, sizes, np.asarray(copy_probs))
                if (noise_p > 0):
                    G_t = add_noise(noise_p, G_t)
                G_times.append(G_t)
                print ("generating " + str(t), end="\r")
                isEvent = False
            else:
                if ((list_idx+1) > len(list_sizes)-1):
                    list_idx = 0
                else:
                    list_idx = list_idx + 1
                sizes = list_sizes[list_idx]
                probs = list_probs[list_idx]
                G_t = SBM_snapshot(G_t, alpha, sizes, probs)
                if (noise_p > 0):
                    G_t = add_noise(noise_p, G_t)
                G_times.append(G_t)
                print ("generating " + str(t), end="\r")
                isEvent = True

        else:
            G_t = SBM_snapshot(G_t, alpha, sizes, probs)
            if (noise_p > 0):
                G_t = add_noise(noise_p, G_t)
            G_times.append(G_t)
            print ("generating " + str(t), end="\r")

    to_edgelist(G_times, fname)


def main():
    print ("hi")
    


if __name__ == "__main__":
    main()
