import numpy as np
import networkx as nx
from matplotlib import pyplot, patches
import pylab as plt
import dateutil.parser as dparser
import re

def load_temporarl_edgelist(fname, draw=False):
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()
    cur_t = 0

    '''
    t u v
    '''
    G_times = []
    G = nx.Graph()

    for i in range(0, len(lines)):
        line = lines[i]
        values = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+",line)
        t = int(values[0])
        u = int(values[1])
        v = int(values[2])
        #start a new graph with a new date
        if (t != cur_t):
            G_times.append(G)   #append old graph
            if (draw):
                if (t in [1,17,32,62,77,92,107,137]):
                    draw_adjacency_matrix(G, t,  node_order=None, partitions=[], colors=[])
            G = nx.Graph()  #create new graph
            cur_t = t 
        G.add_edge(u, v) 
    G_times.append(G)
    print ("maximum time stamp is " + str(len(G_times)))
    return G_times


def draw_adjacency_matrix(G, t,  node_order=None, partitions=[], colors=[]):
    nodelist = list(range(0,500))
    adjacency_matrix = nx.to_numpy_matrix(G, dtype=np.bool, nodelist=nodelist)

    #Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(5, 5)) # in inches
    pyplot.imsave(str(t)+".pdf", adjacency_matrix,
                  cmap="Greys")
    pyplot.close()
    assert len(partitions) == len(colors)
    ax = pyplot.gca()
    for partition, color in zip(partitions, colors):
        current_idx = 0
        for module in partition:
            ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                          len(module), # Width
                                          len(module), # Height
                                          facecolor="none",
                                          edgecolor=color,
                                          linewidth="1"))
            current_idx += len(module)