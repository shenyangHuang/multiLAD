import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import networkx as nx
from scipy import sparse
import pylab as plt
import dateutil.parser as dparser
import re

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, 2)

def load_object(filename):
    output = 0
    with open(filename, 'rb') as fp:
        output = pickle.load(fp)
    return output

def edgelist2numEdge(edgelist):
    num_edges = len(set(edgelist))
    return num_edges

def edgelist2weights(edgelist):
    unique_edges = list(set(edgelist))
    weights = [0]*len(unique_edges)
    for edge in edgelist:
        weights[unique_edges.index(edge)] += 1
    return weights

def edgelist2degrees(edgelist):
    unique_nodes = []
    for (u,v) in edgelist:
        if u not in unique_nodes:
            unique_nodes.append(u)
        if v not in unique_nodes:
            unique_nodes.append(v)
    
    degrees = [0]*len(unique_nodes)
    for (u,v) in edgelist:
        degrees[unique_nodes.index(u)] +=1
        degrees[unique_nodes.index(v)] +=1
    return degrees








