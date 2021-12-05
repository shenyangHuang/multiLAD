import sys
from . import Anomaly_Detection
import numpy as np
import networkx as nx
import sparse
from sklearn.preprocessing import normalize
sys.path.append('../')
from util import normal_util
from subroutines import metrics 
import pylab as plt
import argparse
from numpy import linalg as LA

'''
naive aggregation of LAD
find the max /average anomaly score out of all views at each step
input is the computed eigenfiles of each view
'''
def maxmeanLAD(fnames, verbose=True):
    timestamps = 151
    percent_ranked= 0.045
    difference=True
    symmetric=False
    normalized = True
    real_events = [16,31,61,76,91,106,136]
    view_scores = []
    windows = [5,10]


    for fname in fnames:
        eigen_file = fname + ".pkl"
        z_score = Anomaly_Detection.detection_many_windows(fname, real_events, windows, eigen_file, timestamps=timestamps, percent_ranked=percent_ranked, difference=difference, normalized=normalized, symmetric=symmetric, verbose=verbose)
        view_scores.append(z_score)

    view_scores = np.asarray(view_scores)
    max_scores = np.amax(view_scores, axis=0)
    max_anomalies = Anomaly_Detection.find_anomalies(max_scores, percent_ranked, max(windows))
    maxLAD_accu = metrics.compute_accuracy(max_anomalies, real_events)
    if (verbose):
        print ("combined maxLAD accuracy is " + str(maxLAD_accu))
        print ("found anomalies " + str(max_anomalies))

    mean_scores = np.mean(view_scores, axis=0)
    mean_anomalies = Anomaly_Detection.find_anomalies(mean_scores, percent_ranked, max(windows))
    meanLAD_accu = metrics.compute_accuracy(mean_anomalies, real_events)
    if (verbose):
        print ("combined meanLAD accuracy is " + str(meanLAD_accu))
        print ("found anomalies " + str(mean_anomalies))
        scores = [max_scores, mean_scores]
        Anomaly_Detection.plot_anomaly_score("naiveLAD", scores, ["maxLAD", "meanLAD"], max_anomalies, real_events, initial_window=max(windows))

    return maxLAD_accu, meanLAD_accu


