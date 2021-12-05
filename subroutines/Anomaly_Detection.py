import sys
sys.path.append('../')

import numpy as np
import networkx as nx
import sparse
from sklearn.preprocessing import normalize
from util import normal_util
from subroutines import metrics 
import pylab as plt
import argparse

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.xmargin'] = 0

from scipy.stats import spearmanr
from numpy import linalg as LA

def find_anomalies(z_scores, percent_ranked, initial_window):
    z_scores = np.array(z_scores)
    for i in range(initial_window+1):
        z_scores[i] = 0        
    num_ranked = int(round(len(z_scores) * percent_ranked))
    outliers = z_scores.argsort()[-num_ranked:][::-1]
    outliers.sort()
    return outliers


def average_typical_behavior(context_vecs):
    avg = np.mean(context_vecs, axis=0)
    return avg

'''
find the left singular vector of the activity matrix
'''
def principal_vec_typical_behavior(context_vecs):
    activity_matrix = context_vecs.T
    u, s, vh = np.linalg.svd(activity_matrix, full_matrices=False)
    return u[:,0]



def compute_Z_score(cur_vec, typical_vec):
    cosine_similarity = abs(np.dot(cur_vec, typical_vec) / LA.norm(cur_vec) / LA.norm(typical_vec))
    z = (1 - cosine_similarity)
    return z


def rank_outliers(x, window=5, initial_period=10):
    x = np.asarray(x)
    mv_std = []

    for i in range(0, initial_period):
        mv_std.append(0)

    for i in range(initial_period,len(x)):
        #compute moving average until this point
        avg = np.mean(x[i-window:i])
        std = np.std(x[i-window:i])
        if (std == 0):
            std = 1
        mv_std.append(abs(x[i]-avg) / std)
        
    mv_std = np.asarray(mv_std)
    outlier_ranks = mv_std.argsort()

    return outlier_ranks


def set_non_negative(z_scores):
    for i in range(len(z_scores)):
        if (z_scores[i] < 0):
            z_scores[i] = 0
    return z_scores


def plot_anomaly_score(fname, scores, score_labels, events, real_events, initial_window=10):
    for k in range(len(scores)):
        scores[k] = set_non_negative(scores[k])

    for score in scores:
        for l in range(initial_window+1):
            score[l] = 0

    max_time = len(scores[0])
    t = list(range(0, max_time))
    plt.rcParams.update({'figure.autolayout': True})
    plt.rc('xtick')
    plt.rc('ytick')
    fig = plt.figure(figsize=(4, 2))
    ax = fig.add_subplot(1, 1, 1)
    colors = ['#fdbb84', '#43a2ca', '#bc5090', '#e5f5e0','#fa9fb5','#c51b8a', '#bf812d', '#35978f','#542788','#b2182b', '#66c2a5', '#fb9a99','#e31a1c','#ff7f00','#8dd3c7']
    for i in range(len(scores)):
        ax.plot(t, scores[i], color=colors[i], ls='solid', lw=0.5, label=score_labels[i])

    
    for event in events:
        max_score = 0
        for i in range(len(scores)):
            if scores[i][event] > max_score:
                max_score = scores[i][event]


        plt.annotate(str(event), # this is the text
                 (event, max_score), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(8,0), # distance from text to points (x,y)
                 ha='center',
                 fontsize=4) # horizontal alignment can be left, right or center

    addLegend = True

    for event in events:
        max_score = 0
        for i in range(len(scores)):
            if scores[i][event] > max_score:
                max_score = scores[i][event]
        if (addLegend):
            ax.plot( event, max_score, marker="*", mew=0.1, color='#de2d26', ls='solid', lw=0.5, label="detected anomalies")
            addLegend=False
        else:
            ax.plot( event, max_score, marker="*", mew=0.1, color='#de2d26', ls='solid', lw=0.5)

    plt.xticks(fontsize=4)
    plt.yticks(fontsize=4)
    ax.set_xlabel('time steps', fontsize=8)
    ax.set_ylabel('anomaly score', fontsize=8)
    plt.legend(fontsize=4)
    plt.savefig(fname +'anomalyScores.pdf')

    print ("plotting anomaly scores complete")
    plt.close()


def plot_anomaly_and_spectro(fname, scores, score_labels, events, eigen_file, initial_window=10, l2normed=False):


    labels = list(range(0,len(scores[0]),1))
    for k in range(len(scores)):
        scores[k] = set_non_negative(scores[k])

    '''
    scores at initial windows is 0
    '''
    for score in scores:
        for l in range(initial_window+1):
            score[l] = 0


    fig, axs = plt.subplots(2)
    plt.rcParams.update({'figure.autolayout': True})
    diag_vecs = normal_util.load_object(eigen_file)

    '''
    plot the l2 normalized singular values
    '''
    if (l2normed):
        diag_vecs = np.asarray(diag_vecs)
        diag_vecs = diag_vecs.real
        diag_vecs= normalize(diag_vecs, norm='l2')

    diag_vecs = np.transpose(np.asarray(diag_vecs))     #let time be x-axis
    diag_vecs = np.flip(diag_vecs, 0)


    max_time = len(scores[0])
    t = list(range(0, max_time))
    colors = ["red", "blue"]
    for i in range(len(scores)):
        axs[0].plot(t, scores[i], color=colors[i], ls='solid', label=score_labels[i])

    for event in events:
        max_score = 0
        for i in range(len(scores)):
            if scores[i][event] > max_score:
                max_score = scores[i][event]
        axs[0].annotate(str(labels[event]), # this is the text
                 (event, max_score), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-12), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

    addLegend = True

    for event in events:
        max_score = 0
        for i in range(len(scores)):
            if scores[i][event] > max_score:
                max_score = scores[i][event]
        if (addLegend):
            axs[0].plot( event, max_score, marker="*", color='#de2d26', ls='solid', label="detected anomalies")
            addLegend=False
        else:
            axs[0].plot( event, max_score, marker="*", color='#de2d26', ls='solid')



    axs[0].set_xlabel('Time Point', fontsize=8)
    axs[0].set_ylabel('anomaly score')
    axs[0].legend(fontsize=6)


    axs[1].set_xlabel('Time Point', fontsize=8)
    axs[1].set_ylabel('spectrum index')

    im = axs[1].imshow(diag_vecs, aspect='auto')


    plt.savefig(fname+'Spectro.pdf')
    plt.close()


def change_detection_many_windows(spectrums, windows,initial_window, principal=True, percent_ranked=0.05, difference=False):
    z_scores = []
    for i in range(len(windows)):
        z_scores.append([])

    counter = 0
    for j in range(0, initial_window):
        for i in range(0, len(z_scores)):
            z_scores[i].append(0)

    
    #compute the z score for each signature vector after initial window
    #1. find typical behavior
    #2. compute anomaly score
    for i in range(initial_window, len(spectrums)):

        for j in range(len(windows)):

            if (windows[j] > i):
                z_scores[j].append(0)
            else:
                #1. compute short term window first
                if (principal):
                    typical_vec = principal_vec_typical_behavior(spectrums[i-windows[j]:i])
                else:
                    typical_vec = average_typical_behavior(spectrums[i-windows[j]:i])
                cur_vec = spectrums[i]
                z = compute_Z_score(cur_vec, typical_vec)
                z_scores[j].append(z)

    #check the change in z score instead
    if (difference):
        for i in range(len(windows)):
            z_scores[i] = difference_score(z_scores[i])

    z_overall = [0] * len(z_scores[0])
    for j in range(len(z_scores[0])):
        if (j < initial_window):
            continue
        else:
            z_overall[j] = max([x[j] for x in z_scores])

    z_overall = np.array(z_overall)
    for i in range(initial_window+1):
        z_overall[i] = 0        

    num_ranked = int(round(len(z_overall) * percent_ranked))
    outliers = z_overall.argsort()[-num_ranked:][::-1]
    outliers.sort()
    return (z_overall,z_scores,outliers)






def difference_score(z_scores):
    z = []
    for i in range(len(z_scores)):
        if (i==0):
            z.append(z_scores[0])
        else:
            z.append(z_scores[i] - z_scores[i-1])
    return z






def detection_many_windows(fname, real_events, windows, eigen_file, timestamps=195, percent_ranked=0.05, difference=False, normalized=True,symmetric=False, verbose=True):
    principal = True



    spectrums = normal_util.load_object(eigen_file)
    if (len(spectrums) == 2):
        spectrums = spectrums[1][1]


    if (type(spectrums) == sparse._coo.core.COO):
        spectrums = spectrums.todense()
    spectrums = np.asarray(spectrums).real
    spectrums = spectrums.reshape((timestamps,-1))
    
    if (normalized):
        spectrums= normalize(spectrums, norm='l2')


    initial_window = max(windows)

    if (verbose):
        print ("window sizes are :")
        print (windows)
        print ("initial window is " + str(initial_window))
        print (spectrums.shape)

    if (not symmetric):
        (z_overall, z_scores, anomalies) = change_detection_many_windows(spectrums, windows,  initial_window, principal=principal, percent_ranked=percent_ranked, difference=difference)
    else:
        (z_overall, z_scores, anomalies) = change_detection_symmetric_windows(spectrums, windows,  principal=principal, percent_ranked=percent_ranked, difference=difference)

    if (verbose):
        print ("found anomalous time stamps are")
        print (anomalies)

    events = anomalies
    scores = z_scores
    score_labels = ["window size "+ str(window) for window in windows]
    if (verbose):
        plot_anomaly_and_spectro(fname, scores, score_labels, events, eigen_file, initial_window=initial_window)
    return z_overall



def multi_SBM(fname, verbose=True):
    timestamps = 151
    percent_ranked= 0.045
    real_events = [16,31,61,76,91,106,136]
    eigen_file = fname + ".pkl"
    difference=True
    symmetric=False
    normalized=True


    windows = [5,10]
    z_scores = detection_many_windows(fname, real_events, windows, eigen_file, timestamps=timestamps, percent_ranked=percent_ranked, difference=difference, normalized=normalized, symmetric=symmetric, verbose=verbose)
    anomalies = find_anomalies(z_scores, percent_ranked, max(windows))

    accu = metrics.compute_accuracy(anomalies, real_events)
    print ("combined accuracy is " + str(accu))
    z_scores = set_non_negative(z_scores)

    return z_scores, accu




def main():

    parser = argparse.ArgumentParser(description='anomaly detection on signature vectors')
    parser.add_argument('-f','--file', 
                    help='Description for foo argument', required=True)
    args = vars(parser.parse_args())
    multi_SBM("../" + args["file"])

if __name__ == "__main__":
    main()
