## MultiCPD
official python repository for "Official repository for "Laplacian Change Point Detection for Single and Multi-view Dynamic Graphs".

<a href="https://arxiv.org/abs/2302.01204">
    <img src="https://img.shields.io/badge/arXiv-pdf-yellowgreen">
 </a>


# How to run

Case 1: python run_trials.py -e limit -t 30  -v 3 -c 6 -o cout6

Case 2: python run_trials.py -e noisySBM -t 30  -v 3 -n 0.1 -o noise01

Case 3: python run_trials.py -e limit -t 30  -v 4 -c 6 -o cout6_4views

Case 4: python run_trials.py -e AB -t 30 -v 3 -o BA_3views

real world: go to real_world folder

# requirements

see install.sh

1.python 3.8

2.numpy 1.18.1

3.tensorly 0.4.5

4.scipy 1.4.1

5.scikit-learn 0.22.1

6.networkx 2.4

7.sparse 0.9.1

8.matplotlib 3.3.1



