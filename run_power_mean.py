
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from util import power_mean
from subroutines import Anomaly_Detection
import os
import argparse

def run_powermean(num_eigen, fnames, direct=False, out_dir=None, p=-10):
    if (out_dir is None):
        outName = "PML"
    else:
        outName = out_dir + "PML"

    if (direct):
        power_mean.direct_power_mean_eigenvalues(fnames, outName, num_eigen, p)
    else:
        num_eigen = 498
        power_mean.find_power_mean_eigenvalues(fnames, outName, num_eigen, p)
    eigen_file = outName + str(num_eigen) + "power" + str(p)
    _, accu = Anomaly_Detection.multi_SBM(eigen_file)

    return accu




def main():
    parser = argparse.ArgumentParser(description='run power mean methods')
    parser.add_argument("-n",'--num', type=int, default=499,
                    help="number of eigenvalues to compute")
    parser.add_argument('-v1','--view1', 
                    help='Description for view1', required=True)
    parser.add_argument('-v2','--view2', 
                    help='Description for view2', required=True)
    parser.add_argument('-v3','--view3', 
                    help='Description for view3', required=False, default="no")
    parser.add_argument('--direct', dest='direct', action='store_true', help="To compute direct powermean singular values")
    parser.set_defaults(direct=False)
    args = vars(parser.parse_args())
    if (args["view3"] == "no"):
        fnames = [args["view1"], args["view2"]]
    else:
        fnames = [args["view1"], args["view2"], args["view3"]]

    run_powermean(args["num"], fnames, args["direct"])



if __name__ == "__main__":
    main()
