from datasets.multi_SBM import case2_generator, limit_generator, AB_generator
from subroutines import compute_SVD, Anomaly_Detection, naive_LAD, compute_CPD
from run_power_mean import run_powermean
import numpy as np
import os 
import argparse

def run_CPD_trial(exp, trials, out, c_out, num_views, noise, rank=30):
    out_dir = "output/" + out + "/"
    CPD_scores = []

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print ("output to " + str(out_dir))
    if (exp == "limit"):
        print (" in the SBM limit experiment, this is using cout " + str(c_out))

    seeds = list(range(0, trials))
    outnames = []

    for k in range(num_views):
        outnames.append(out_dir+'view_'+str(k))

    for seed in seeds:
        trial_seeds = list(range(seed*num_views, (seed+1)*num_views,1))

        #first generate the edgelists with set random seeds
        if (exp == "noisySBM"):
            #generate the edgelists
            alpha = 1.0
            inter_prob = 0.004  #c_out=2
            intra_prob = 0.024  #c_in=12
            increment = 0.008   #4
            noises = [noise]*num_views
            for i in range(len(trial_seeds)):
                case2_generator.generate_event_change(inter_prob, intra_prob, alpha, increment, noise_p=noises[i], seed=trial_seeds[i], outname=outnames[i])

        elif (exp == "limit"):
            alpha = 1.0
            c_in = 12
            num_nodes = 500
            intra_prob = c_in / num_nodes
            inter_prob = c_out / num_nodes
            noise_p = 0.0
            for i in range(len(trial_seeds)):
                limit_generator.generate_change_point(inter_prob, intra_prob, alpha, noise_p=noise_p, seed=trial_seeds[i], outname=outnames[i])

        elif (exp == "AB"):
            alpha = 1.0
            m_list = np.arange(1,9,1)
            m_list = list(m_list)
            for i in range(len(trial_seeds)):
                AB_generator.generate_AB_ChangePoint(m_list, alpha, n=500, seed=trial_seeds[i], outname=outnames[i])


        else:
            print ("not automated yet")

        Sparse=True
        compute_CPD.find_synthetic_factors(outnames, "CPD", rank, Sparse=Sparse, normalized_laplacian=False, normalized_adjacency=False)
        _, accu = Anomaly_Detection.multi_SBM("CPD"+str(rank), verbose=False)
        CPD_scores.append(accu)

        with open(out + '.txt', 'w') as f:
            f.write("CPD scores are" + '\n')
            f.write(str(CPD_scores) + '\n' )
        print ("write complete")

    print ("ran " + str(trials) + " trials")
    CPD_scores = np.array(CPD_scores)
    print ("average CPD score is " + str(np.mean(CPD_scores)))
    print ("std CPD score is " + str(np.std(CPD_scores)))
    print ("----------------------------------------------------")

    with open(out + '.txt', 'a') as f:
        f.write('\n'+'\n'+'\n'+'\n'+'\n')
        f.write("average CPD score is " + str(np.mean(CPD_scores))+'\n')
        f.write("std CPD score is " + str(np.std(CPD_scores))+'\n')
        f.write("----------------------------------------------------"+'\n')












def run_trial(exp, trials, out, c_out, num_views, noise, power):
    print ("multiCPD uses power " + str(power))

    out_dir = "output/" + out + "/"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print ("output to " + str(out_dir))
    if (exp == "limit"):
        print (" in the SBM limit experiment, this is using cout " + str(c_out))

    seeds = list(range(0, trials))

    multiCPD_scores = []
    maxLAD_scores = []
    meanLAD_scores = []
    NLmaxLAD_scores = []
    NLmeanLAD_scores =[]
    LADview_scores = []
    NLview_scores = []
    outnames = []
    NL_names = []
    for k in range(num_views):
        outnames.append(out_dir+'view_'+str(k))
        NL_names.append(out_dir+'view_'+str(k)+'NL')
        LADview_scores.append([])
        NLview_scores.append([])

    for seed in seeds:
        trial_seeds = list(range(seed*num_views, (seed+1)*num_views,1))

        #first generate the edgelists with set random seeds
        if (exp == "noisySBM"):
            #generate the edgelists
            alpha = 1.0
            inter_prob = 0.004      #c_out = 2
            intra_prob = 0.024      #c_in = 12
            increment = 0.008        #increment = 4
            noises = [noise]*num_views
            for i in range(len(trial_seeds)):
                case2_generator.generate_event_change(inter_prob, intra_prob, alpha, increment, noise_p=noises[i], seed=trial_seeds[i], outname=outnames[i])

        elif (exp == "limit"):
            alpha = 1.0
            c_in = 12
            num_nodes = 500
            intra_prob = c_in / num_nodes
            inter_prob = c_out / num_nodes
            noise_p = 0.0
            for i in range(len(trial_seeds)):
                limit_generator.generate_change_point(inter_prob, intra_prob, alpha, noise_p=noise_p, seed=trial_seeds[i], outname=outnames[i])

        elif (exp == "AB"):
            alpha = 1.0
            m_list = np.arange(1,9,1)
            m_list = list(m_list)
            for i in range(len(trial_seeds)):
                AB_generator.generate_AB_ChangePoint(m_list, alpha, n=500, seed=trial_seeds[i], outname=outnames[i])


        else:
            print ("not automated yet")



        print ("SBM generation complete")
        #run power mean laplacian to also get normalized laplacian singular values
        multiCPD_scores.append(run_powermean(499, outnames, direct=False, out_dir=out_dir, p=power))

        for i in range(len(outnames)):
            compute_SVD.compute_multiSBM_SVD(outnames[i], num_eigen=499, top=True, adj=False, normalized=False)
            _, accu = Anomaly_Detection.multi_SBM(outnames[i], verbose=True)
            LADview_scores[i].append(accu)
            _, NL_accu = Anomaly_Detection.multi_SBM(NL_names[i], verbose=False)
            NLview_scores[i].append(NL_accu)


        maxLAD_accu, meanLAD_accu = naive_LAD.maxmeanLAD(outnames, verbose=False)
        maxLAD_scores.append(maxLAD_accu)
        meanLAD_scores.append(meanLAD_accu)

        NLmaxLAD_accu, NLmeanLAD_accu = naive_LAD.maxmeanLAD(NL_names, verbose=False)
        NLmaxLAD_scores.append(NLmaxLAD_accu)
        NLmeanLAD_scores.append(NLmeanLAD_accu)
        print ("trial " + str(seed) + " complete")


        with open(out + '.txt', 'w') as f:
            f.write("multiCPD_scores are" + '\n')
            f.write(str(multiCPD_scores) + '\n' )

            f.write("maxLAD_scores are" + '\n')
            f.write(str(maxLAD_scores) + '\n' )

            f.write("meanLAD_scores are" + '\n')
            f.write(str(meanLAD_scores) + '\n' )

            f.write("NLmaxLAD_scores are" + '\n')
            f.write(str(NLmaxLAD_scores) + '\n' )

            f.write("NLmeanLAD_scores are" + '\n')
            f.write(str(NLmeanLAD_scores) + '\n' )
        print ("write complete")

    print ("ran " + str(trials) + " trials")
    multiCPD_scores = np.array(multiCPD_scores)
    print ("average multiCPD score is " + str(np.mean(multiCPD_scores)))
    print ("std multiCPD score is " + str(np.std(multiCPD_scores)))
    print ("----------------------------------------------------")

    maxLAD_scores = np.array(maxLAD_scores)
    print ("average max LAD score is " + str(np.mean(maxLAD_scores)))
    print ("std max LAD score is " + str(np.std(maxLAD_scores)))
    print ("----------------------------------------------------")

    meanLAD_scores = np.array(meanLAD_scores)
    print ("average mean LAD score is " + str(np.mean(meanLAD_scores)))
    print ("std mean LAD score is " + str(np.std(meanLAD_scores)))
    print ("----------------------------------------------------")


    NLmaxLAD_scores = np.array(NLmaxLAD_scores)
    print ("average NLmaxLAD_score is " + str(np.mean(NLmaxLAD_scores)))
    print ("std NLmaxLAD_score score is " + str(np.std(NLmaxLAD_scores)))
    print ("----------------------------------------------------")

    NLmeanLAD_scores = np.array(NLmeanLAD_scores)
    print ("average NLmeanLAD_scores is " + str(np.mean(NLmeanLAD_scores)))
    print ("std NLmeanLAD_scores is " + str(np.std(NLmeanLAD_scores)))
    print ("----------------------------------------------------")

    LAD_means = []
    LAD_stds = []
    NLLAD_means = []
    NLLAD_stds = []
    for i in range(num_views):
        LAD_scores = np.array(LADview_scores[i])
        LAD_means.append(np.mean(LAD_scores))
        LAD_stds.append(np.std(LAD_scores))

        NLLAD_scores = np.array(NLview_scores[i])
        NLLAD_means.append(np.mean(NLLAD_scores))
        NLLAD_stds.append(np.std(NLLAD_scores))

    with open(out + '.txt', 'a') as f:
        f.write('\n'+'\n'+'\n'+'\n'+'\n')
        f.write("average multiCPD score is " + str(np.mean(multiCPD_scores))+'\n')
        f.write("std multiCPD score is " + str(np.std(multiCPD_scores))+'\n')
        f.write("----------------------------------------------------"+'\n')
        f.write("average max LAD score is " + str(np.mean(maxLAD_scores))+'\n')
        f.write("std max LAD score is " + str(np.std(maxLAD_scores))+'\n')
        f.write("----------------------------------------------------"+'\n')
        f.write("average mean LAD score is " + str(np.mean(meanLAD_scores))+'\n')
        f.write("std mean LAD score is " + str(np.std(meanLAD_scores))+'\n')
        f.write("----------------------------------------------------"+'\n')
        f.write("average NLmaxLAD_score is " + str(np.mean(NLmaxLAD_scores))+'\n')
        f.write("std NLmaxLAD_score score is " + str(np.std(NLmaxLAD_scores))+'\n')
        f.write("----------------------------------------------------"+'\n')
        f.write("average NLmeanLAD_scores is " + str(np.mean(NLmeanLAD_scores))+'\n')
        f.write("std NLmeanLAD_scores is " + str(np.std(NLmeanLAD_scores))+'\n')
        f.write("----------------------------------------------------"+'\n')

        LADidx = LAD_means.index(max(LAD_means))
        NLLADidx = NLLAD_means.index(max(NLLAD_means))

        f.write("best LAD single view score is " + str(max(LAD_means))+'\n')
        f.write("best LAD single view std is" + str(LAD_stds[LADidx])+'\n')
        f.write("----------------------------------------------------"+'\n')
        f.write("best NL LAD single view score is " + str(max(NLLAD_means))+'\n')
        f.write("best NL LAD single view std is" + str(NLLAD_stds[NLLADidx])+'\n')
        f.write("----------------------------------------------------"+'\n')









def main():

    parser = argparse.ArgumentParser(description='run LAD on files')
    parser.add_argument('-e','--exp', 
                    help='which experiment to run', required=True)
    parser.add_argument("-t",'--trials', type=int, default=30,
                    help="number of trials to run")
    parser.add_argument("-v",'--views', type=int, default=3,
                    help="number of views per trial")
    parser.add_argument("-o",'--out', type=str, default="output",
                    help="output csv name")
    parser.add_argument("-c",'--cout', type=float, default=6,
                    help="what c_out to use in the SBM limit experiment")
    parser.add_argument("-n",'--noise', type=float, default=0.024,
                    help="what noise ratio to use in the noisy SBM experiment")
    parser.add_argument("-p",'--power', type=int, default=-10,
                    help="what power to use for scalar power mean, recommend to use the default")
    parser.add_argument('--CPD', dest='CPD', action='store_true', help="Run the trial for the TENSORSPLAT / CPD decomposition baseline")
    parser.set_defaults(CPD=False)

    args = vars(parser.parse_args())

    if not os.path.exists("output"):
        os.makedirs("output")

    if (args["CPD"]):
        run_CPD_trial(args["exp"], args["trials"], args["out"], args["cout"],args["views"],args["noise"])
    else:
        run_trial(args["exp"], args["trials"], args["out"], args["cout"],args["views"],args["noise"], args["power"])




    


if __name__ == "__main__":
    main()
