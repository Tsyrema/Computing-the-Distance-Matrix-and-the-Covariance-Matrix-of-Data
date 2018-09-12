# test sklearn data

from ComputeMatrices import compute_distance_naive, \
    compute_distance_smart, compute_correlation_naive, \
    compute_correlation_smart
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer, load_digits



# testing
def main():
    iris = load_iris()
    X = iris.data
    bc = load_breast_cancer()
    Y = bc.data
    digits = load_digits()
    Z = digits.data

    # Distance
    # X - iris
    st = time.time()
    dist_loop = compute_distance_naive(X)
    et = time.time()
    iris_dist_loop = et - st  # time difference
    print "iris_dist_loop: ", iris_dist_loop

    st = time.time()
    dist_cool = compute_distance_smart(X)
    et = time.time()
    iris_dist_cool = et - st
    print "iris_dist_cool: ", iris_dist_cool

    # breast_cancer
    st = time.time()
    dist_loop = compute_distance_naive(Y)
    et = time.time()
    bc_dist_loop = et - st  # time difference
    print "bc_dist_loop: ", bc_dist_loop

    st = time.time()
    dist_cool = compute_distance_smart(Y)
    et = time.time()
    bc_dist_cool = et - st
    print "bc_dist_cool: ", bc_dist_cool


    #digits
    st = time.time()
    dist_loop = compute_distance_naive(Z)
    et = time.time()
    digits_dist_loop = et - st  # time difference
    print "digits_dist_loop: ", digits_dist_loop

    st = time.time()
    dist_cool = compute_distance_smart(Z)
    et = time.time()
    digits_dist_cool = et - st
    print "digits_dist_cool: ", digits_dist_cool

    # Correlation

    # iris
    st = time.time()
    corr_loop = compute_correlation_naive(X)
    et = time.time()
    iris_corr_loop = et - st  # time difference
    print "iris_corr_loop: ", iris_corr_loop


    st = time.time()
    corr_cool = compute_correlation_smart(X)
    et = time.time()
    iris_corr_cool = et - st
    print "iris_corr_coo: ", iris_corr_cool

    #breast_cancer

    st = time.time()
    corr_loop = compute_correlation_naive(Y)
    et = time.time()
    bc_corr_loop = et - st  # time difference
    print "bc_corr_loop: ", bc_corr_loop


    st = time.time()
    corr_cool = compute_correlation_smart(Y)
    et = time.time()
    bc_corr_cool = et - st
    print "bc_corr_cool: ", bc_corr_cool

    #digits
    st = time.time()
    corr_loop = compute_correlation_naive(Z)
    et = time.time()
    digits_corr_loop = et - st  # time difference
    print "digits_corr_loop: ", digits_corr_loop

    st = time.time()
    corr_cool = compute_correlation_smart(Z)
    et = time.time()
    digits_corr_cool = et - st
    print "digits_corr_cool: ", digits_corr_cool

    # data to plot
    n_groups = 3

    distloop = (iris_dist_loop, bc_dist_loop, digits_dist_loop)
    distsmart = (iris_dist_cool, bc_dist_cool, digits_dist_cool)

    corrloop = (iris_corr_loop, bc_corr_loop, digits_corr_loop)
    corrsmart = (iris_corr_cool, bc_corr_cool,digits_corr_cool)


    # create plot
    plt.figure(1)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, distloop, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Loop')

    rects2 = plt.bar(index + bar_width, distsmart, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Smart')

    plt.xlabel('Distance')
    plt.ylabel('Time')
    plt.title('Distance Matrix')
    plt.xticks(index + bar_width, ('Iris', 'BC', 'Digits'))
    plt.legend()

    #plt.show()
    plt.savefig('SKLearnCompareDistance.pdf')
    print "result is written to SKLearnCompareDistance.pdf"

    plt.figure(2)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, corrloop, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Loop')

    rects2 = plt.bar(index + bar_width, corrsmart, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Smart')

    plt.xlabel('Correlation')
    plt.ylabel('Time')
    plt.title('Correlation Matrix')
    plt.xticks(index + bar_width, ('Iris', 'BC', 'Digits'))
    plt.legend()

    # plt.show()
    plt.savefig('SKLearnCompareCorrelation.pdf')
    print "result is written to SKLearnCompareCorrelation.pdf"


if __name__ == "__main__": main()
