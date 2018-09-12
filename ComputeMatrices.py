# you should fill in the functions in this file,
# do NOT change the name, input and output of these functions

import numpy as np
import time
import matplotlib.pyplot as plt

# first function to fill, compute distance matrix using loops
def compute_distance_naive(X):
    N = X.shape[0]      # num of rows
    D = X[0].shape[0]   # num of cols
    
    M = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            xi = X[i,:]    # take ith row
            xj = X[j,:]    # take jth row

            dist = np.sqrt(np.sum((xi-xj)*(xi-xj)))  # a placetaker line

            M[i,j] = dist

    return M

# second function to fill, compute distance matrix without loops
def compute_distance_smart(X):
    N = X.shape[0]  # num of rows
    D = X[0].shape[0]  # num of cols

    # use X to create M
    M = np.zeros([N, N])
    A = np.zeros((N, N))
    B = np.zeros((N, N))
    C = np.zeros((N, N))

    xiSum = np.sum(np.multiply(X,X), axis=1)   # row vector of length N. Sum over rows

    A = np.transpose((xiSum, ) * N)   # clone vector xisum to a matrix of size NxN
    #A = (xiSum * np.ones([N,1]))

    #Xtr = np.transpose(X)

    #B = np.dot(X, Xtr)
    B = np.inner(X,X)
    C = np.transpose(A)

    Sum = np.add(np.add(A, -2 * B), C)

    M = np.sqrt(abs(Sum))   # M = sqrt(A - 2*B + C)

    return M

# third function to fill, compute correlation matrix using loops
def compute_correlation_naive(X):
    N = X.shape[0]  # num of rows
    D = X[0].shape[0]  # num of cols

    # use X to create M
    M = np.zeros([D, D])

    for i in range(D):
        for j in range(D):
            xi = X[:, i]   # take ith column
            xj = X[:, j]   # take jth column
            xih = xi - np.sum(xi).astype(float) / N   # xi - mui
            xjh = xj - np.sum(xj).astype(float) / N   # xj - muj
            sij = (np.dot(xih, xjh)).astype(float) / (N-1)    # covariance at i row j column

            stdi = np.sqrt(np.dot(xih, xih).astype(float) / (N - 1))  # sigmai
            stdj = np.sqrt(np.dot(xjh, xjh).astype(float) / (N - 1))  # sigmaj
            corr = sij.astype(float) / stdi / stdj # a placetaker line,
                    #  you have to change it to correlation between xi and xj
            M[i, j] = corr

    return M

# fourth function to fill, compute correlation matrix without loops
def compute_correlation_smart(X):
    N = X.shape[0]  # num of rows
    D = X[0].shape[0]  # num of cols

    # use X to create M
    M = np.zeros([D, D])

    mu = (np.sum(X, axis=0)).astype(float) / N    #mean vector. column sum of column vectors

    MatrixMu = mu * np.ones([N,1])    #matrix with  mu

    Xmu = X - MatrixMu

    S = (np.dot(np.transpose(Xmu),Xmu)).astype(float) / (N-1)     #covariance matrix

    varvector = (np.sum(np.multiply(Xmu, Xmu),axis=0)).astype(float) / (N-1)     #variance vector

    std = np.sqrt(varvector)    #std vector

    Denom = np.outer(std, std)     #denominator matrix

    M = np.multiply(S, np.power(Denom, -1))

    return M

def main():
    print 'starting comparing distance computation .....'
    np.random.seed(100)
    params = range(10,141,10)   # different param setting
    nparams = len(params)       # number of different parameters

    perf_dist_loop = np.zeros([10,nparams])  # 10 trials = 10 rows, each parameter is a column
    perf_dist_cool = np.zeros([10,nparams])
    perf_corr_loop = np.zeros([10,nparams])  # 10 trials = 10 rows, each parameter is a column
    perf_corr_cool = np.zeros([10,nparams])


    counter = 0

    for ncols in params:
        nrows = ncols * 10

        print "matrix dimensions: ", nrows, ncols

        for i in range(10):
            X = np.random.rand(nrows, ncols)   # random matrix

            # compute distance matrices
            st = time.time()
            dist_loop = compute_distance_naive(X)
            et = time.time()
            perf_dist_loop[i,counter] = et - st              # time difference

            st = time.time()
            dist_cool = compute_distance_smart(X)
            et = time.time()
            perf_dist_cool[i,counter] = et - st

            assert np.allclose(dist_loop, dist_cool, atol=1e-06) # check if the two computed matrices are identical all the time

            # compute correlation matrices
            st = time.time()
            corr_loop = compute_correlation_naive(X)
            et = time.time()
            perf_corr_loop[i,counter] = et - st              # time difference

            st = time.time()
            corr_cool = compute_correlation_smart(X)
            et = time.time()
            perf_corr_cool[i,counter] = et - st

            assert np.allclose(corr_loop, corr_cool, atol=1e-06) # check if the two computed matrices are identical all the time

        counter = counter + 1

    mean_dist_loop = np.mean(perf_dist_loop, axis = 0)    # mean time for each parameter setting (over 10 trials)
    mean_dist_cool = np.mean(perf_dist_cool, axis = 0)
    std_dist_loop = np.std(perf_dist_loop, axis = 0)      # standard deviation
    std_dist_cool = np.std(perf_dist_cool, axis = 0)

    plt.figure(1)
    plt.errorbar(params, mean_dist_loop[0:nparams], yerr=std_dist_loop[0:nparams], color='red',label = 'Loop Solution for Distance Comp')
    plt.errorbar(params, mean_dist_cool[0:nparams], yerr=std_dist_cool[0:nparams], color='blue', label = 'Matrix Solution for Distance Comp')
    plt.xlabel('Number of Cols of the Matrix')
    plt.ylabel('Running Time (Seconds)')
    plt.title('Comparing Distance Computation Methods')
    plt.legend()
    plt.savefig('CompareDistanceCompFig.pdf')
    # plt.show()    # uncomment this if you want to see it right way
    print "result is written to CompareDistanceCompFig.pdf"

    mean_corr_loop = np.mean(perf_corr_loop, axis = 0)    # mean time for each parameter setting (over 10 trials)
    mean_corr_cool = np.mean(perf_corr_cool, axis = 0)
    std_corr_loop = np.std(perf_corr_loop, axis = 0)      # standard deviation
    std_corr_cool = np.std(perf_corr_cool, axis = 0)

    plt.figure(2)
    plt.errorbar(params, mean_corr_loop[0:nparams], yerr=std_corr_loop[0:nparams], color='red',label = 'Loop Solution for Correlation Comp')
    plt.errorbar(params, mean_corr_cool[0:nparams], yerr=std_corr_cool[0:nparams], color='blue', label = 'Matrix Solution for Correlation Comp')
    plt.xlabel('Number of Cols of the Matrix')
    plt.ylabel('Running Time (Seconds)')
    plt.title('Comparing Correlation Computation Methods')
    plt.legend()
    plt.savefig('CompareCorrelationCompFig.pdf')
    # plt.show()    # uncomment this if you want to see it right way
    print "result is written to CompareCorrelationCompFig.pdf"

if __name__ == "__main__": main()
