#coding:utf8

"""Implements K-SVD algorithm through a KSVD class.
For details about teh ideas behind the implementation,
see:
    - Efficient implementation of the K-SVD algorithm
      and Batch-OMP Method by Rubinstein, 
      Zibulevsky and Elad
    - K-SVD: an algorithm for designing of overcomplete
      dictionaries for sparse representation by Aharon,
      Elad, Bruckstein and Katz

We mainly used scikit-learn conventions to define the 
class's methods.
"""

import numpy as np
from sklearn.linear_model import orthogonal_mp

class KSVD:

    def __init__(self, D, K = None, n_iter = 200):
        """Initialization method. Args:
        - D: tuple of ints -> dimension of intial 
             dictionary
             or numpy 2d-array -> actual initial
             dictionary
        - K: (int) target sparsity. If None is passed
          K will be fixed to 10% of signals present in 
          intial dictionary D
        - n_iter: (int) number of iterations for
          OMP algorithm"""

        #Set initial dictionary
        if type(D) == tuple:
            #TODO: what is best initial dictionary?
            #Columns have to be normalized...
            D = self.init_dic(D[0], D[1])
        self.D = D

        #Set target sparsity
        if not K:
            K = D.shape[1]/10
        self.K = K

        #Set number of iterations
        self.n_iter = n_iter

    def init_dic(self, n_features, n_samples):
        """Return a not so bad dictionary of dimension:
        (n_features, n_samples).
        """
        D = np.zeros((n_features, n_samples))
        mask = np.random.randint(low = 0, high = n_samples,
                size = n_features)

        #TODO: make this faster
        for i in range(len(mask)):
            D[i, mask[i]] = 1

        #Normalize dict columns
        D = D/D.sum(axis = 0)

        return D

    def find_indices(self, gamma, j):
        """Return array of indices of the signals
        whose representation use the jth column of 
        self.D, according to passed gamma matrix"""

        #Initialize indices' set
        I = []

        #For each column of gamma, check wether jth
        #value is null or not
        for i in range(gamma.shape[1]):
            if gamma[j, i] != 0:
                I.append(i)

        return I

    def fit(self, X):
        """Actual implementation of K-SVD algorithm.
        Args:
        - X: numpy 2d-array of dimensions :
          (len(signal) = D.shape[0], n_samples)
        """

        #Check wether data is coherent
        if self.D.shape[0] != X.shape[0]:
            raise TypeError("Supplied X matrix is not "
            "coherent with dictionary dimensions: you "
            "should have same number of lines for "
            "both the dictionary and the input data ")


        #self.n_iter iterations
        for _ in range(self.n_iter):

            #Step 1: Compute sparse representation of X
            #given current dictionary D
            gamma = orthogonal_mp(self.D, X, n_nonzero_coefs = self.K)

            #Step 2: Adjust dictionary D and sparse
            #representation gamma at the same time
            #column by column
            for j in range(self.D.shape[1]):
                
                #Set D_j to zero
                self.D[:,j] = np.zeros_like(self.D[:,j])

                #Compute I = {indices of the signals in X
                #whose representations use jth column of D
                I = self.find_indices(gamma, j)

                #From now, we use a certain number of tricks
                #explained in [1] to accelerate the (therefore
                #approximate) K-SVD algorithm
                #TODO: try to understand better...
                g = gamma[j,:][I].T
                d = X[:,I].dot(g) - self.D.dot(gamma[:,I].dot(g))
                if d.sum() != 0:
                    d = d/np.linalg.norm(d)
                else:
                    print "sum = 0 for col n°{}".format(j)
                g = (X[:,I].T).dot(d) - ((self.D.dot(gamma[:,I])).T).dot(d)

                #Store new values
                self.D[:,j] = d
                gamma[j,:][I] = g.T

        return gamma

