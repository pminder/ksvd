#coding:utf8

import warnings
warnings.filterwarnings("ignore")

"""Implements K-SVD algorithm through a KSVD class.
For details about the ideas behind the implementation,
see:
    [1] Efficient implementation of the K-SVD algorithm
      and Batch-OMP Method by Rubinstein, 
      Zibulevsky and Elad
    [2] K-SVD: an algorithm for designing of overcomplete
      dictionaries for sparse representation by Aharon,
      Elad, Bruckstein and Katz

We mainly used scikit-learn conventions to define the 
class's methods.
"""

import numpy as np
from sklearn.linear_model import orthogonal_mp
from ipy_progressbar import ProgressBar

class KSVD:
    """Implement K-SVD algorithm, more or less using
    scikit-learn method conventions"""

    def __init__(self, D, K = None, n_iter = 200, precompute = True):
        """Initialization method. Args:
        - D: tuple of ints -> dimension of intial 
             dictionary
             or numpy 2d-array -> actual initial
             dictionary
        - K: (int) target sparsity. If None is passed
          K will be fixed to 10% of signals present in 
          intial dictionary D
        - n_iter: (int) number of iterations for
          OMP algorithm
        - precompute: if True use Batch OMP algorithm.
          Else, use standard OMP algorithm
        """

        #Set initial dictionary
        if type(D) == tuple:
            #TODO: what is best initial dictionary?
            #Maybe we should rather take the first
            #self.D.shape[1] columns from X matrix
            #just before starting to fit the model...
            D = self.init_dic(D[0], D[1])
        self.D = D

        #Set target sparsity
        if not K:
            K = D.shape[1]/10
        self.K = K

        #Set number of iterations
        self.n_iter = n_iter
        #Decide wether to use or not Batch OMP
        self.precompute = precompute

    def init_dic(self, n_features, n_samples):
        """Return a not so bad dictionary of dimension:
        (n_features, n_samples).
        It is an entirely random dictionary.
        """
        D = np.random.randint(low = 0, high = 2,
                size = (n_features, n_samples))

        #Normalize dict columns
        D = D.astype('float')/D.sum(axis = 0)

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

    def worst_represented(self, gamma, X):
        """Returns X columns with the worst representation
        in gamma for self.D dictionary"""
        X_ = self.D.dot(gamma)
        errors = np.abs(X - X_).sum(axis = 0)
        return X[:,errors.argmax()]

    def fit(self, X):
        """Actual implementation of K-SVD algorithm.
        Args:
        - X: numpy 2d-array of dimensions :
          (len(signal) = D.shape[0], n_samples)
        TODO: add a stopping condition like an epsilon
        (and return corresponding number of iterations
        """

        #Check wether data is coherent
        if self.D.shape[0] != X.shape[0]:
            raise TypeError("Supplied X matrix is not "
            "coherent with dictionary dimensions: you "
            "should have same number of lines for "
            "both the dictionary and the input data ")

        #ProgressBar setup
        print "Training dictionary over {} iterations".format(self.n_iter)
        progress = ProgressBar(self.n_iter)

        #self.n_iter iterations
        for it in progress:

            #Step 1: Compute sparse representation of X
            #given current dictionary D
            gamma = orthogonal_mp(self.D, X, n_nonzero_coefs = self.K,
                            precompute = self.precompute)
            #Step 2: Adjust dictionary D and sparse
            #representation gamma at the same time
            #column by column
            for j in range(self.D.shape[1]):
                
                #Compute I = {indices of the signals in X
                #whose representations use jth column of D
                I = self.find_indices(gamma, j)
                
                #If one column is not used, it won't be until 
                #the algorithm actually stops, which is a shame
                #So, we use heuristics: we set teh values of the
                #column to the worst represented columns of
                #X matrix
                if I == []:
                    #find worst represented column in X
                    d = self.worst_represented(gamma, X)
                    #normalize
                    d = d/np.linalg.norm(d)
                    #set D column to d
                    self.D[:,j] = d
                    #jump to the next column optimization
                    continue

                #Set D_j to zero
                self.D[:,j] = np.zeros_like(self.D[:,j])

                #From now, we use a certain number of tricks
                #explained in [1] to accelerate the (therefore
                #approximate) K-SVD algorithm
                #TODO: try to understand better... -> maybe we could
                #solve the equations in the report ;)
                g = gamma[j,:][I].T
                d = X[:,I].dot(g) - self.D.dot(gamma[:,I].dot(g))
                if d.sum() != 0:
                    d = d/np.linalg.norm(d)
                g = (X[:,I].T).dot(d) - ((self.D.dot(gamma[:,I])).T).dot(d)

                #Store new values
                self.D[:,j] = d
                gamma[j,:][I] = g.T

    def sparse_rep(self, X):
        """Return sparse representation of column vectors
        present in X, according to current self.D dictionary.
        We use batch OMP algorithm to find the representation"""
        return orthogonal_mp(self.D, X, n_nonzero_coefs = self.K,
                precompute = self.precompute)

