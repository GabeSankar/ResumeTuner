import numpy as np
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
import re
class VectorSpaceMath():

    def __init__(self):
        #initialize sentance transformer
        self.sentance_transformer = SentenceTransformer('all-MiniLM-L6-v2')

    def gaussian_kernel(self, x, y, sigma=1.0):
        """Compute Gaussian (RBF) kernel between x and y."""
        pairwise_sq_dists = cdist(x, y, 'sqeuclidean')
        return np.exp(-pairwise_sq_dists / (2 * sigma ** 2))

    def compute_weighted_maxmimum_mean_discrepancy(self, X, Y, w_X, w_Y, sigma):
        """Running Maximum Mean Discrepancy between embeddings with weights to emphasize job title
        over tasks within job to analyze most important similarity
        
        inputs are:
        X, which is the array of n samples by the feature vectors
        Y, which is the array of n samples by the feature vectors of the other distribution
        w_X, which is the optional weights for X
        w_Y, which is the optional weights for Y
        weights is 1d numpy tensor
        sigma, which is the bandwith of the gaussian kernel, controls standard deviation"""
        K_XX = self.gaussian_kernel(X, X, sigma)
        K_YY = self.gaussian_kernel(Y, Y, sigma)
        K_XY = self.gaussian_kernel(X, Y, sigma)

        #Normalize the weights
        n_x, n_y = X.shape[0], Y.shape[0]
        #initialize equal weights if no weights
        if w_X is None:
            w_X = np.ones(n_x) / n_x
        else:
            w_X = w_X / np.sum(w_X)
        
        if w_Y is None:
            w_Y = np.ones(n_y) / n_y
        else:
            w_Y = w_Y / np.sum(w_Y)

        #Compute squared maximum mean discrepancy with weighted sums
        mmd = np.sum(K_XX * np.outer(w_X, w_X)) + np.sum(K_YY * np.outer(w_Y, w_Y)) - 2 * np.sum(K_XY * np.outer(w_X, w_Y))
        #Compute maximum mean discrepancy
        return np.sqrt(mmd)
    
    def cosine_similarity(self, a, b):
        """Compute the cosine similarity between two embedding"""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0  #No division by zero
        return np.dot(a, b) / (a_norm * b_norm)
    
    def average_embedding(self, embeddings):
        """Averages embeddings along first dimension with input of shape (n_samples, embedding dim)"""
        return np.mean(embeddings, axis=0)
    
    def create_embeding_vectors(self, texts):
        """Creates embeddings with sentance transformer for list of texts"""
        return self.sentance_transformer.encode(texts)
    