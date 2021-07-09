#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:36:13 2021

@author: danbiderman
TODO: support batched matrix multiplication. see the following example of using @ matmul to support e.g., 100-d observations
a = np.random.normal(size=
                (6,3))
b = np.random.normal(size=
                (3,3))
c = np.random.normal(size=
                (100, 3,1))
res = a@b@c
res.shape
"""

import scipy
import numpy as np

class LinearGaussianModel:
    """p(z) = N(z; \mu, Lambda^{-1})
       p(x|z) = N(x; Az + b, L^}-1{})"""
    def __init__(self, mu, Lambda, A, b, L, simplify_posterior=False):
        """Parametrized as in Bishop PRML Appendix B.32-B.51"""
        # prior quantities
        self.prior_mean = mu  # maybe redundant? can use just the scipy obj? or consider classes for prior and like?
        self.prior_precision = Lambda # M by M
        self.prior_cov = np.linalg.inv(Lambda) # M by M 
        self.prior_obj = scipy.stats.multivariate_normal(
            self.prior_mean.squeeze(), self.prior_cov)
        # likelihood quantities
        self.likelihood_A = A # D by M
        self.likelihood_b = b # D by 1
        self.likelihood_precision = L # D by D
        self.likelihood_cov = np.linalg.inv(L) # D by D
        self.marginal_obj = self.compute_marginal_using_prior()
        self.simplify_posterior= simplify_posterior # this is for the PPCA posterior.
        # general posterior quantities
#         self.Sigma = np.linalg.inv(self.prior_precision + np.linalg.multi_dot(
#             self.likelihood_A.T, self.likelihood_precision, self.likelihood_A)) # posterior cov; not dependant on obss

    def compute_like_mean(self, z):
        '''compute Az+b.
        z can be batched in last dimension'''
        assert (len(z.shape) == len(self.likelihood_b.shape)
                )  # even if last dim isn't equal, broadcasting is fine
        return self.likelihood_A.dot(z) + self.likelihood_b
    
    @staticmethod
    def extract_blocks_from_inds(valid_inds, cov_mat):
        '''recieves a covariance matrix and extracts the relevant blocks'''
        assert (np.diff(valid_inds)>=0).all() # ensure valid_inds are sorted
        row_inds = np.tile(valid_inds, (len(valid_inds)))
        col_inds = np.repeat(valid_inds, len(valid_inds), axis=0)
        return cov_mat[row_inds, col_inds].reshape(len(valid_inds), len(valid_inds))
        
    def compute_posterior(self, obs):
        """
        Posterior mean and variance are:
        $\mu = \Sigma{A^{\top}L(y-b) + \Lambda \mu}$
        $\Sigma = (\Lambda + A^{\top} L A)^{-1}$
        obs could have missing values, 
        and therefore we'll take the corresponding blocks using the marginalization property of Gaussians
        """
        # better to return scipy obj?
        # ToDo: currently assuming obs is a vector, not a matrix. extend
        valid_inds = np.where(~np.isnan(obs))[0] # assuming obs is a vector
        L_valid = self.extract_blocks_from_inds(valid_inds, self.likelihood_precision)
        A_valid = self.likelihood_A[valid_inds,:]
        Sigma_valid = np.linalg.inv(self.prior_precision + np.linalg.multi_dot(
           [A_valid.T, L_valid, A_valid])) # posterior cov; if no nans, could be cached beforehand
        y_minus_b = (obs[valid_inds]-self.likelihood_b[valid_inds]).reshape(-1,1) # again, vector not matrix
        mean_valid = Sigma_valid.dot(
         np.linalg.multi_dot([A_valid.T, L_valid, y_minus_b]) + \
            self.prior_precision.dot(self.prior_mean))
        if self.simplify_posterior:
            M = np.unique(np.diag(self.likelihood_cov))*np.eye(np.shape(self.prior_precision)[0]) + \
                np.dot(A_valid.T, A_valid)
            M_inv = np.linalg.inv(M)
            Sigma_valid = M_inv*np.unique(np.diag(self.likelihood_cov))
            mean_valid = np.linalg.multi_dot([M_inv, A_valid.T, y_minus_b])
        assert(all((Sigma_valid-Sigma_valid.T).flatten() < 0.00001))
        return mean_valid, Sigma_valid
    
    def predict(self, posterior_mean, posterior_cov, use_epsilon_var=False):
        """compute the predictive distribution:
        p(x) = \mathcal{N}(mu, Sigma)
        where mu = A*posterior_mean + b
        ans Sigma = A*posterior_cov*A^{\top} + R
        compute preds for all data points, no missing vals.
        you can later pick those dims of interest"""
        mu_x = self.likelihood_A.dot(posterior_mean) + self.likelihood_b.reshape(-1,1)
        if not use_epsilon_var: # the exact linear gaussian predictive
            Sigma_x =  np.linalg.multi_dot([self.likelihood_A, posterior_cov, self.likelihood_A.T]) + \
                    self.likelihood_cov
        else:
            Sigma_x =  np.linalg.multi_dot([self.likelihood_A, posterior_cov, self.likelihood_A.T]) + \
                    np.eye(self.likelihood_cov.shape[0])* 0.001
        
        return mu_x, Sigma_x
    
    def compute_marginal_using_prior(self):
        # ToDo: consider unifying these ops with predict's ops. to just input mean and cov of z.
        mu_x = self.likelihood_A.dot(self.prior_mean) + self.likelihood_b.reshape(-1,1)
        Sigma_x =  np.linalg.multi_dot([self.likelihood_A, self.prior_cov, self.likelihood_A.T]) + \
                self.likelihood_cov
        return scipy.stats.multivariate_normal(
           mu_x.squeeze(), Sigma_x)

    def calc_zhat_w_pinv(self, obs):
        """
        obs: observation (2K-dim vec), K x's followed by K y's. may contain nans
        compute latent z using obs and pseuduinv(A). pseudoinverse of an e.g., (N x M) matrix is (M x N).
        """
        valid_inds = np.where(~np.isnan(obs))[0]
        diff = (obs[valid_inds] - self.likelihood_b[valid_inds]).reshape(-1, 1)
        return np.linalg.pinv(self.likelihood_A[valid_inds, :]).dot(diff)

    def predict_w_zhat(self, z_hat):
        # same as mean prediction in self.predict
        return self.likelihood_A.dot(z_hat) + self.likelihood_b.reshape(-1, 1)

    def sample_from_prior(self):
        return 0
        # in fact one can just sample from p(x) = \int_z p(z)p(x|z) dz; z \sim prior()