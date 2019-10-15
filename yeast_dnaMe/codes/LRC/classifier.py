#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 19:50:42 2019

@author: afinneg2
"""
from __future__ import  division
import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.optimize import root as find_root 
from numpy.linalg import multi_dot
import numpy.linalg as LA
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import matplotlib.pyplot as plt
import seaborn as sns


class LogReg_Continuous(object):
    def __init__(self):
        """
        """
        self.beta = None
        return None
     
    def fit(self, X,  y, lmbda, method = "scipy", n_categ_levels = 0 , penalty_type = "D2_sq"):
        """
        Inputs:
        -------
            X - ndarray rows are observations columns are features
            y - 1d array  (XX 1d array of class lambes or 2d array of one-hot class encoding)
        """
        self.X = X
        self.y = y
        self.lmbda = lmbda
#         if len(y.shape) == 1:
#             self.y = self.to_onehot(y)
#         else:
#             self.y = y
        if n_categ_levels == 0:
            self._X = np.hstack( [np.ones(X.shape[0], dtype = float)[:,None], self.X])
            self.n_categ_levels = 1
            self.add_intercept = True
        else:
            self._X  = self.X
            self.n_categ_levels = n_categ_levels
            self.add_intercept = False
        self.n_bins =  self._X.shape[1] - n_categ_levels
        #self.nObs_train = self._X.shape[0]

        self._DtD = self._make_DtD(self.n_bins, n_categ_levels, penalty_type =  penalty_type ) 
        f_newton = self.get_f_newton(X = self._X, y=self.y, lmbda = self.lmbda, DtD = self._DtD)
        fprime_newton = self.get_fprime_newton(X = self._X, lmbda  = self.lmbda, DtD = self._DtD )
        beta0 = np.zeros(shape = self._X.shape[1], dtype = float)
        if method.lower()  == "scipy":
            soln = find_root(fun = f_newton, jac = fprime_newton, x0 = beta0, method = "hybr", 
                    options = {"maxfev": 1000*len(beta0), 'factor': 10. })  
            if not soln["success"]:
                raise Exception("find_root failed with message\n {}\n fun is\n {}".format(soln["message"] , soln["fun"]))
            self.beta = soln["x"]
        elif method.lower() == "custom":
            soln = self.findRoot_newton(f= f_newton, fprime = fprime_newton, x0= beta0,  max_iter= 10000, tol = 0.0000001 )
            if not soln[3]:
                raise Exception("my_newton failed, fun is\n {},\n{}".format(soln["message"] , soln[1]))
            self.beta = soln[0]
        else:
            raise NotImplementedError()
        return soln


    def predict(self, X, y = None ):
        """
        Inputs
            X - ndarray rows are observations columns are features
            y - 1d array of 0, 1 class labels
        """
        if self.beta is None:
            raise Exception("run .fit method first")
        if self.add_intercept:
            X_pred =  np.hstack( [np.ones(X.shape[0], dtype = float)[:,None], X] )
        else:
            X_pred  = X
        preds = self.p_pos(X_pred, self.beta )
        if y is not None:
            y_pred = (preds > 0.5).astype(float)
            acc = np.isclose(y_pred, y).sum() / len(y)
            return preds, acc
        else:
            return preds
        
    def ROC(self , X , y, ax = None):
        preds = self.predict(X)
        print(preds)
        fpr, tpr,  thresholds = roc_curve(y,  preds ) 
        auc_val = auc(fpr, tpr)
        if ax is None:
            return auc_val, [fpr, tpr,  thresholds]
        else:
            ax.plot(fpr, tpr, color='darkorange',
                         lw=2, label='ROC curve (area = %0.2f)' % auc_val)
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver operating characteristic example')
            ax.legend(loc="lower right")
            return auc_val, [fpr, tpr,  thresholds], ax
            
    @staticmethod
    def _make_DtD(n_bins, n_categ_levels, penalty_type = "D2_sq"):
        """
        DtD is matrix that gives discrete approximation of integral( (beta'')**2 )
        """
        if penalty_type  == "D2_sq":
            D = np.zeros( shape= (n_categ_levels + n_bins -2, n_categ_levels + n_bins) ,dtype = float )
            for i in range(n_categ_levels, D.shape[0]):
                D[i, [i, i+1 , i+2] ] = 1,-2, 1
            D  = D * (n_bins**2) ## D / (\deta x)**2 = D / (1/n_bins)**2 so that derivative is in units of standardized gene length 
        elif penalty_type == "D1_sq":
            D = np.zeros( shape= (n_categ_levels + n_bins -1, n_categ_levels + n_bins) ,dtype = float )
            for i in range(n_categ_levels, D.shape[0]):
                D[i, [i, i+1 ] ] = -1, 1
            D = D * n_bins ## D / (\deta x) = D/ (1/n_bins)
        DtD = np.dot( np.transpose(D), D)
        return DtD
    
    @staticmethod
    def get_f_newton(X, y, lmbda, DtD ):
        X_t = np.transpose(X)
        def f_newton(beta):
            """
            beta - ndarray with shape (1 + n_features, )
            """
            r_val = np.dot(X_t,
                           y -  LogReg_Continuous.p_pos(X, beta) 
                          ) - lmbda*np.dot(DtD, beta)
            return r_val
        return f_newton
    
    @staticmethod
    def get_fprime_newton(X,  lmbda, DtD ):
        X_t = np.transpose(X)
        def fprime_newton(beta):
            p = LogReg_Continuous.p_pos(X, beta) 
            W = np.diag(p*(1.0 - p ))
            r_val = -1.0*multi_dot([X_t , W , X] ) - lmbda*DtD
            return r_val
        return fprime_newton
    
    @staticmethod
    def p_pos(X, beta ):
        r_val = np.divide(1.0, 1 + np.exp(-1*np.dot(X, beta)) )
        return r_val
             
    @staticmethod    
    def to_onehot(y):
        nClasses = np.max(y)
        onehot = np.stack( [np.eye(nClasses)[i, :] for i in y ] )
        return onehot
    
    @staticmethod
    def findRoot_newton( f, fprime, x0, max_iter= 10000, tol = 0.0000001  ):
        """
        simple root finder for testing purpose. Sometimes throws non-invertable
        error when DtD is much larger that -1.0*multi_dot([X_t , W , X] ).
        """
        x= x0
        for i in range(max_iter):
            f_val = f(x)
            if np.all(np.abs(f_val) < tol):
                return x , f_val, i, True, jac
            else:
                jac = fprime( x )
                x = x - np.dot(LA.inv(jac), f_val) 
        print("Failed to converge after {} iteratoins".format( max_iter))
        return x, f_val, i, False, jac
    




