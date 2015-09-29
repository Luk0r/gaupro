# -*- coding: utf-8 -*-"""
"""
Created on Wed Aug  6 11:06:31 2014

@author: schmidan
"""

import ctypes
import numpy as np


class GP(object):
    '''Gaussian Process class (C interface)
    
    available covariance functions:
    cov functions:
    CovLinearard
    CovLinearone
    CovMatern3iso
    CovMatern5iso
    CovNoise
    CovRQiso
    CovSEard
    CovSEiso
    CovPeriodicMatern3iso
    CovPeriodic
    
    CovSum
    CovProd
    InputDimFilter
    
    use like:
    "CovSum ( CovSEiso, CovNoise)"
    
    '''

    def __init__(self,ndim,covf):
        #print("__init__(self,ndim,covf):")
        print('gaupro v0.1')
        self.libgp = ctypes.cdll.LoadLibrary('libgaupro.so')

        self.new = self.libgp.gp_new
        self.new.restype = ctypes.c_void_p
        self.new.argtypes = [ctypes.c_size_t, ctypes.c_char_p]
        self.libgp_ptr = self.new(ndim, covf.encode("ascii"))

        #self.libgp_ptr = self.libgp.gp_new(ctypes.c_size_t(ndim), ctypes.c_char_p(covf.encode("ascii")))
        print("libgp_ptr:\n"+str(self.libgp_ptr))
        
        self.libgp.gp_get_loghyperparam_dim.argtypes = [ctypes.c_void_p]
        dim = self.libgp.gp_get_loghyperparam_dim(self.libgp_ptr)
        
        #print("dim = "+str(dim))
        self.ndim = ndim
        self.covf = covf
       # print("__init__(self,ndim,covf):")

    def train(self, x, y, optimizer="rprop", opti_iters=100, eps_stop=0.0):#optimizer="cg" or "rprop"
        if x.shape[0] == y.shape[0]:
            self.libgp.gp_add_train(ctypes.c_void_p(self.libgp_ptr), x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                    ctypes.c_int(len(x.shape)), x.ctypes.shape_as(ctypes.c_int),
                                    y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                                    )

            # set hyperparam starting point before optimizing


            self.optimize = self.libgp.gp_optimize
            self.optimize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_double]
            self.optimize.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape=(opti_iters,))
            self.likelihood_curve = self.optimize(self.libgp_ptr, optimizer.encode("ascii"), opti_iters, eps_stop)
            print("finished GP training")
            print("log_likelihood = "+str(self.likelihood_curve[-1]))
            self.trained = True
            
        else:
            print("ERROR: x.shape[1] == y.shape[0]")        
            
    def predict(self, x):
        #print("dffsfs")
        if self.trained:
            self.pred = self.libgp.gp_predict_value
            #print(x.shape[0]*2)
            
            if len(x.shape) == 1:
                #print("if len(x.shape) == 1:")
                self.pred.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape=(1+1,))
                self.pred.argtypes = [ctypes.c_void_p, np.ctypeslib.ndpointer(ctypes.c_double),
                                     ctypes.c_int, ctypes.POINTER(ctypes.c_int)] 
                values_variance = self.pred(self.libgp_ptr, x, len(x.shape), x.ctypes.shape_as(ctypes.c_int))
                                     
                self.predicted_values = np.copy(values_variance[0])
                self.predicted_variances = np.copy(values_variance[1])
                
                #return np.array([self.predicted_values]), np.array([self.predicted_variances])
                return self.predicted_values, self.predicted_variances
                
            else:
                #print("if len(x.shape) == 1: ELSE")
                self.pred.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape=(x.shape[0]+x.shape[0],))
                
                self.pred.argtypes = [ctypes.c_void_p, np.ctypeslib.ndpointer(ctypes.c_double),
                                         ctypes.c_int, ctypes.POINTER(ctypes.c_int)]            
                #values_variance = self.predict(self.libgp_ptr, x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                #                        ctypes.c_int(len(x.shape)), x.ctypes.shape_as(ctypes.c_int)
                #                        )
                #values_variance = self.predict(self.libgp_ptr, x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                #                        ctypes.c_int(len(x.shape)), x.ctypes.shape_as(ctypes.c_int)
                #                        )
                values_variance = self.pred(self.libgp_ptr, x, len(x.shape), x.ctypes.shape_as(ctypes.c_int))
                
                self.predicted_values = np.copy(values_variance[0:x.shape[0]])
                self.predicted_variances = np.copy(values_variance[x.shape[0]:])
                return self.predicted_values, self.predicted_variances
                #return np.array([self.predicted_values]), np.array([self.predicted_variances])
                
                
                
        else:
            print("need trainign before")

    def get_loghyper_dims(self):
        self.libgp.gp_get_loghyperparam_dim.restype = ctypes.c_int
        self.libgp.gp_get_loghyperparam_dim.argtypes = [ctypes.c_void_p]
        return self.libgp.gp_get_loghyperparam_dim(self.libgp_ptr)
         
    def set_loghyper(self, loghyperparam):
        self.libgp.gp_get_loghyperparam_dim.restype = ctypes.c_int
        self.libgp.gp_get_loghyperparam_dim.argtypes = [ctypes.c_void_p]
        dim = self.libgp.gp_get_loghyperparam_dim(self.libgp_ptr)
        if (len(loghyperparam.shape) == 1 and loghyperparam.shape[0] == dim):
            self.libgp.gp_set_loghyper(ctypes.c_void_p(self.libgp_ptr), loghyperparam.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                       ctypes.c_int(loghyperparam.shape[0]))
        else:
            print("loghyperparam dims do not match (python)")
            
    def set_constraints(self, lowerConstraints, upperConstraints):
        """setting the constraints for the hyperparameters.
        dimensions need to match with the dims of the hyperparameters
        
        :param lowerConstraints: The lower constraint.
        :param upperConstraints: The upper constraint.
        
        """
        self.libgp.gp_get_loghyperparam_dim.restype = ctypes.c_int
        self.libgp.gp_get_loghyperparam_dim.argtypes = [ctypes.c_void_p]
        dim = self.libgp.gp_get_loghyperparam_dim(self.libgp_ptr)
        if (len(lowerConstraints.shape) == 1 and lowerConstraints.shape[0] == dim):
            if (len(upperConstraints.shape) == 1 and upperConstraints.shape[0] == dim):
            #virtual void set_constraints(const double lower[], const double upper[]);
                self.libgp.gp_set_loghyper_constraints( ctypes.c_void_p(self.libgp_ptr),
                    lowerConstraints.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    upperConstraints.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    ctypes.c_int(lowerConstraints.shape[0]) )
            else:
                print('Error (python): upperConstraints.shape not matching hyperparams shape')
        else:
            print("Error (python): lowerConstraints.shape not matching hyperparams shape")

        
    def get_loghyper(self):
        self.libgp.gp_get_loghyper_len.restype = ctypes.c_int
        self.libgp.gp_get_loghyper_len.argtypes = [ctypes.c_void_p]
        lengths = self.libgp.gp_get_loghyper_len(self.libgp_ptr)
        #print(lengths)
        #gp_get_loghyper = self.libgp.gp_get_loghyper
        self.libgp.gp_get_loghyper.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape=(lengths,))
        self.libgp.gp_get_loghyper.argtypes = [ctypes.c_void_p, ctypes.c_int]
        #gp_get_loghyper.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape=(lengths,))
        #gp_get_loghyper.argtypes = [ctypes.c_int, ctypes.c_int]
        self.loghyperparam = self.libgp.gp_get_loghyper(self.libgp_ptr, lengths)
        return self.loghyperparam
        
    def get_log_likelihood(self):
        self.libgp.gp_get_loglikelihood.restype = ctypes.c_double
        self.libgp.gp_get_loglikelihood.argtypes = [ctypes.c_void_p]
        return self.libgp.gp_get_loglikelihood(self.libgp_ptr)
        
        
        
        
        
        
        
        
