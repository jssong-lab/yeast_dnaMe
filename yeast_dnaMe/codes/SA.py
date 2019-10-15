#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:51:15 2019

@author: afinneg2
"""
from __future__ import  absolute_import, division, print_function
import numpy as np
import sys
import os
import pickle
import gzip
from collections import OrderedDict
from scipy.stats import  entropy
import argparse
import time

from tensorflow.python.client import device_lib
import keras
from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Concatenate,Input, concatenate, Dropout
from keras.optimizers import SGD
import keras.backend as K

import matplotlib.pyplot as plt
from AFutils.parsers.misc import  readListFromFile


gpu_lst = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
n_gpu = len(gpu_lst)
print(gpu_lst )


##################################################
## GLOBALS
indices_allowed = np.concatenate([np.arange(0,100,1, dtype = np.int64), np.arange(102, 201, dtype = np.int64)])

##################################################3
## FUNCTION DEFINITIONS
#############################################
### Functions 
## Somang
def log_lkh_loss(y_true, y_pred):
    m = y_true[:,0]
    m = K.reshape(m,(-1,1))
    n = y_true[:,1]
    n = K.reshape(n,(-1,1))
    return -K.mean(m*K.log(y_pred)+n*K.log(1-y_pred),axis=-1)

def loadModel(hdf5path):
    maxlen=201
    filter_N = 80
    hd_layer_N = 40
    droprate2 = 0.2
    droprate = 0.2
    input_shape2 = (5,maxlen,1)

    inp2 = Input(shape=input_shape2)
    conv2 = Conv2D(filter_N,kernel_size=(5,6),strides=(4,1),activation='relu',padding='valid')(inp2)
    pool2 = MaxPooling2D(pool_size=(1,7),padding='valid')(conv2)
    drop2 = Dropout(droprate2)(pool2)
    flt2 = Flatten()(drop2)
    dns = Dense(hd_layer_N, activation='relu')(flt2)
    drop = Dropout(droprate)(dns)
    outp = Dense(1, activation='sigmoid')(drop)
    model = Model(inputs=inp2,outputs=outp)

    model.load_weights(hdf5path)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=log_lkh_loss,optimizer=sgd)
   
    return model

class Annealer(object):
    """
    """
    def __init__(self, model, idx_layer, idx_unit, minimize = False):
        """
        """
        self.idx_layer = idx_layer
        self.idx_unit = idx_unit
        self.model = model
        self.minimize = minimize
        self.calcObjective = self.get_calcObjective(self.model, self.idx_layer, self.idx_unit, minimize = self.minimize)
        self.d_est = None
        
    @staticmethod
    def get_calcObjective(model, idx_layer, idx_unit, minimize = False):
        """
        """
        act_preLayer = model.layers[idx_layer - 1].output
        weights_layer, biases_layer =   model.layers[idx_layer].weights 
        if minimize:
            objective = -1.0*(K.dot(act_preLayer , weights_layer ) +  biases_layer)
        else:
            objective = K.dot(act_preLayer , weights_layer ) +  biases_layer
        calcObjective = K.function(inputs= [model.input , K.learning_phase()] , outputs= [objective])
        return calcObjective 
    
    @staticmethod
    def get_calcAct(model, idx_layer, idx_unit):
        """
        Intended to function as check on self.get_objectiv
        """
        act_preLayer = model.layers[idx_layer - 1].output
        weights_layer, biases_layer =   model.layers[idx_layer].weights 

        preact_unit = K.dot(act_preLayer , weights_layer ) +  biases_layer 
        act_unit = K.sigmoid(preact_unit)
        calcAct = K.function(inputs= [model.input, K.learning_phase()] , outputs= [act_unit])
        return  calcAct
    
    @staticmethod
    def calcTemp(t, d):
        """
        """
        T = d/np.log(t)
        return T

    def propose(self, state, proposed, debug = False ):
        """
        get proposed state
        Inputs
        -------
            state - (n_chain, 5 ,201,1) array
            proposed - should be a copy of state
        Returns
            proposed - (n_chain, 5, 201,1) array
        """
        ## Get sequence positions along etach chain to update
        indices_update = np.stack( [np.random.choice(self.indices_allowed, 
                                    size = self.n_updates, replace = False) \
                                        for _ in range(self.n_chain) ]  )
        ## extract one-hot encoding
        slices_toUpdate = np.stack( [state[i, 0:4 ,  idxs ] for  i , idxs in enumerate(indices_update)] )
        ## updae one-hot encoding
        nts_categ_toUpdate = np.argmax(slices_toUpdate  , axis = 2).squeeze()
        nts_categ_toUpdate = np.mod(nts_categ_toUpdate + \
                                    np.random.randint(1,3 , size = nts_categ_toUpdate.shape ),
                                    4)
        slices_new = np.apply_along_axis(lambda i : np.eye(4)[i,:], axis = -1,arr = nts_categ_toUpdate ).squeeze()
        ## update proposed
        for i in range(slices_new.shape[0]):
            proposed[i, 0:4, indices_update[i]] = slices_new[i][:,:,None]
        if debug:
            return indices_update, slices_toUpdate, nts_categ_toUpdate, slices_new , proposed
        return proposed
        
    def metropolisAccept(self, state, obj_state, proposed, obj_proposed, T):
        """
        """
        accept = np.exp(-(obj_state - obj_proposed)/T) > np.random.ranf(size=obj_proposed.shape) ## we are minimizing -obj_state
        state = np.where(accept.reshape(-1,1,1,1) , proposed, state)
        obj_state = np.where(accept, obj_proposed,  obj_state )
        return state , obj_state, accept.astype(int)
    
    def estimate_d(self, states_highAct , states_lowAct):
        """
        """
        act_stacked = self.getAct( [np.concatenate([states_highAct, states_lowAct], axis = 0), 0]).squeeze()
        act_states_highAct = act_stacked[ : states_highAct.shape[0] ]
        act_states_lowAct =  act_stacked[ states_highAct.shape[0] : ]
        self.d_est = act_states_highAct.max() - act_states_lowAct.min() ## -act_states_lowAct.min() - (-act_states_highAct.max()) 
        return  self.d_est
        
    def run(self, n_iter, state_init, n_updates, d=None, indices_allowed=None, sampleInterval = 10):
        """
        """
        self.n_chain = state_init.shape[0]
        self.n_updates = n_updates
        self.sampleInterval = sampleInterval
        if indices_allowed is None:
            self.indices_allowed = np.arange(0, state_init.shape[2])
        else:
            self.indices_allowed = indices_allowed
        if not (d is None):
            self.d_est = d
        
        state = state_init.copy()
        proposed = state.copy()
        obj_state =self.calcObjective([state, 0])[0].squeeze()
        samples = np.zeros( (n_iter//sampleInterval,)+ state.shape, dtype = np.float32 )
        nAccept = np.zeros( (self.n_chain,),dtype= int)
        self.maxObjective = np.ones( (self.n_chain,),dtype= np.float32 )*(-99999.0)
        i = 0
        for t in range(0, n_iter):
            T = self.calcTemp(t+2, self.d_est)  ## +2 to prevent division by 0
            proposed = self.propose(state, proposed)
            obj_proposed =  self.calcObjective([proposed, 0])[0].squeeze()
            state, obj_state, accept = self.metropolisAccept( state, obj_state, proposed, obj_proposed, T )
            proposed = state.copy()
            nAccept += accept
            self.maxObjective = np.where(obj_state > self.maxObjective, obj_state, self.maxObjective)
            ## store iteration results
            if t % sampleInterval == 0:
                samples[i] = state
                i+=1
        return samples, nAccept/(n_iter) , -2*(int(self.minimize) -0.5)*self.maxObjective
        
    def run_save(self, n_iter, state_init, n_updates, d=None, indices_allowed=None, sampleInterval = 10,
           saveInterval = None, f_save = "tmp.pkl", n_keep = 10000 ):
        """
        n_iter - number of iterations
        n_updates - number of NT to propose changing per iteration
        sampleInterval - iteration period for sampling chain
        saveInterval - iteration period for saving results (must be multiple of sample interval)
        n_keep - number of samples at end of chain to keep. Must have 2*n_keep*sampleInterval <  saveInterval
                (data collected across last n_keep*sampleIntervaliterations)
        """
        self.n_chain = state_init.shape[0]
        self.n_updates = n_updates
        self.sampleInterval = sampleInterval
        if saveInterval is None:
            saveInterval = n_iter  ## don't save
        if (saveInterval % sampleInterval) != 0:
                raise Exception("saveInterval  must be multiple of sampleInterval" )
        if n_keep is None:
            n_keep = saveInterval // sampleInterval  ## keep every sample in saveInterval
        if indices_allowed is None:
            self.indices_allowed = np.arange(0, state_init.shape[2])
        else:
            self.indices_allowed = indices_allowed
        if not (d is None):
            self.d_est = d
        
        state = state_init.copy()
        proposed = state.copy()
        obj_state =self.calcObjective([state, 0])[0].squeeze()
        samples = np.zeros( ( n_keep ,)+ state.shape, dtype = np.float32 )
        nAccept = np.zeros( (self.n_chain,),dtype= int)
        nAccept_save = np.zeros( (self.n_chain,),dtype= int)
        self.maxObjective = np.ones( (self.n_chain,),dtype= np.float32 )*(-99999.0)
        i = 0
        for t in range(0, n_iter):
            T = self.calcTemp(t + 2, self.d_est) ## +2 to prevent division by 0
            proposed = self.propose(state, proposed)
            obj_proposed =  self.calcObjective([proposed, 0])[0].squeeze()
            state, obj_state, accept = self.metropolisAccept( state, obj_state, proposed, obj_proposed, T )
            nAccept += accept
            proposed = state.copy()
            self.maxObjective = np.where(obj_state > self.maxObjective, obj_state, self.maxObjective)
            ## Collect samples from beginning of chain
            if t <= sampleInterval*n_keep:
                if t == sampleInterval*n_keep:
                    data = OrderedDict([(0 , { "iter_idxs" : np.arange(t - n_keep*sampleInterval, t, sampleInterval),
                                   "accFrac":  nAccept /( t +1 ),
                                  "objective_best": -2*(int(self.minimize) -0.5)*self.maxObjective ,
                                  "samples": samples  }) ])
                    print("Writing results after iterations {}".format(t))
                    f = gzip.open(f_save + "_0.pklz" , 'wb')
                    pickle.dump(data, f)
                    f.close()
                    ## Reset
                    samples = np.zeros( ( n_keep ,)+ state.shape, dtype = np.float32 )
                    i=0
                else:
                    if t % sampleInterval == 0:
                        samples[i] = state
                        i+=1
            ## Collect samples every saveInterval iterations
            if (t % saveInterval == 0) and t > 0:
                ## save results and reset
                nAccept_save = nAccept - nAccept_save
                data_write = OrderedDict([(t , { "iter_idxs" : np.arange(t - n_keep*sampleInterval,t , sampleInterval) ,
                                   "accFrac":  nAccept_save / saveInterval  ,
                                  "objective_best": -2*(int(self.minimize) -0.5)*self.maxObjective ,
                                  "samples": samples  }) ])
#                if os.path.isfile(f_save):
#                    f = open(f_save, 'rb')
#                    data = pickle.load(f)
#                    f.close()
#                    data.update(data_write)
#                else:
#                    data = data_write
                print("Writing results after iterations {}".format(t))
                f = gzip.open(f_save + "_{:.1e}.pklz".format(t) , 'wb')
                pickle.dump(data_write, f)
                f.close()
                ## Reset
                samples = np.zeros( ( n_keep ,)+ state.shape, dtype = np.float32 )
                i=0
                print("sleeping 3 min")
                time.sleep(60*3)
            if (saveInterval - (t % saveInterval))//sampleInterval <= n_keep: ## within n_keep*sampleInterval iteration of a save point 
                ## store iteration results
                if t % sampleInterval == 0:
                    samples[i] = state
                    i+=1
        return  samples, nAccept/n_iter , -2*(int(self.minimize) -0.5)*self.maxObjective
    
    ## PLOTTING
    @staticmethod
    def KLdivRolling_fromSamples( samples, window_width, window_step, q = np.array(4*[0.25])):
        """
        samples-(nSamples ,4, seqLen). each slice samples[i, : , k] is a one hot encoding of sequence
        """
        q = np.repeat(q[:, None], repeats = samples.shape[2],  axis= 1 )

        KLdivRolling = np.zeros(shape =( -(-(samples.shape[0]-(window_width-1))//window_step), samples.shape[2]), dtype = float ) ## (n_windows, seqLen)
        windowCenters = np.array(range(0, samples.shape[0]-(window_width-1), window_step)) + window_width//2
        i = 0
        for window_start in range(0, samples.shape[0]-(window_width-1), window_step):
            p = samples[window_start: window_start+window_width].mean(axis = 0)
            KLdivRolling[i] = np.array( [ entropy(p[:,j], q[:,j]) for  j in range(q.shape[-1]) ] )
            i+=1 
        return  KLdivRolling,  windowCenters

    def plot_KLdivRolling(self, samples, window_width, window_step, q = np.array(4*[0.25]), t_to_T = None ,
                         ax = None, figsize = (7.5,4), logScale_T= True, kws_legend = {"loc": 0} ):
        """
        t_to_T - a function for converting window_center to Temperature
        """
        if t_to_T is None:
            t_to_T = lambda t : self.calcTemp( (t+1)*self.sampleInterval, self.d_est)
            print("Using t_to_T function with d = self.d_est = {:.2e}".format(self.d_est))
        
        if len(samples.shape) == 4:
            ## assume axis 1 corresponds to different SA instances and calc KLdiv for each separately
            KLdivRolling_list = []
            for i in range(samples.shape[1]):
                KLdivRolling, windowCenters = self.KLdivRolling_fromSamples( samples = samples[:,i], window_width= window_width, 
                                                                       window_step = window_step, q = q)
                KLdivRolling_list.append(KLdivRolling)
            KLdivRolling = np.stack(KLdivRolling_list)
            print(KLdivRolling.shape)
        else:
            KLdivRolling, windowCenters = self.KLdivRolling_fromSamples( samples = samples, window_width= window_width, 
                                                                       window_step = window_step, q = q)
            KLdivRolling = KLdivRolling.reshape((1,) + KLdivRolling.shape )

        T = np.array( list(map(t_to_T , windowCenters)) )
        if ax is None:
            fig , ax = plt.subplots(figsize = figsize)
        for i in range(KLdivRolling.shape[0]):
            ax.plot(T , KLdivRolling[i].sum(axis = -1), marker = 'o', label = "instance {}".format(i))
        if logScale_T:
            ax.set_xscale('log')
        ax.legend(**kws_legend)
        ax.set_xlabel('T')
        ax.set_ylabel("KL divergence")
        return ax
    
    def histogram_d(self, states_highAct, states_lowAct, ax = None, figsize = (7.5, 4), bins = 40,
                   batchSize = 1000, returnDiffs = False ):
        """
        Plot a histogram of all pairwise distances between  states_highAct and states_lowAct
        """
        act_states_highAct = np.concatenate([ self.calcObjective([states_highAct[i: i + batchSize], 0])[0] 
                                             for i in range(0,states_highAct.shape[0], batchSize)]).squeeze()
        act_states_lowAct = np.concatenate([ self.calcObjective([states_lowAct[i: i + batchSize], 0])[0] 
                                            for i in range(0,states_lowAct.shape[0], batchSize)] ).squeeze()
        diffs =  (act_states_highAct[:,None] - act_states_lowAct).ravel()
        if returnDiffs:
            return diffs
        else:
            if ax is None:
                fig , ax = plt.subplots(figsize = figsize)
            ax.hist(diffs, bins = bins, histtype = 'step')
            ax.set_xlabel("(Preact. HighME) - (Preact LowME)")
            return ax
        
#############################################################################
### Script:
#####################################################################
### PARSE CMD LINE
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run SA algorithm on ANN output initialized at specific itputs")
    parser.add_argument( "--inOut" ,  type = str , 
             default = "/home/groups/song/songlab2/shared/Somang_Alex_share/forAlex_6conditions/pkls/input_output.pk" ,
                   help = "pickled file of network inputs / outputs created by Somang" )
    parser.add_argument( "--inOut_idxs", required = True,  type = str , 
                        help = "text files with indices of InOut to use for SA initialization" )
    parser.add_argument( "-a" , "--architecture",  type = str ,
                        help = ".py file with function loadModel such that model = loadModel(hdf5_path)" )
    parser.add_argument( "-p" ,"--params", type = str, help= "path to hdf5 file with network params" )
    
    parser.add_argument("--n_iter", type = int, default = 500000 )
    parser.add_argument("--n_save" , type =int, default = 10000 )
    parser.add_argument( "-d" , type = float, default = 0.5 ) 
    parser.add_argument("--n_updates" , type = int , default = 4 )
    parser.add_argument("--sampleInterval" ,type =int , default = 2 )
    parser.add_argument("-o", "--fo", type = str , help = "output file — a pickled dictionary")
    parser.add_argument("--minimize" , action = "store_true" , help = "Run SA to minimize, rather than maximize network output")
    parser.add_argument( "--saveInterval", type = int , 
                    help = "Save intermdiate results evenly spaced iteration intervals. Must have saveInterval % sampleInterval = 0" )

    args =  parser.parse_args()
#dummyArgs = '-p ../../forAlex_6conditions/weights/3A23L/243-2.6437-2.6117.hdf5 \
#--inOut_idxs  ./initializations/3A23L_highest50.txt \
#--n_iter 500000 --n_save 10000  -d0.4 --n_updates 4  --fo 3A23Lsamples_max_tmp.pkl'
#args =  parser.parse_args(dummyArgs.split())
    
    ########################################
    ## LOAD NETWORK
    print("Loading network")
    if args.architecture is not None:
        f = open(args.architecture, 'r')
        print("\tusing function loadModel from {}".format(args.architecture))
        script = f.read()
        f.close()
        exec(script)
        del script
    else: 
        print("\tUsing default loadModel function")
    model = loadModel( args.params )
    print("Done")
    
    ##########################################
    ## Load initializing inputs
    print("Loading initializing inputs")
    inOut_idxs = readListFromFile(args.inOut_idxs)
    inOut_idxs = np.array([int(x) for x in inOut_idxs ])
    
    f = open(args.inOut, 'rb')
    NN_in = pickle.load(f)[0][inOut_idxs]
    f.close()
    print("Done")
    
    #########################################
    ## RUN SA
    print("Running SA")
    annealer = Annealer(model, idx_layer=7, idx_unit=0,  minimize = args.minimize)
    
    if args.saveInterval is None:
        t_i = time.time()
        samples, accFrac, objective_best = annealer.run( n_iter = args.n_iter ,
                                                  state_init = NN_in,
                                                n_updates = args.n_updates, 
                                                d= args.d, 
                                                indices_allowed = indices_allowed, 
                                                sampleInterval = args.sampleInterval)
        print("SA done took: {:.2f} s".format(time.time() - t_i))
        ######################
        ## Write output
        print("Writing output")
        if os.path.isfile(args.fo):
            f = open( args.fo , 'rb' )
            results_dict = pickle.load(f)
            f.close()
        else:
            results_dict = OrderedDict([])

        for i in range(len(inOut_idxs)):
            results_dict[inOut_idxs[i]] = { "accFrac" : accFrac[i] ,
                                           "objective_best": objective_best[i] , 
                                           "samples" : (samples[-1*args.n_save: , i ]).squeeze()}    
        f = open( args.fo , 'wb' )
        pickle.dump(results_dict , f)
        f.close()
        print("Done!")
    else:
        t_i = time.time()
        samples1, accFrac1, maxObjective1 = annealer.run_save(n_iter = args.n_iter + 1, state_init = NN_in ,#+1 insures last saveInterval written
                                                                n_updates = args.n_updates, d = args.d,
                                                                indices_allowed=indices_allowed,
                                                                sampleInterval = args.sampleInterval ,
                                                                saveInterval = args.saveInterval , 
                                                                f_save = args.fo, n_keep = args.n_save)
        print("SA done took: {:.2f} s".format(time.time() - t_i))
    
