
from __future__ import  division
import numpy as np
import pandas as pd
from collections import OrderedDict
from numpy.linalg import multi_dot
import numpy.linalg as LA
import multiprocessing as mp

import matplotlib.pyplot as plt
import seaborn as sns

class LRC_preprocessor(object):
    """
    Preprocess data for training/prediction using LogReg_Continuous object
    
    To preprocess a single dataset use 
        lrc_pp = LRC_preprocessor( n_features = 30, start = -0.1, stop = 1.1, stdz_width = 0.1)
        lrc_pp.set_stdz_params(data)
        lrc_pp.add_data(dataTrain_byClass, dataTest_byClass)
        dataset = lrc_pp.dataset  ## the preprocessed data
    To preprocess for cross-validation use:
        lrc_pp = LRC_preprocessor( n_features = 30, start = -0.1, stop = 1.1, stdz_width = 0.1)
        stdz_params_CV = lrc_pp.add_CV_datasets(data_byClass, n_folds , stdz_name) ## the parameters used for preprocssing of each fold
        dataset = lrc_pp.dataset  ## the preprocessed data
    """
    def __init__(self,  n_features = 30, start = -0.1, stop = 1.1, stdz_width = 0.1, multipleEntries = "mean", stdz_method = "mean_std", n_process = 1 ):
        self.n_features = n_features 
        self.start = start
        self.stop = stop
        self.stdz_width = stdz_width
        self.stdz_params = None
        self.stdz_name = None
        self.bins = np.hstack([np.linspace(start ,stop, n_features+1)[0:-1,None],
                                 np.linspace(start ,stop, n_features+1)[1:,None]])
        self.bin_centers = np.round( self.bins.mean(axis = 1) , 3)
        self.dataset =  pd.DataFrame(index = [], 
                                    columns =  pd.MultiIndex.from_tuples([("feature", x ) for x in  self.bin_centers ] + \
                                             [("metaData", "y"), ("metaData", "cls_name"), ("metaData", "geneID"), 
                                              ("metaData",  "stdz_name"), ("metaData", "fold"), ("metaData" , "train/test")])
                                    )
        self.n_folds = None
        self.multipleEntries = multipleEntries
        self.stdz_method = stdz_method
        self.n_process = n_process
                                        
    def set_stdz_params(self, data, name = None):
        """
        Inputs
        ------
            data - list of pd.DataFrames with columns  [position_rel_scaled, est_p]
        """
        
        self.stdz_params = self.calc_stdz_params(data, n_features = self.n_features, 
                                                 start = self.start, stop = self.stop, 
                                                 stdz_width = self.stdz_width, stdz_method = self.stdz_method, n_process = self.n_process )
        if name is not None:
            self.stdz_name = name
        else:
            if self.stdz_name is None:
                self.stdz_name = "0"
            else:
                self.stdz_name = str(int(self.stdz_name) +1 )
        return None
    
    def add_data(self, dataTrain_byClass, dataTest_byClass, nan_to_zero = True, metadata = {"fold": 0, "stdz_type": "0"}, tanh_scale = 0.):
        """
        Append observations to self.X
        Inputs
        --------
        dataTrain_byClass - OrderedDict with class names as keys and dictionaries 
                        of observations as values, e.g.
                        OrderedDict([  ("className0", OrderedDict([ ("geneA1", data_df ),
                                                                    ("geneA2", data_df), ... ])
                                        ),
                                        ("className1", OrderedDict([ ("geneB1", data_df ),
                                                                    ("geneB2", data_df), ... ])
                                        ),
                                    ])
                         where data_df is pd.DataFrame with columns: [position_rel_scaled, est_p]
        dataTest_byClass - Like datatrain_byClass
        """
        if self.stdz_params is None:
            raise Exception("run set_stdz_params method first")
        data_train = self.preprocess_data( dataTrain_byClass, self.stdz_params,  multipleEntries = self.multipleEntries, n_process= self.n_process, tanh_scale= tanh_scale)
        data_train.loc[: , ("metaData", "train/test")]  = "train"  
        for k, v in  metadata.items():
            data_train.loc[:, ("metaData", k)] = v                         
        data_test = self.preprocess_data( dataTest_byClass, self.stdz_params, multipleEntries = self.multipleEntries, n_process = self.n_process , tanh_scale= tanh_scale)                          
        data_test.loc[: , ("metaData", "train/test")]  = "test"
        for k, v in  metadata.items():
            data_test.loc[:, ("metaData", k)] = v                            
        self.dataset = pd.concat([ self.dataset , data_train , data_test], axis = 0, ignore_index = True)                   
        return None                                                                   

    def add_CV_datasets(self, data_byClass, n_folds , stdz_name,  tanh_scale = 0. ):
        """
        data_byClass - OrderedDict with class names as keys and dictionaries 
                        of observations as values, e.g.
                        OrderedDict([  ("className0", OrderedDict([ ("geneA1", data_df ),
                                                                    ("geneA2", data_df), ... ])
                                        ),
                                        ("className1", OrderedDict([ ("geneB1", data_df ),
                                                                    ("geneB2", data_df), ... ])
                                        ),
                                    ])
                         where data_df is pd.DataFrame with columns: [position_rel_scaled, est_p]
        n_folds - int
        """
        if self.n_folds is None:
            self.n_folds = n_folds
        else:
            if self.n_folds != n_folds:
                raise Exception( "self.n_folds is already set" )
            
        geneIDs = [geneID for x in data_byClass.values() for geneID in x.keys() ]
        np.random.shuffle(geneIDs)
        fold_size = -( (-len(geneIDs))//self.n_folds ) 
        fold_to_geneID = OrderedDict([(i, geneIDs[i*fold_size : (i+1)*fold_size]) for i in range(self.n_folds)  ] )
        print(OrderedDict([ (k, len(v) ) for k,v in fold_to_geneID.items()] ))
        stdz_params_CV = pd.DataFrame(index = [], columns = [ "bin_lbound" , "bin_ubound", "mean", "std" , "stdz_name" , "fold" ])
        for fold , genes in  fold_to_geneID.items():
                                       
            dataTrain_byClass = OrderedDict([ (className, OrderedDict([(geneID, x[geneID]) for geneID in  x.keys() if geneID not in genes ]))
                                              for className, x in  data_byClass.items() ] )
            print([ len(x) for x in dataTrain_byClass.values() ])
            dataTest_byClass = OrderedDict([ (className, OrderedDict([(geneID, x[geneID]) for geneID in  x.keys() if geneID in genes ]))
                                              for className, x in  data_byClass.items() ] )
            ## data {list} -- list of pd.DataFrames with columns  [position_rel_scaled, est_p]
            self.stdz_params = self.calc_stdz_params( [ g_df for gene_dict in dataTrain_byClass.values() for g_df in  gene_dict.values()  ] , 
                                                        n_features = self.n_features,  start = self.start, stop = self.stop, 
                                                        stdz_width = self.stdz_width, stdz_method = self.stdz_method , n_process=self.n_process )                             
            self.add_data(dataTrain_byClass, dataTest_byClass, nan_to_zero = True, metadata = {"fold": fold, "stdz_name": stdz_name }, tanh_scale = tanh_scale )
                  
            self.stdz_params.loc[: , "stdz_name"] = stdz_name   
            self.stdz_params.loc[: , "fold"] = fold
            stdz_params_CV = pd.concat( [ stdz_params_CV ,  self.stdz_params] , axis = 0 )
                
        return  stdz_params_CV    

    @staticmethod
    def _standardize_geneFeatures_combineMean(df, bins, stdz_params ):
        """[summary]
        
        Arguments:
            df {pd.DataFrame} -- DataFrame describing methylation in meta gene coordinates. Must have columns ["position_rel_scaled", "est_p" ]
            bins {[type]} -- [description]
            geneID {string} -- unique identifier for gene
        """
        sort_idxs = np.argsort( df["position_rel_scaled"].values)
        position_idxs = np.copy( df["position_rel_scaled"].values[sort_idxs] )
        est_p =  np.copy(df["est_p"].values[sort_idxs] )
        est_p_stdz = np.zeros( len(bins) )
        for j, bin_ends in enumerate(bins):
            b_l , b_u =bin_ends
            idx_lower = np.searchsorted( position_idxs, b_l, side='left')
            idx_upper = np.searchsorted( position_idxs, b_u, side='right')
            est_p_inBin = est_p[idx_lower: idx_upper ]
            if len(est_p_inBin ) == 0:
                est_p_stdz[j] = 0
            else:
                est_p_stdz[j] = np.mean( (est_p_inBin - stdz_params.loc[j, "mean"]) / stdz_params.loc[j, "std"]  )
        return est_p_stdz

    @staticmethod
    def _standardize_geneFeatures_combineSum(df, bins, stdz_params ):
        """[summary]
        
        Arguments:
            df {pd.DataFrame} -- DataFrame describing methylation in meta gene coordinates. Must have columns ["position_rel_scaled", "est_p" ]
            bins {[type]} -- [description]
            geneID {string} -- unique identifier for gene
        """
        sort_idxs = np.argsort( df["position_rel_scaled"].values)
        position_idxs = np.copy( df["position_rel_scaled"].values[sort_idxs] )
        est_p =  np.copy(df["est_p"].values[sort_idxs] )
        est_p_stdz = np.zeros( len(bins) )
        for j, bin_ends in enumerate(bins):
            b_l , b_u =bin_ends
            idx_lower = np.searchsorted( position_idxs, b_l, side='left')
            idx_upper = np.searchsorted( position_idxs, b_u, side='right')
            est_p_inBin = est_p[idx_lower: idx_upper ]
            if len(est_p_inBin ) == 0:
                est_p_stdz[j] = 0
            else:
                est_p_stdz[j] = np.sum( (est_p_inBin - stdz_params.loc[j, "mean"]) / stdz_params.loc[j, "std"]  )
        return est_p_stdz

    @staticmethod
    def _standardize_geneFeatures_combineSum_tanh(df, bins, stdz_params,  tanh_scale ):
        """[summary]
        
        Arguments:
            df {pd.DataFrame} -- DataFrame describing methylation in meta gene coordinates. Must have columns ["position_rel_scaled", "est_p" ]
            bins {[type]} -- [description]
            geneID {string} -- unique identifier for gene
        """
        sort_idxs = np.argsort( df["position_rel_scaled"].values)
        position_idxs = np.copy( df["position_rel_scaled"].values[sort_idxs] )
        est_p =  np.copy(df["est_p"].values[sort_idxs] )
        est_p_stdz = np.zeros( len(bins) )
        for j, bin_ends in enumerate(bins):
            b_l , b_u =bin_ends
            idx_lower = np.searchsorted( position_idxs, b_l, side='left')
            idx_upper = np.searchsorted( position_idxs, b_u, side='right')
            est_p_inBin = est_p[idx_lower: idx_upper ]
            if len(est_p_inBin ) == 0:
                est_p_stdz[j] = 0
            else:
                est_p_stdz[j] = np.sum( 
                                    np.tanh( 
                                        (est_p_inBin - stdz_params.loc[j, "mean"])/(stdz_params.loc[j, "std"]*tanh_scale ) 
                                            )  )
        return est_p_stdz

    @staticmethod
    def preprocess_data( data_byClass, stdz_params, multipleEntries = "mean" , n_process=1, tanh_scale = 0.0):
        """
        Inputs
        --------
        data_byClass - OrderedDict with class names as keys and dictionaries 
                        of observations as values, e.g.
                        OrderedDict([  ("className0", OrderedDict([ ("geneA1", data_df ),
                                                                    ("geneA2", data_df), ... ])
                                        ),
                                        ("className1", OrderedDict([ ("geneB1", data_df ),
                                                                    ("geneB2", data_df), ... ])
                                        ),
                                    ])
                         where data_df is pd.DataFrame with columns: [position_rel_scaled, est_p]
        """
        n_obs = sum([len(v) for v in data_byClass.values()])
        bins = stdz_params.loc[:, ["bin_lbound" , "bin_ubound" ] ].values
        bin_centers = np.round( bins.mean(axis = 1) , 3)
        ds = pd.DataFrame(data = np.zeros( shape=(n_obs, len(bin_centers)+3), dtype = float ) ,
                          columns =  pd.MultiIndex.from_tuples([("feature", x ) for x in bin_centers ] + \
                                                 [("metaData", "y"), ("metaData", "cls_name"), ("metaData", "geneID") ] )
                         )
        features_stdz = []
        for clsID, cls_obsDict in enumerate(data_byClass.items()):
            cls_name , obs_dict =  cls_obsDict
            if multipleEntries == "mean":
                if n_process == 1:
                    for geneID, df in obs_dict.items():
                        features_stdz.append(LRC_preprocessor._standardize_geneFeatures_combineMean(df, bins, stdz_params ) ) 
                else:
                    with mp.Pool(processes=n_process) as pool:
                        features_stdz.extend(  pool.starmap( LRC_preprocessor._standardize_geneFeatures_combineMean, [ (df, bins, stdz_params) for df in obs_dict.values() ] ) )
            elif multipleEntries == "sum":
                if tanh_scale > 0.:
                    if n_process == 1:
                        for geneID, df in obs_dict.items():
                            features_stdz.append(LRC_preprocessor._standardize_geneFeatures_combineSum_tanh(df, bins, stdz_params, tanh_scale = tanh_scale ) )
                    else:
                        with mp.Pool(processes=n_process) as pool:
                            features_stdz.extend(  pool.starmap( LRC_preprocessor._standardize_geneFeatures_combineSum_tanh, [ (df, bins, stdz_params, tanh_scale ) for df in obs_dict.values() ] ) )
                else:
                    if n_process == 1:
                        for geneID, df in obs_dict.items():
                            features_stdz.append(LRC_preprocessor._standardize_geneFeatures_combineSum(df, bins, stdz_params ) )
                    else:
                        with mp.Pool(processes=n_process) as pool:
                            features_stdz.extend(  pool.starmap( LRC_preprocessor._standardize_geneFeatures_combineSum, [ (df, bins, stdz_params) for df in obs_dict.values() ] ) )
            else:
                raise ValueError("{} is not a valid value of multipleEntries".format(multipleEntries))
        ds.iloc[: , 0:len(bin_centers) ] = np.stack(features_stdz)
        ds[ ("metaData", "y") ] =  [clsID for clsID, obs_dict in enumerate(data_byClass.values()) for dummy in range(len(obs_dict )) ]  
        ds[ ("metaData", "cls_name") ] =  [ cls_name for cls_name, obs_dict in data_byClass.items() for dummy in range(len(obs_dict )) ]  
        ds[ ("metaData", "geneID") ] =  [ geneID for  obs_dict in data_byClass.values()  for geneID in  obs_dict.keys()  ]
        return ds

    @staticmethod
    def calc_stdz_params(data,  n_features = 30, start = -0.1, stop = 1.1, stdz_width = 0.1, stdz_method = "mean_std" , n_process = 1 ):
        """Construct dataframe with parameters for standardizing regression inputs. Returned pd.DataFrame has columns [ "bin_lbound" , "bin_ubound", "mean","std" ]
        
        Arguments:
            data {list} -- list of pd.DataFrames with columns  [position_rel_scaled, est_p]

        Keyword Arguments:
            _features {int} -- [description] (default: {30})
            start {float} -- [description] (default: {-0.1})
            stop {float} -- [description] (default: {1.1})
            stdz_width {float} -- [description] (default: {0.1})
            stdz_method {str} -- [description] (default: {"mean_std"})
        Returns:
            stdz_params - pd.DataFrame with columns [ "bin_lbound" , "bin_ubound", "mean","std" ]
        """
        bins = np.hstack([np.linspace(start, stop, n_features+1)[0:-1,None],
                         np.linspace(start, stop, n_features+1)[1:,None]])
        bins_stdz = np.array([ [center -  stdz_width/2.0 , center +  stdz_width/2.0 ] for center in bins.mean(axis = 1)])
        position_and_estP_df = pd.concat( data, axis = 0 ) 
        sort_idxs =  np.argsort(  position_and_estP_df["position_rel_scaled"].values )
        positions_sorted = np.copy( position_and_estP_df["position_rel_scaled"].values[ sort_idxs] )
        est_p_sorted =  np.copy( position_and_estP_df["est_p"].values[ sort_idxs] )
        if stdz_method == "mean_std":
            if n_process == 1:
                stdz_params_data = [ LRC_preprocessor._inSlice_mean_std(bins_stdz[i,0], bins_stdz[i,1], positions_sorted, est_p_sorted) 
                                                for i in range(len(bins_stdz)) ]
                stdz_params_data  = np.hstack( [bins, np.array(stdz_params_data)  ] )                            
                stdz_params = pd.DataFrame(data = stdz_params_data , columns = [ "bin_lbound" , "bin_ubound", "mean","std"] )
            elif n_process  > 1:
                with mp.Pool(processes=n_process) as pool:
                     stdz_params_data =  pool.starmap( LRC_preprocessor._inSlice_mean_std, 
                                            [ (bins_stdz[i,0], bins_stdz[i,1], positions_sorted, est_p_sorted) for  i in range(len(bins_stdz)) ] )
                stdz_params_data  = np.hstack( [bins, np.array(stdz_params_data)  ] )    
                stdz_params = pd.DataFrame(data = stdz_params_data , columns = [ "bin_lbound" , "bin_ubound", "mean","std"] )
            else:
                raise ValueError( "n_process must be >= 1" )
        elif stdz_method == "med_mad":
            if n_process == 1:
                stdz_params_data = [ LRC_preprocessor._inSlice_median_MAD(bins_stdz[i,0], bins_stdz[i,1], positions_sorted, est_p_sorted) 
                                                for i in range(len(bins_stdz)) ]
                stdz_params_data  = np.hstack( [bins, np.array(stdz_params_data)  ] )                            
                stdz_params = pd.DataFrame(data = stdz_params_data , columns = [ "bin_lbound" , "bin_ubound", "mean", "std" ] )
            elif n_process > 1:
                with mp.Pool(processes=n_process) as pool:
                     stdz_params_data =  pool.starmap( LRC_preprocessor._inSlice_median_MAD, 
                                            [ (bins_stdz[i,0], bins_stdz[i,1], positions_sorted, est_p_sorted) for  i in range(len(bins_stdz)) ] )
                stdz_params_data  = np.hstack( [bins, np.array(stdz_params_data)  ] )    
                stdz_params = pd.DataFrame(data = stdz_params_data , columns = [ "bin_lbound" , "bin_ubound", "mean","std"] )
            else:
                raise ValueError( "n_process must be >= 1" )
        elif stdz_method == "med_mad.above":
            if n_process == 1:
                stdz_params_data = [ LRC_preprocessor._inSlice_median_MAD_above(bins_stdz[i,0], bins_stdz[i,1], positions_sorted, est_p_sorted) 
                                                for i in range(len(bins_stdz)) ]
                stdz_params_data  = np.hstack( [bins, np.array(stdz_params_data)  ] )                            
                stdz_params = pd.DataFrame(data = stdz_params_data , columns = [ "bin_lbound" , "bin_ubound", "mean", "std" ] )
            elif n_process > 1:
                with mp.Pool(processes=n_process) as pool:
                     stdz_params_data =  pool.starmap( LRC_preprocessor._inSlice_median_MAD_above, 
                                            [ (bins_stdz[i,0], bins_stdz[i,1], positions_sorted, est_p_sorted) for  i in range(len(bins_stdz)) ] )
                stdz_params_data  = np.hstack( [bins, np.array(stdz_params_data)  ] )    
                stdz_params = pd.DataFrame(data = stdz_params_data , columns = [ "bin_lbound" , "bin_ubound", "mean","std"] )
        else:
            raise ValueError("{} is not a valid value for stdz_method".format(stdz_method))
        return  stdz_params

    @staticmethod
    def _inSlice_median_MAD_above( b_l , b_u,  position_idxs , est_p  ):
        """Calcuate median and mad of est_p values occurin at position_indices in range [b_l, b_u) 
        Arguments:
            bin_l {float} -- [description]
            bin_u {float} -- [description]
            positions {np.ndarray} -- 1d array MUST BE SORTED
            est_p {np.ndarray} -- 1d array MUST FOLLOW SORT ORDER OF positions
        """
        idx_lower = np.searchsorted( position_idxs, b_l, side='left')
        idx_upper = np.searchsorted( position_idxs, b_u, side='right')
        median =  np.median( est_p[idx_lower: idx_upper] ) 
        est_p_minus_median = est_p[idx_lower: idx_upper] - median  
        mad_above = np.median(  est_p_minus_median[est_p_minus_median >= 0.0]  )
        return  median, mad_above

    @staticmethod
    def _inSlice_median_MAD( b_l , b_u,  position_idxs , est_p  ):
        """Calcuate median and mad of est_p values occurin at position_indices in range [b_l, b_u) 
        Arguments:
            bin_l {float} -- [description]
            bin_u {float} -- [description]
            positions {np.ndarray} -- 1d array MUST BE SORTED
            est_p {np.ndarray} -- 1d array MUST FOLLOW SORT ORDER OF positions
        """
        idx_lower = np.searchsorted( position_idxs, b_l, side='left')
        idx_upper = np.searchsorted( position_idxs, b_u, side='right')
        median =  np.median( est_p[idx_lower: idx_upper] ) 
        mad = np.median(  np.abs(est_p[idx_lower: idx_upper] - median )   )
        return  median, mad
    @staticmethod
    def _inSlice_mean_std(b_l , b_u,  position_idxs , est_p  ):
        """Calcuate mean and std of est_p values occurin at position_indices in range [b_l, b_u) 
        Arguments:
            bin_l {float} -- [description]
            bin_u {float} -- [description]
            positions {np.ndarray} -- 1d array MUST BE SORTED
            est_p {np.ndarray} -- 1d array MUST FOLLOW SORT ORDER OF positions
        """
        idx_lower = np.searchsorted( position_idxs, b_l, side='left')
        idx_upper = np.searchsorted( position_idxs, b_u, side='right')
        mean =  np.mean( est_p[idx_lower: idx_upper] ) 
        stdz = np.std( est_p[idx_lower: idx_upper]  ) 
        return  mean , stdz 