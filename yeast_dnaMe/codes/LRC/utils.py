from __future__ import  division
import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.optimize import root as find_root 
from numpy.linalg import multi_dot
import numpy.linalg as LA
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from LRC.classifier import LogReg_Continuous

import matplotlib.pyplot as plt
import seaborn as sns

def run_CV(CV_data, categ_feature= "" , lmbda_vals =[ 0.001,0.01, 0.1,1.0 , 2.0 ],
         performanceStat = ["acc", "auROC", "auPRC", "auPRC_baseline" ], categ_levels_measure = [], return_preds = False ):
    """Run Cross validation 
    
    Arguments:
        CV_data {pd.DataFrame} -- columns - pd.Multiindex : (features, categ_level0 ), (features, categ_level1 ) , 
                                                            (features, categ_level2 ) , ... (features, binCenter0) , (features, binCenter1),
                                                            (metaData, y) , (metaData, train/test) , (metaData, fold)
                                Note: categorical variable columns MUST proceed continuous variable columns (e.g those labeled by bin index)
                                index- arbitrary

    Keyword Arguments:
        categ_feature  {str} -- a column of feature that encodes a categorical variable or an empyt string if no categorical features
        lmbda_vals {list} -- [description] (default: {[ 0.001, 0.1,1.0 , 2.0 , 4.0 , 8.0, 16.0 , 32.0, 64.0, 128.0 ]})
        performanceStat {list} -- List of test set statistics to mesure (default: {["acc", "auROC", "auPRC", "auPRC_baseline" ]})
        categ_levels_measure {list} -- [description] (default: {[]})

    Returns:
        pd.DataFrame -- index- fold_IDs columns performance stats for each lambda and element of categ_levels_measure 
    """
    if CV_data.isna().any().any():
        raise ValueError("CV_data should not contain Nans" )
    fold_IDs = sorted( CV_data.loc[:, ("metaData" , "fold")].unique(), key = lambda x: int(x) )
    if isinstance(performanceStat , str ):
        performanceStat = [performanceStat]
    if categ_feature:
        categ_dummies = pd.get_dummies(CV_data[("feature" , categ_feature)])
        n_categ_levels = categ_dummies.shape[1]
        categ_dummies.columns = pd.MultiIndex.from_product( [("feature_categ",) , categ_dummies.columns ] )
        CV_data =pd.concat([ categ_dummies, CV_data.drop(columns=[("feature", categ_feature)]) ] , axis = 1)  
        CV_data = CV_data.rename( columns = {"feature_categ":  "feature"} , level = 0 )
    else:
        n_categ_levels = 0
    ## Setup dataframe for storing results
    if categ_levels_measure:
        CV_results = pd.DataFrame( index = fold_IDs, 
                                columns =  pd.MultiIndex.from_product([tuple(performanceStat), 
                                                                        tuple(lmbda_vals), tuple(["marginal"]+categ_levels_measure)]) ,
                                  dtype = float )
    else:
        CV_results = pd.DataFrame( index = fold_IDs, 
                                columns =  pd.MultiIndex.from_product([tuple(performanceStat), tuple(lmbda_vals)]) ,
                                  dtype = float )
    if return_preds:
        preds_dict = OrderedDict([])
    for lmbda in lmbda_vals:
        CV_data.loc[:, ("metaData", "model_out")] =  -1.
        for fold_id in  CV_results.index:
            mask_train = (CV_data.loc[:, ("metaData", "fold")] == fold_id) & (CV_data.loc[:, ("metaData", "train/test") ] == "train")
            X_train = CV_data.loc[ mask_train , ("feature" , slice(None)) ].values.copy()
            y_train = CV_data.loc[ mask_train , ("metaData" , "y") ].values.copy()
            
            mask_test = (CV_data.loc[:, ("metaData", "fold")] == fold_id) & (CV_data.loc[:, ("metaData", "train/test") ] == "test")
            X_test = CV_data.loc[ mask_test , ("feature" , slice(None)) ].values.copy()
            y_test = CV_data.loc[ mask_test , ("metaData" , "y") ].values.copy()
            
            logreg = LogReg_Continuous()
            _ = logreg.fit(X= X_train, y = y_train, lmbda = lmbda, n_categ_levels = n_categ_levels )         
            
            pred, acc = logreg.predict(X= X_test, y = y_test)
            y_pred = (pred > 0.5).astype(float)
            if return_preds:
                CV_data.loc[mask_test, ("metaData", "model_out")] =  pred
            if categ_levels_measure:
                for stat in performanceStat:
                    if stat == "acc":
                        CV_results.loc[fold_id, (stat , lmbda, "marginal")] = acc
                    elif stat == "auROC":
                        fpr, tpr, _ = roc_curve(y_test,  pred ) 
                        CV_results.loc[fold_id, (stat , lmbda, "marginal")] = auc(fpr, tpr)
                    elif stat == "auPRC":
                        precision, recall, _ = precision_recall_curve(y_test, pred)
                        CV_results.loc[fold_id, (stat , lmbda, "marginal")] = auc(recall, precision)
                    elif stat == "auPRC_baseline":
                        CV_results.loc[fold_id, (stat , lmbda, "marginal")] = y_test.astype(bool).sum() / len(y_test)
                    else:
                        raise NotImplementedError()
                categ_level_idxs = [ list(CV_data["feature"].columns).index(lvl) for lvl in categ_levels_measure ]
                for lvl_idx, lvl in zip(categ_level_idxs, categ_levels_measure):
                    mask = np.isclose(X_test[:, lvl_idx] , 1 ) 
                    for stat in performanceStat:
                        if stat == "acc":
                            CV_results.loc[fold_id, (stat , lmbda, lvl )] =  np.isclose(y_pred[mask], y_test[mask ]).sum()/np.sum(mask) 
                        elif stat == "auROC":
                            fpr, tpr, _ = roc_curve( y_test[mask],  pred[mask] ) 
                            CV_results.loc[fold_id, (stat , lmbda, lvl )] = auc(fpr, tpr)
                        elif stat == "auPRC":
                            precision, recall, _ = precision_recall_curve(y_test[mask], pred[mask])
                            CV_results.loc[fold_id, (stat , lmbda, lvl )] = auc(recall, precision)
                        elif stat == "auPRC_baseline":
                            CV_results.loc[fold_id, (stat , lmbda, lvl )] = (y_test[mask]).astype(bool).sum() / len(y_test[mask ])
                        else:
                            raise NotImplementedError()
            else:
                for stat in performanceStat:
                    if stat == "acc":
                        CV_results.loc[fold_id, (stat , lmbda)] = acc
                    elif stat == "auROC":
                        fpr, tpr, _ = roc_curve(y_test,  pred ) 
                        CV_results.loc[fold_id, (stat , lmbda)] = auc(fpr, tpr)
                    elif stat == "auPRC":
                        precision, recall, _ = precision_recall_curve(y_test, pred)
                        CV_results.loc[fold_id, (stat , lmbda)] = auc(recall, precision)
                    elif stat == "auPRC_baseline":
                        CV_results.loc[fold_id, (stat , lmbda)] = y_test.astype(bool).sum() / len(y_test)
                    else:
                        raise NotImplementedError()
        if return_preds:
            preds_dict[lmbda] = CV_data.loc[CV_data[("metaData", "train/test")] == "test",:  ].copy()
    
    if return_preds:
         return CV_results , preds_dict
    else:
        return CV_results 


def plot_CV_perfomStat( plotData, stat_name = "auROC" , aspect =2, col_wrap = 3, groupby_day = True  ,**kwargs  ):  

    plotData.index = plotData.index.set_names(names = "fold")
    plotData = plotData.reset_index( col_level = 1, col_fill = "metaData")
    plotData = plotData.melt(id_vars = [("metaData" , "fold")],
                                    var_name = ["Penalty" , "Day"], value_name = stat_name)
    height = 8.0 / (aspect* col_wrap)
    if groupby_day:
        g= sns.catplot(data = plotData, x= "Penalty", y= stat_name , col = "Day" , kind = "box", 
                         height = height , aspect =aspect ,col_wrap= col_wrap,  **kwargs )
    else:
        g = sns.catplot( data = plotData, x= "Day", y= stat_name , col = "Penalty" , kind = "box", 
                            height = height , aspect =aspect ,col_wrap= col_wrap, **kwargs )
    g.set_xticklabels(rotation = 90) 
    return g


def plot_CV_results(CV_results, ax_height = 3 , figwidth= 8.5, ylabel = "" ):
    
    if CV_results.columns.nlevels == 1: 
        ## assume columns describe lambda values only
        CV_results.columns = pd.MultiIndex.from_tuples([ (ylabel , colname) for colname in CV_results.columns])
    level0_unique = list(set(CV_results.columns.get_level_values(0)))
    nrows = len(level0_unique )
    fig , axes = plt.subplots(nrows = nrows, ncols = 2, figsize=( figwidth, ax_height*nrows))
    for ax_row , level0_val in zip(axes, level0_unique):
        ## Box plot
        ax  = ax_row[0]
        plotData = CV_results.loc[: , (level0_val, slice(None))].melt(var_name = "lambda" , value_name = level0_val, col_level = 1)
        plotData.loc[: , "lambda"] = plotData.loc[:, "lambda"].apply(lambda x: "{:.1e}".format(x))
        ax = sns.boxplot(data = plotData ,x = "lambda" , y = level0_val, notch = True,
                            order = sorted( np.unique(plotData.loc[: , "lambda"]) , key = lambda x: float(x) ), ax = ax) 
        _ = ax.set_xticklabels([float(x.get_text()) for x in ax.get_xticklabels()] ,rotation=90)
        ## Plot Mean acc
        ax  =  ax_row[1]
        y_vals = CV_results.loc[: , (level0_val, slice(None))].mean(axis = 0).values
        ax.plot( np.arange(len(y_vals)),  y_vals, marker = "o")
        ax.set_xticks( np.arange(len(y_vals)))
        ax.set_xticklabels( CV_results.loc[: , (level0_val, slice(None))].columns.get_level_values(1), rotation =90)
        ax.set_xlabel("lambda")
    fig.tight_layout()  
    return fig
    
