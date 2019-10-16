## This folder contains code for predicting gene DE status in time-course experiments from metagene methylation patterns and the number of day after knock-in induction

### `./fitCV_LRC_models_aggregateDays.ipynb`
- Analysis: 
	- Standardize methylation data for each time-course day
	- Run cross-validation of logistic regression classifiers to predict differential expression status ("DEdown.v.rest" and "DEup.v.rest" tasks ) from gene methylation and 1-hot encoding of time-course day
	- Write statistics of test set performance for each cross-validation fold, write box plots of performance across folds
- Input files
	- "timeCourse.geneDEflow.pkl" - a python dictionary of geneIDs that are DEup, not DE or DEdown on time-course days 1-4 
	- 3B13L-dN_mCpG.bayes_gene.extendRel.pklz (where dN is one of d1, d2,d3,d4) - A nested python dictionary of the form
```python
{"d1 :" { "gene1": { "position_rel_scaled" : position_mgene_1, "est_p" : mCpG_1_1} , 
          "gene2" : { "position_rel_scaled" : position_mgene_2 ,"est_p" : mCpG_2_1},
	 ...
	} ,
{"d2 :" { "gene1": { "position_rel_scaled" : position_mgene_1, "est_p" : mCpG_1_2 } , 
          "gene2" : { "position_rel_scaled" : position_mgene_2 , "est_p" : mCpG_2_2 },  
         ... 
        } , 
...
}
```
where `position_mgene_i` is a numpy.ndarray of the metagene positions of CpG context cytosines for gene i and where `mCpG_i_d` is a numpy.ndarray of the mCpG rates of CpG context cytosines associated with gene i on day d
	- genome feature file 

- Output files 
	- CV_d.plus0_DE.down_v_rest_performStats.tsv, CV_d.plus0_DE.up_v_rest_performStats.tsv - table of performance statistics for different values of the regularization parameter on regression coefficients associated with metagene bins 
	- CV_d.plus0_DE.down_preds.test.tsv.gz, CV_d.plus0_DE.up_preds.test.tsv.gz  -  table of predicted class label probabilies for each gene on each time-course day. Prediction is made using the CV fold that had this gene, day pair in its test set
	- Assorted box plots showing the distribution of different performance statistics across the test sets of different cross-validation folds


### `./fitAll_LRC_aggregateDay.ipynb`

- Analysis:
	- Standardize methylation data for each time-course day (this standardization uses all data not just data in training set)
	- Fit logistic regression classifiers for the two classification tasks ( DEdown.v.rest" and "DEup.v.rest" tasks ) using all data
- Input files
	- same as for `./fitCV_LRC_models_aggregateDays.ipynb`
- Output files
	- fitlAll_LRC_regressionCoeffs_best.tsv - table of regression coefficients methylation in each metagene bin and for each time-course day

### `./GO_enriched_performance.ipynb`

- Analysis
	- plot receiver-operating characteristics (ROCs) describing the test set performance for each classification task on each time-course day
	- Illustrate the ratio of distributions of minimum false positive rates (FPRs) for correct classification of test set genes that belong and do not belong to an enriched gene ontology category
- Input files
	- 3B13L-d\<number\>\_DEgenes\_\<type\>-chartAnnot.tsv where \<number\> $\in {1,2,3,4}$ and \<type\> is either up or down— lists GO terms enriched for DE up and for DE down genes on each time-course day and gene names associated with each term
	- CV_d.plus0_DE.down_preds.test.tsv.gz, CV_d.plus0_DE.up_preds.test.tsv.gz
	- geneNames.proper.tsv— table mapping between gene names and geneIDs
- Output files
	- CV.downVrest_ROC_GO.FPR.enrichment.pdf , CV.upVrest_ROC_GO.FPR.enrichment.pdf 
	- CV.downVrest_ROC_GO.FPR.enrichment.stats.tsv, CV.upVrest_ROC_GO.FPR.enrichment.stats.tsv


