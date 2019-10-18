# Codes associated with the paper "Epigenetic engineering of yeast reveals dynamic molecular adaptation to methylation stress and genetic modulators of specific DNMT3 family members"

## Repository Contents
`codes`

- `SA.py`script for performing Simulated Annealing (SA), 
- `mCpG_to_feature.py `script for averaging CpG methylation rates across features
- `./LRC`Logisitic regerssion classifier code

`predictDE`

- `fitCV_LRC_models_aggregateDays.ipynb` - run cross validation for to evaluate "DE.down vs rest" and "DE.up vs rest" classifier  at choose best regulatization value
- `fitAll_LRC_aggregateDay.ipynb` - get regression coefficients for fit using all data for best regularization value
- `GO_enriched_performance.ipynb` - Draw ROC curves and get distribution of minimum false positive rates fopr putative endogenously regulated (PER) genes and non-PER genes

`predictMethyl`

	- `./trainingCNN.ipynb` - code to fit and evaluate convolutional neural networks to predict CpG methylation rates for 6 DNMT knock-in conditions 
	- `CNN_params.tar.gz` - parameters for final CNN models for each of the 6 conditions

`SA`

- `./commands.sh` - demonstrate calling SA code (we do not include all sampled seqeunces due to large file sizes)
- `d_byConditions.maximize.txt` , `d_byCondition.minimize.txt` - lists of d parameters (related to ininitial temperature) used for SA maximization in minimization for each condition

`cellResponse`

 - `geneNames.proper.txt` list mapping ID values of genes annotated in xx genome feature file to Saccharomyces Genome Database (SGD) IDs. Only genes with SGD IDs were considered for gene ontology analysis.

   

