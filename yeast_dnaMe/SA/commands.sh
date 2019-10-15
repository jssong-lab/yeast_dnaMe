#!/bin/bash
set -euo pipefail
mkdir -p "./oe"
workDir=$(pwd)

conditions=(3A1 \
3A13L \
3A2 \
3A23L \
3B1 \
3B13L-d1)

###############################################################
## Minimizing

n_iter=8000000
sampleInterval=2
n_save=20000
n_updates=2
saveInterval=800000
init_per_condition=25

#############
## First 25
for ((c_idx=4; c_idx<6; c_idx++ ))
        do
        cond=${conditions[$c_idx]}
	d=$( cat ./d_byCondition.minimize.txt | grep -E "${cond}[[:space:]]" | cut -f2  )
	d_round=$(printf "%.3f" $d )
	echo $cond $d $d_round        
	inOut_idxs=${cond}_idxs_lowest${init_per_condition}.txt
        p=$( ls  ../predictMethyl/CNN_param/$cond/*.hdf5 ) 
        tail -n+2 ./initializations/${cond}_lowest50.tsv | cut -f1 | head -n $init_per_condition  > $inOut_idxs
        fo=${cond}_min_d.${d_round}_samples.set1_iter.8M
        name=SA_${cond}_min_d.${d_round}_iter.8M
        sbatch -p gpusong --mem 80G --gres gpu:1 --nodelist=compute-3-2 -D $workDir --job-name=$name -o "./oe/$name.o" -e "./oe/$name.e" <<EOF
#!/bin/bash
source ~/lib/loadModules_TensorFlow.bash
python ../codes/SA.py  --inOut_idxs $inOut_idxs -p $p --n_iter $n_iter --n_save $n_save -d $d --n_updates $n_updates --sampleInterval $sampleInterval  --fo $fo --saveInterval $saveInterval --minimize
EOF
done

###################
## Second 25
n_iter=8000000
sampleInterval=2
n_save=20000
n_updates=2
saveInterval=800000
init_per_condition=25

for ((c_idx=5; c_idx<6; c_idx++ ))
        do  
        cond=${conditions[$c_idx]}
        d=$( cat ./d_byCondition.minimize.txt | grep -E "${cond}[[:space:]]" | cut -f2  )
        d_round=$(printf "%.3f" $d )
        echo $cond $d $d_round        
        inOut_idxs=${cond}_idxs_lowest${init_per_condition}.txt
        p=$( ls ../predictMethyl/CNN_param/$cond/*.hdf5 ) 
        tail -n+2 ./initializations/${cond}_lowest50.tsv | cut -f1 | tail -n $init_per_condition  > $inOut_idxs
        fo=${cond}_min_d.${d_round}_samples.set2_iter.8M
        name=SA_${cond}_min_d.${d_round}_set2_iter.8M
        sbatch -p gpusong --mem 80G --gres gpu:1 --nodelist=compute-3-2 -D $workDir --job-name=$name -o "./oe/$name.o" -e "./oe/$name.e" <<EOF
#!/bin/bash
source ~/lib/loadModules_TensorFlow.bash
python ../codes/SA.py  --inOut_idxs $inOut_idxs -p $p --n_iter $n_iter --n_save $n_save -d $d --n_updates $n_updates --sampleInterval $sampleInterval  --fo $fo --saveInterval $saveInterval --minimize
EOF
done


##################################################################
## Maximizing
n_iter=8000000
sampleInterval=2
n_save=20000
n_updates=2
saveInterval=800000
init_per_condition=25

#############
## First 25
for ((c_idx=5; c_idx<6; c_idx++ ))
        do
        cond=${conditions[$c_idx]}
        d=$( cat ./d_byCondition.maximize.txt | grep -E "${cond}[[:space:]]" | cut -f2  )
        d_round=$(printf "%.3f" $d )
        echo $cond $d $d_round        
        inOut_idxs=${cond}_idxs_highest${init_per_condition}.txt
        p=$( ls ../predictMethyl/CNN_param/$cond/*.hdf5)  
        tail -n+2 ./initializations/${cond}_highest50.tsv | cut -f1 | head -n $init_per_condition  > $inOut_idxs
        fo=${cond}_max_d.${d_round}_samples.set1_iter.8M
        name=SA_${cond}_max_d.${d_round}_set1_iter.8M
        sbatch -p gpusong --mem 80G --gres gpu:1 --nodelist=compute-3-2 -D $workDir --job-name=$name -o "./oe/$name.o" -e "./oe/$name.e" <<EOF
#!/bin/bash
source ~/lib/loadModules_TensorFlow.bash
python ../codes/SA.py  --inOut_idxs $inOut_idxs -p $p --n_iter $n_iter --n_save $n_save -d $d --n_updates $n_updates --sampleInterval $sampleInterval  --fo $fo --saveInterval $saveInterval 
EOF
done


###################
## Second 25
n_iter=8000000
sampleInterval=2
n_save=20000
n_updates=2
saveInterval=800000
init_per_condition=25

for ((c_idx=3; c_idx<6; c_idx++ ))
        do  
        cond=${conditions[$c_idx]}
        d=$( cat ./d_byCondition.maximize.txt | grep -E "${cond}[[:space:]]" | cut -f2  )
        d_round=$(printf "%.3f" $d )
        echo $cond $d $d_round        
        inOut_idxs=${cond}_idxs_highest${init_per_condition}.txt
        p=$( ls ../predictMethyl/CNN_param/$cond/*.hdf5 ) 
        tail -n+2 ./initializations/${cond}_highest50.tsv | cut -f1 | tail -n $init_per_condition  > $inOut_idxs
        fo=${cond}_max_d.${d_round}_samples.set2_iter.8M
        name=SA_${cond}_max_d.${d_round}_set2_iter.8M
        sbatch -p gpusong --mem 80G --gres gpu:1 --nodelist=compute-3-2 -D $workDir --job-name=$name -o "./oe/$name.o" -e "./oe/$name.e" <<EOF
#!/bin/bash
source ~/lib/loadModules_TensorFlow.bash
python ../codes/SA.py  --inOut_idxs $inOut_idxs -p $p --n_iter $n_iter --n_save $n_save -d $d --n_updates $n_updates --sampleInterval $sampleInterval  --fo $fo --saveInterval $saveInterval
EOF
done



