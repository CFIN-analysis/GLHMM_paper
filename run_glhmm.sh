#!/bin/bash

#This script is used in the context of the GLHMM paper, doi:
#https://doi.org/10.48550/arXiv.2312.07151
#This script runs and measures the computational performance of stochastic and non-stochastic GLHMM training on 
#data from the HCP and MEGUK datasets. The datasets are described in Van Essen et al.(2013) and Hunt et al.(2016), 
#and publicly available.
#See papers: https://pubmed.ncbi.nlm.nih.gov/23684880/ and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5127325/
#
#!!!Note!!!
#This script calls on stoc_v_NonStoc.py. Please make sure the path for it is set correctly. It outputs *_report.txt,
#containing the information about computational demands. The data generated from this script are plotted by the
#plotting.py and plotting_fr.py scripts. This script has high computational demands!
#
#Requirements:
#- perf
#- tsp
#
#Author: Lenno Ruijters
#email: au710840@uni.au.dk
#Date: 13/12/2023


### Run the GLHMM training stochastically and non-stochastically and measure the floating point operations (FLOPs) required to complete it.
### This script uses task spooler (tsp) to queue and parallelise training sessions.

#activate conda environment
source activate base
conda activate hmm

#open task spooler pool
tsp -C -S 8

#set data modality
mod=MRC

#run training script Stoc_v_NonStoc.py through <perf> to retreive FLOPs: -e fp_ret_sse_avx_ops.all
#define batch size (0=non-stochastic)
for batch_size in 0 1 2 5 10 20 40 50 55
do
	#define forgetting rate
	fr=0.8
	#label runs
	for run in {1..10}
	do		
		echo "Run ${run} of HMM with batch-size ${batch_size} on ${mod} data"
		tsp -L ${batch_size}_${run} \
			perf stat -o <data_path>/${mod}/n${batch_size}_FR${fr}_run${run}_report.txt \
				--append -e fp_ret_sse_avx_ops.all \
				-e cpu-clock \
				python <data_path>/Stoc_v_NonStoc.py ${mod} ${batch_size} ${run} ${fr} &
	done
done

#run training script Stoc_v_NonStoc.py through <perf> to retreive FLOPs: -e fp_ret_sse_avx_ops.all
#define forgetting rate
for fr in 0.5 0.6 0.7 0.9
do
	#define batch size
	batch_size=10
	#label runs
	for run in {1..10}
	do		
		echo "Run ${run} of HMM with batch-size ${batch_size} on ${mod} data"
		tsp -L ${batch_size}_${run} \
			perf stat -o <data_path>/${mod}/n${batch_size}_FR${fr}_run${run}_report.txt \
				--append -e fp_ret_sse_avx_ops.all \
				-e cpu-clock \
				python <data_path>/Stoc_v_NonStoc.py ${mod} ${batch_size} ${run} ${fr} &
	done
done



#set data modality
mod=HCP

#run training script Stoc_v_NonStoc.py through <perf> to retreive FLOPs: -e fp_ret_sse_avx_ops.all
#define batch size (0=non-stochastic)
for batch_size in 0 5 10 20 100 200 500 800 1003
do
	#define forgetting rate
	fr=0.8
	#label runs
	for run in {1..10}
	do
		echo "Run ${run} of HMM with batch-size ${batch_size} on ${mod} data"
		tsp -L ${batch_size}_${run} \
			perf stat -o <data_path>/${mod}/n${batch_size}_FR${fr}_run${run}_report.txt \
			--append -e fp_ret_sse_avx_ops.all \
			-e cpu-clock \
			python <data_path>/Stoc_v_NonStoc.py ${mod} ${batch_size} ${run} ${fr} &
	done
done

#run training script Stoc_v_NonStoc.py through <perf> to retreive FLOPs: -e fp_ret_sse_avx_ops.all
#define forgetting rate
for fr in 0.5 0.6 0.7 0.9
do
	#define batch size
	batch_size=200
	#label runs
	for run in {1..10}
	do
		echo "Run ${run} of HMM with batch-size ${batch_size} on ${mod} data"
		tsp -L ${batch_size}_${run} \
			perf stat -o <data_path>/${mod}/n${batch_size}_FR${fr}_run${run}_report.txt \
			--append -e fp_ret_sse_avx_ops.all \
			-e cpu-clock \
			python <data_path>/Stoc_v_NonStoc.py ${mod} ${batch_size} ${run} ${fr} &
	done
done



