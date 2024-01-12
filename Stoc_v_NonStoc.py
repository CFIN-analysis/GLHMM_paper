"""
This script is used in the context of the GLHMM paper, doi:
https://doi.org/10.48550/arXiv.2312.07151
This script concerns GLHMM performance during training on data from the HCP and MEGUK datasets. 
The datasets are described in Van Essen et al.(2013) and Hunt et al.(2016), and publicly available.
See papers: https://pubmed.ncbi.nlm.nih.gov/23684880/ and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5127325/

!!!Note!!!
This script is designed to be called by run_glhmm.sh, and should not be run by itself. It depends on 
data being present in the defined in_dir path, and outputs both Free Energy and Stdout to the out_dir
in *_output.txt and *_report.txt respectively. The data generated from this script are plotted by the
plotting.py and plotting_fr.py scripts.

Requirements:
- GLHMM
The script assumes you have already correctly installed version 0.1.13 of the GLHMM toolbox.
You can download the toolbox using "pip install --user git+https://github.com/vidaurre/glhmm".
See documentation at: https://glhmm.readthedocs.io/en/latest/index.html

Other requirements to run this script:
- numpy

Author: Lenno Ruijters
email: au710840@uni.au.dk
Date: 13/12/2023
"""



### Import dependancies
import numpy
import glhmm.glhmm
import os
import time
import sys

### Retreive command arguments
dataset     =   sys.argv[1]
n_batch     =   int(sys.argv[2])
n_run       =   int(sys.argv[3])
fr          =   float(sys.argv[4])

### Define path and file names
in_dir      =   "<data path>" + dataset

### .mat file input requires Y=data; X=predictors
in_files    =   [in_dir+"/"+f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))] 
in_files = [file for file in in_files if file.startswith(in_dir+"/subj")]


### Output directory and file
out_dir     =   '<data path>' + dataset
out_file    =   '_output.txt'

### Save stdout to a file
sys.stdout  =   open(out_dir + '/n'+ str(n_batch) + '_FR' + str(fr) + '_run' + str(n_run) + '_report.txt', 'wt')

### Open output files
FE          =   open(out_dir + '/FE' + str(n_batch) + '_FR' + str(fr) + '_run' + str(n_run) + out_file, 'wb')


### Create HMM class
### Model_beta="no" as X=None in our training data
HMM = glhmm.glhmm.glhmm(model_beta="no")


if n_batch==0:
    ### Train HMM non-stochastically.
    ### Init cycles set to 25 for more direct comparison to the stochastic models.
    ### Max cycles set to 3000 to encourage convergence.
    Options={"initcyc":25,
        "cyc":3000}

    print("Running non-stochastic training. \n")
    t = time.time()

    ### Retreive free energy
    E_out  =   HMM.train(X=None,Y=None,files=in_files,options=Options)[2]

    ### Save to output file
    numpy.savetxt(FE,E_out)
    FE.close()

    elapsed = time.time() - t
    print("Elapsed time non-stochastic: ", elapsed, "\n.")
    
else:
    ### Train HMM stochastically over different batch sizes
        Options={"stochastic":True,
            "Nbatch":n_batch,
            "forget_rate":fr,
            "cyc":3000}

        print("Running stochastic training with batch size ", n_batch, "\n")
        t = time.time()

        ### Train HMM and Retreive free energy (NB: one data-set-> X=None,Y=data)
        E_out  =   HMM.train(X=None,Y=None,files=in_files,options=Options)[2]

        ### Save to output file
        numpy.savetxt(FE,E_out)
        FE.close()

        elapsed = time.time() - t
        print("Elapsed time stochastic training with batch size = ", n_batch, ": ",  elapsed, "\n.")