"""
This script is used in the context of the GLHMM paper, doi:https://doi.org/10.48550/arXiv.2312.07151
The script runs the GL HMM analysis and behavioral prediction on an ECoG dataset of monkeys performing a motor task. See the paper for details on the analyses.

The dataset is described in Chao et al. (2010) and publicly available. See paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2856632/

The script saves the analyses performed on each subject individually as .npy files. 


Requirements: 
- GLHMM
The script assumes you have already correctly installed version 0.1.13 of the GLHMM toolbox. 
You can download the toolbox using "pip install --user git+https://github.com/vidaurre/glhmm".
See documentation at: https://glhmm.readthedocs.io/en/latest/index.html 

Other requirements to run this script:
- pandas
- numpy

!!!Note!!!
The script loads the ECoG data, supposed to be stored in a sub-folder called 'ECoG_data', and saves the analysis results in a sub-folder called 'out', assuming they are both located in the same folder as this script.
Before running the script, make sure the data are available to you. Change the data paths in the variables 'datadir' and 'outdir' if needed.


Author: Diego Vidaurre
email: dvidaurre@cfin.au.dk
Date: 12/12/2023
"""

# imports
import pandas as pd
import numpy as np

from glhmm import preproc
from glhmm import glhmm
from glhmm import graphics
from glhmm import utils


# --------- SET PATHS and PARAMETERS specific to the analysis --------------
outdir = '/out/'
datadir = '/ECoG_data/'

to_do = 1

impose_cyclic_structure = False
KK = (2,4,6,8,10,12,14,16,18,20,22,24)
dd = (10,50,100,500,1000,2000,4000,8000,15000)
if impose_cyclic_structure: str_P = '_cyclic'
else: str_P = ''

    
# ---------------------------- LOAD DATA ----------------------------    
if to_do==1:
    x1 = pd.read_csv(datadir + 'x1.txt', header = None)
    y1 = pd.read_csv(datadir + 'y1.txt', header = None)
    x,y = x1,y1
elif to_do==2:
    x2 = pd.read_csv(datadir + 'x2.txt', header = None)
    y2 = pd.read_csv(datadir + 'y2.txt', header = None)
    x,y = x2,y2
else:
    x3 = pd.read_csv(datadir + 'x3.txt', header = None)
    y3 = pd.read_csv(datadir + 'y3.txt', header = None)
    x,y = x3,y3

x = x.to_numpy()
y = y.to_numpy()
ind = np.zeros((1,2)).astype(int)
ind[0,0] = 0
ind[0,1] = x.shape[0]

_ind = np.copy(ind)
x,ind =  preproc.preprocess_data(x,ind,
    fs = 1000,
    pca = 10,
    downsample = 250)

y,_ =  preproc.preprocess_data(y,_ind,
    fs = 1000,
    downsample = 250)

options = {'cyc':1000,'initrep':1}



# --------------------------- RUN HMM and save analysis results ----------------------- 
if False: # grid run

    r2 = np.zeros((len(KK),len(dd),y.shape[1]))
    fe = np.zeros((len(KK),len(dd)))
    fo = np.zeros((len(KK),len(dd),KK[-1]))

    for ik in range(len(KK)):
        for id in range(len(dd)):

            K = KK[ik]
            d = dd[id]
        
            if impose_cyclic_structure:
                Pstructure = np.eye(K, dtype=bool)
                for k in range(K-1): Pstructure[k,k+1] = True
                Pistructure = np.zeros(K, dtype=bool)
                Pistructure[0] = True
            else:
                Pstructure = np.ones((K,K), dtype=bool)
                Pistructure = np.ones(K, dtype=bool)

            hmm = glhmm.glhmm(K=K,
                covtype='shareddiag',
                model_mean='no',
                model_beta='state', 
                Pistructure = Pistructure, Pstructure = Pstructure,
                dirichlet_diag=d)
            
            Gamma,Xi,fe_ = hmm.train(x,y,ind,options=options)
            fe[ik,id] = fe_[-1]
            r2[ik,id,:] = hmm.get_r2(x,y,Gamma,ind)
            fo[ik,id,0:K] = utils.get_FO(Gamma,ind)

            np.save(outdir + 'fe' + str(to_do) + str_P + '.npy', fe)
            np.save(outdir + 'r2' + str(to_do) + str_P + '.npy', r2)
            np.save(outdir + 'fo' + str(to_do) + str_P + '.npy', fo)

            print(str(ik) + ', ' + str(id))

            # vpath = hmm.decode(x,y,ind,viterbi=True)

if True: # focus on one 

    R = 20
    K = 4

    r2 = np.zeros((len(dd),y.shape[1],R))
    fe = np.zeros((len(dd),R))
    fo = np.zeros((len(dd),K,R))

    for ir in range(R):
        for id in range(len(dd)):

            #K = KK[ik]
            d = dd[id]
        
            if impose_cyclic_structure:
                Pstructure = np.eye(K, dtype=bool)
                for k in range(K-1): Pstructure[k,k+1] = True
                Pistructure = np.zeros(K, dtype=bool)
                Pistructure[0] = True
            else:
                Pstructure = np.ones((K,K), dtype=bool)
                Pistructure = np.ones(K, dtype=bool)

            hmm = glhmm.glhmm(K=K,
                covtype='shareddiag',
                model_mean='no',
                model_beta='state', 
                Pistructure = Pistructure, Pstructure = Pstructure,
                dirichlet_diag=d)
            
            Gamma,Xi,fe_ = hmm.train(x,y,ind,options=options)
            #fe[id,ir] = fe_[-1]
            r2[id,:,ir] = hmm.get_r2(x,y,Gamma,ind)
            fo[id,0:K,ir] = utils.get_FO(Gamma,ind)
            fe[id,ir] = np.sum(hmm.get_fe(x,y,Gamma,Xi,indices=ind,non_informative_prior_P=True))

            np.save(outdir + 'DD_fe' + str(to_do) + str_P + '.npy', fe)
            np.save(outdir + 'DD_r2' + str(to_do) + str_P + '.npy', r2)
            np.save(outdir + 'DD_fo' + str(to_do) + str_P + '.npy', fo)

            print(str(ir) + ', ' + str(id))

            # vpath = hmm.decode(x,y,ind,viterbi=True)

if True:

    R = 20
    d = 100

    r2 = np.zeros((len(KK),y.shape[1],R))
    fe = np.zeros((len(KK),R))
    fo = np.zeros((len(KK),np.max(KK),R))

    for ir in range(R):
        for ik in range(len(KK)):

            K = KK[ik]
        
            if impose_cyclic_structure:
                Pstructure = np.eye(K, dtype=bool)
                for k in range(K-1): Pstructure[k,k+1] = True
                Pistructure = np.zeros(K, dtype=bool)
                Pistructure[0] = True
            else:
                Pstructure = np.ones((K,K), dtype=bool)
                Pistructure = np.ones(K, dtype=bool)

            hmm = glhmm.glhmm(K=K,
                covtype='shareddiag',
                model_mean='no',
                model_beta='state', 
                Pistructure = Pistructure, Pstructure = Pstructure,
                dirichlet_diag=d)
            
            Gamma,Xi,fe_ = hmm.train(x,y,ind,options=options)
            #fe[ik,ir] = fe_[-1]
            r2[ik,:,ir] = hmm.get_r2(x,y,Gamma,ind)
            fo[ik,0:Gamma.shape[1],ir] = utils.get_FO(Gamma,ind)
            fe[ik,ir] = np.sum(hmm.get_fe(x,y,Gamma,Xi,indices=ind,non_informative_prior_P=True))

            np.save(outdir + 'KK_fe' + str(to_do) + str_P + '.npy', fe)
            np.save(outdir + 'KK_r2' + str(to_do) + str_P + '.npy', r2)
            np.save(outdir + 'KK_fo' + str(to_do) + str_P + '.npy', fo)

            print(str(ir) + ', ' + str(ik))

            # vpath = hmm.decode(x,y,ind,viterbi=True)



a = 1

