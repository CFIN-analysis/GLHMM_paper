"""
This script is used in the context of the GLHMM paper, doi:https://doi.org/10.48550/arXiv.2312.07151
The script runs the GL HMM analysis and behavioral prediction on a MEG dataset regarding human subjects performing a visual -memory task. See the paper for details on the analyses.

The dataset is described in Myers et al. (2015), and publicly available. See paper: https://elifesciences.org/articles/09000

The script saves the analyses performed on each subject individually as .npy files. 


Requirements: 
- GLHMM
The script assumes you have already correctly installed version 0.1.13 of the GLHMM toolbox. 
You can download the toolbox using "pip install --user git+https://github.com/vidaurre/glhmm".
See documentation at: https://glhmm.readthedocs.io/en/latest/index.html 

Other requirements to run this script:
- pandas
- matplotlib
- numpy
- scipy
- math

!!!Note!!!
The script loads the MEG data, supposed to be stored in a sub-folder called 'MEG_data', and saves the analysis results in a sub-folder called 'out', assuming they are both located in the same folder as this script.
Before running the script, make sure the data are available to you. Change the data paths in the variables 'datadir' and 'outdir' if needed.


Author: Diego Vidaurre
email: dvidaurre@cfin.au.dk
Date: 12/12/2023
"""


# imports
import pandas as pd
import numpy as np
import scipy
import math

from glhmm import preproc
from glhmm import glhmm
from glhmm import utils
from glhmm import auxiliary
from glhmm import io


# --------- SET PATHS and PARAMETERS specific to the analysis --------------
outdir = '/out/'
datadir = '/MEG_data/'


isession = 0 # MEG
K = 10
R = 1

beh_all = scipy.io.loadmat(datadir + 'behavior.mat')
gt_all = scipy.io.loadmat(datadir + 'good_trials_all.mat')

options = {'cyc':1,'initrep':0}


# --------- RUN analyses for each subject --------------
for isub in range(1,11):

    # good trials
    gt = np.squeeze(gt_all['good_trials'][isub-1,0]) == 1
    N0 = len(gt)
    # response 
    resp = beh_all['behavior'][isub-1,0][0][gt]
    rt = beh_all['behavior'][isub-1,0][1][gt]
    there_is_response = np.squeeze(resp)==1
    rt = rt[there_is_response]
    N = len(rt)

    if isub == 10: substr = str(isub)
    else: substr = '0' + str(isub)

    if isession==0: 
        fname = datadir + 'S' + substr + '_MEG.mat'
    else: 
        fname = datadir + 'S' + substr + '_EEG.mat'
    fnameY = datadir + 'S' + substr + '_MEG_response02.mat' # relang

    dat = scipy.io.loadmat(fname)
    X = dat["X"]
    T = dat["T"]
    ind_ = auxiliary.make_indices_from_T(T)

    X,ind =  preproc.preprocess_data(X,ind_,
        fs = 250,
        pca = 48,
        downsample = 100)
    
    Y = scipy.io.loadmat(fnameY)
    Y = Y["Y"]
    Y = np.reshape(Y,(T[0][0],round(Y.shape[0]/T[0][0]),Y.shape[1]),order='F')
    Yshort = Y[0,:,:]
    Y_ds = np.zeros((X.shape[0],Y.shape[2]))
    for j in range(ind.shape[0]):
        Y_ds[ind[j,0]:ind[j,1],:] = Yshort[j,:]
    Y = Y_ds
    Y_noise = Y + np.random.normal(0,0.01,Y.shape)

    ttrial = ind[0,1]-ind[0,0]
    Gamma0 = np.zeros((ttrial,ind.shape[0],K))
    ttrial_K = math.floor(ttrial/K)
    for k in range(K-1):
        Gamma0[(k*ttrial_K):((k+1)*ttrial_K),:,k] = 1
    Gamma0[((K-1)*ttrial_K):,:,K-1] = 1
    Gamma0 = np.reshape(Gamma0,((ttrial*ind.shape[0],K)),order='F')
      
    FE = np.zeros((R,3))
    SER = np.zeros((ttrial,K,R,3))
    REG = np.zeros((ttrial,2,R,3))

    for r in range(R):

        for j in range(3):

            if j==0: # unstructured
                Pstructure = np.ones((K,K), dtype=bool)
                Pistructure = np.ones(K, dtype=bool)
                options['cyc'] = 10
            elif j==1: # chain
                Pstructure = np.eye(K, dtype=bool)
                for k in range(K-1): Pstructure[k,k+1] = True
                Pistructure = np.zeros(K, dtype=bool)
                Pistructure[0] = True
                options['cyc'] = 1
            elif j==2: # circular
                Pstructure = np.eye(K, dtype=bool)
                for k in range(round(K/2)-1): Pstructure[k,k+1] = True
                Pstructure[round(K/2):,round(K/2):] = True
                Pistructure = np.ones(K, dtype=bool)
                options['cyc'] = 1
            # else: # chain and then unstructured
            #     Pstructure = np.eye(K, dtype=bool)
            #     for k in range(K-1): Pstructure[k,k+1] = True
            #     Pstructure[K-1,0] = True
            #     Pistructure = np.zeros(K, dtype=bool)
            #     Pistructure[0] = True

            hmm = glhmm.glhmm(K=K,
                covtype='shareddiag',
                model_mean='no',
                model_beta='state', 
                Pistructure = Pistructure, Pstructure = Pstructure,
                dirichlet_diag=100)
                
            
            Gamma,Xi,fe_ = hmm.train(X,Y_noise,ind,options=options,Gamma = Gamma0)

            FE[r,j] = fe_[-1]
            SER[:,:,r,j] = utils.get_state_evoked_response(Gamma,ind)

            print(str(r) + ', ' + str(j))

            io.save_hmm(hmm, outdir + 'MEG_HMM_S' + str(isub) + '-' + str(r) + '-' + str(j) + '.npy')    
            np.save(outdir + 'MEG_Gamma_S' + str(isub) + '-' + str(r) + '-' + str(j) + '.npy', Gamma)

            (ttrial,K) = (round(Gamma.shape[0]/N0),Gamma.shape[1])
            Gamma = np.reshape(Gamma,(ttrial,N0,Gamma.shape[1]), order='F' )
            Gamma = Gamma[:,gt,:]

            for t in range(ttrial):
                G = np.copy(Gamma[t,:,:])
                G -= np.mean(G,axis=0)
                y = np.copy(resp)
                y = (y - np.mean(y))
                b = np.linalg.inv(G.T @ G + 0.01 * np.eye(K)) @ (G.T @ y)
                REG[t,0,r,j] = np.sqrt(np.sum((G @ b - y) ** 2))
                G = Gamma[t,there_is_response,:]
                G -= np.mean(G,axis=0)
                y = np.copy(rt)
                y = (y - np.mean(y))
                b = np.linalg.inv(G.T @ G + 0.01 * np.eye(K)) @ (G.T @ y)
                REG[t,1,r,j] = np.sqrt(np.sum((G @ b - y) ** 2))  
                
    # save analyses results
    np.save(outdir + 'MEG_reg_S' + str(isub) + '.npy', REG)
    np.save(outdir + 'MEG_fe_S' + str(isub) + '.npy', FE)
    np.save(outdir + 'MEG_ser_S' + str(isub) + '.npy', SER)


