# GLHMM_paper

This folder contains the code used to run analyses and generate figures in the paper "The Gaussian-Linear Hidden Markov model: a Python package". doi: 
https://doi.org/10.48550/arXiv.2312.07151
 <br /> <br />

Files in this repository: <br />
- MEG.py : a python analysis script loading and running the GLHMM analyses on a MEG dataset involving human subjects performing a visual-memory task (see paper for details on the analyses and data). The script assumes data are pre-downloaded. It outputs analyses results as .npy files. <br />
- ecog.py : a python analysis script loading and running the GLHMM analyses on an EGog dataset involving monkeys performing a motor task (see paper for details on the analyses and data). The script assumes data are pre-downloaded. It outputs analyses results as .npy files. <br />
- FIG1_ecog_plot.py : a python script loading the results of the analyses on ecog data and plotting them. The script assumes the analysis results are correctly stored. It outputs plots in paper figure 2. <br />
- FIG2_MEG_plot.py : a python script loading the results of the analyses on MEG data and plotting them. The script assumes the analysis results are correctly stored. It outputs plots in paper figure 4. <br />
- prediction_tutorial.ipynb : a jupyter notebook script containing a tutorial on how to use the GLHMM toolbox. It also runs analyses on the HCP dataset (human fMRI, assuming data are pre-downloaded) and plots its results, as in paper figure 6. <br /> <br />


Please note: the datasets analyzed are not provided within this repository. In order to be able to run the analyses, please make sure you have access to the required data. <br /> <br />

Disclaimer: this is not the toolbox repository, neither the official documentation website for the toolbox! <br /> <br />

Find the toolbox at: https://github.com/vidaurre/glhmm/glhmm <br />
Tutorials and examples at: https://github.com/vidaurre/glhmm/tree/main/docs/notebooks <br />
Toolbox documentation at: https://glhmm.readthedocs.io/en/latest/index.html <br /> <br />

If you have questions, issues, or suggestions, feel free to reach us at laurama@cfin.au.dk  or use the Discussions section in the toolbox github repo: https://github.com/vidaurre/glhmm <br /> <br />


