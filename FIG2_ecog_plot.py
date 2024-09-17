"""
This script is used in the context of the GLHMM paper, doi:
https://doi.org/10.48550/arXiv.2312.07151
This script concerns GL HMM analyses and behavioral predictions run on an ECoG dataset of monkeys performing a motor task. See the paper for details on the analyses.

The dataset is described in Chao et al. (2010) and publicly available. See paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2856632/

!!!Note!!!
The script runs some preprocessing and GL HMM analyses on the ECoG dataset. It also loads the analyses outputed by the script "ecog.py" (in the same folder), so it assumes you already have downloaded the data, run the analyses and correctly stored the results in .npy files in the sub-folder called 'out' (located in the same folder as this script).
Hence, before running this script, make sure the data are available to you, run the "ecog.py" file and make sure the results are correctly saved. Change the data path in the variable 'outdir' if needed.

The script outputs figure 2 of the paper.


Requirements: 
- GLHMM
The script assumes you have already correctly installed version 0.1.13 of the GLHMM toolbox. 
You can download the toolbox using "pip install --user git+https://github.com/vidaurre/glhmm".
See documentation at: https://glhmm.readthedocs.io/en/latest/index.html 

Other requirements to run this script:
- pandas
- matplotlib
- numpy


Author: Sonsoles Alonso Martinez
email: sonsoles.alonso@cfin.au.dk
Date: 12/12/2023

"""
 
    
# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from glhmm import glhmm
from glhmm import graphics
from glhmm import preproc


# --------- SET PATHS and PARAMETERS specific to the analysis --------------

# data directory
outdir = '/out/'

# one subject at a time: specify which subject 
monkey = 1

#plotting fontsize
fontsize = 8

"""
# ---------------------- LOAD DATA -----------------------
x1 = pd.read_csv(outdir +'ECoG_NeuralComputation_PLS/x1.txt', header = None)
y1 = pd.read_csv(outdir + 'ECoG_NeuralComputation_PLS/y1.txt', header = None)

x,y = x1,y1
x = x.to_numpy()
y = y.to_numpy()
ind_ = np.zeros((1,2)).astype(int)
ind_[0,0] = 0
ind_[0,1] = x.shape[0]


# -------------------- PREPROCESS DATA -----------------------

# downsampling, from 1000 Hz to 250 Hz
xf,ind =  preproc.preprocess_data(x,ind_,
    fs = 1000,
    downsample = 250)

# run pca with 10 components
xpca,_ =  preproc.preprocess_data(x,ind_,
    fs = 1000,
    pca = 10,
    downsample = 250)    

# downsample y
y,_ =  preproc.preprocess_data(y,ind_,
    fs = 1000,
    downsample = 250)


# -------------------- HMM PARAMETERS and TRAINING ----------------------

# set number of states K 
K = 4

# initialize HMM with analysis parameters and type
hmm = glhmm.glhmm(K=K,
    covtype='shareddiag',
    model_mean='no',
    model_beta='state', #Pistructure = Pistructure, Pstructure = Pstructure,
    dirichlet_diag=100)

# other options about initialization
options = {'cyc':100,'initrep':0}

# train the HMM
Gamma,Xi,fe = hmm.train(xpca,y,ind,options=options)

# Save Gamma and hmm for reproducibiity
np.save(outdir + 'Gamma.npy', Gamma)
with open(outdir + 'hmm.pkl', 'wb') as f:
    pickle.dump(hmm, f)

"""

# --------------------------- PLOT A -------------------------------

# Load Gamma and hmm
Gamma = np.load(outdir + 'Gamma.npy')
with open(outdir + 'hmm.pkl', 'rb') as f:
    hmm = pickle.load(f)
 
# graphics.show_beta(hmm,only_active_states=True,X=xpca,Y=y,Gamma=Gamma)
graphics.show_beta(hmm)

# Graphics show_beta code here
fig = plt.gcf()
fig.set_size_inches(3.5,3.5)  # Adjust the figure size if needed
fig.set_dpi(300)  # Adjust the dpi if needed
for i, ax in enumerate(fig.get_axes()):
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color('#333333')
    if i % 4 >= i // 4:  # Remove both diagonal and upper diagonal plots
        ax.set_visible(False)

    if i % 4 == 0:  # Add Y label only for the subplots in the first column
        ax.set_ylabel(f'State$_{{{i // 4 + 1}}}$ β', fontsize=fontsize)
    if i // 4 == 3:  # Add X label only for the subplot in the last row
        ax.set_xlabel(f'State$_{{{i % 4 + 1}}}$ β', fontsize=fontsize)

    for item in [ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(fontsize)  # Adjust the fontsize as needed
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fontsize - 2)  # Adjust the fontsize as needed

    ax.grid(False)  # Adjust linewidth, linestyle, and color

    for ax in fig.get_axes():     
        # Access the scatter plot attributes and modify them
        scatter = ax.collections[0]
        scatter.set_sizes([7])  # Set smaller marker size
        scatter.set_alpha(0.5)
        scatter.set_edgecolor('none')
plt.show()



# ---------------------------- PLOT B -------------------------------
# Calculate the average of FO variable
fig = plt.gcf()
fig.set_size_inches(1.5, 1.7)  # Adjust the figure size if needed
fig.set_dpi(300)  # Adjust the dpi if needed
average_fo = np.sort(Gamma.mean(axis=0))[::-1]
# Create a bar plot with the sorted indices
#plt.bar(range(4), average_fo, color=col(range(4)))
plt.bar(range(4), average_fo, color='gray', alpha=0.7)
xtick_labels = [f'State {i}' for i in range(1,5)]
plt.xticks(range(4), xtick_labels, rotation='vertical')  # Rotate labels
plt.ylabel('Fractional occupancy', fontsize=fontsize)
plt.tick_params(axis='both', labelsize=10)
# Adjust grid line width
ax = fig.gca()
ax.grid(False)
for spine in ax.spines.values():
    spine.set_linewidth(0.4)
    spine.set_color('#333333') 
for item in ax.get_yticklabels():
        item.set_fontsize(fontsize - 2)  # Adjust the fontsize as needed
plt.show()



# -------------------------- PLOT C ---------------------------------

# load other results outputed from analysis script "ecog.py"
FO1 = np.load(outdir + 'KK_fo1.npy')
KK_array = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)
(KK, K, R) = FO1.shape
Entropy1 = np.zeros((KK, R, 3))
maxFO1 = np.zeros((KK, R, 3))
R2_1 = np.zeros((KK, R, 3))
FE1 = np.zeros((KK, R, 3))

for m in range(3):
    FOj = np.load(outdir + 'KK_fo' + str(m+1) + '.npy')
    R2j = np.load(outdir + 'KK_r2' + str(m+1) + '.npy')
    FEj = np.load(outdir + 'KK_fe' + str(m+1) + '.npy')

    R2_1[:, :, m] = np.mean(R2j, 1)
    FE1[:, :, m] = FEj

    for j in range(KK):
        for r in range(R):
            fo_ = FOj[j, :, r]
            fo_ = fo_[fo_ > 0]
            Entropy1[j, r, m] = -np.sum(fo_ * np.log(fo_))
            maxFO1[j, r, m] = np.max(fo_)

            
# Code for the second set of subplots
FO2 = np.load(outdir + 'DD_fo1.npy')
dd_array = (10, 50, 100, 500, 1000, 2000, 4000, 8000, 15000)
dd_array_sub = dd_array[0::2]
(DD, K, R) = FO2.shape
Entropy2 = np.zeros((DD, R, 3))
maxFO2 = np.zeros((DD, R, 3))
R2_2 = np.zeros((DD, R, 3))
FE2 = np.zeros((DD, R, 3))

for m in range(3):
    FOj = np.load(outdir + 'DD_fo' + str(m+1) + '.npy')
    R2j = np.load(outdir + 'DD_r2' + str(m+1) + '.npy')
    FEj = np.load(outdir + 'DD_fe' + str(m+1) + '.npy')

    R2_2[:, :, m] = np.mean(R2j, 1)
    FE2[:, :, m] = FEj

    for j in range(DD):
        for r in range(R):
            fo_ = FOj[j, :, r]
            fo_ = fo_[fo_ > 0]
            Entropy2[j, r, m] = -np.sum(fo_ * np.log(fo_))
            maxFO2[j, r, m] = np.max(fo_)


linewidth=1
col = [ "#7b85d4", "#f3738","#83c995"]


# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(6.5, 3.5))
fig.set_dpi(300)  # Adjust the dpi if needed


# Define a function for plotting shaded error bars
def plot_shaded_error(ax, x_values, y_values, color, label=None):
    mean_values = np.mean(y_values, axis=1)
    error_values = np.std(y_values, axis=1)
    
    # Plot shaded error bar
    ax.fill_between(x_values, mean_values - error_values, mean_values + error_values, alpha=0.2, color=color, label=label)

    # Plot line with the average in purple/pink color
    ax.plot(x_values, mean_values, color=color, linewidth=1, label=f'Mean {label}' if label else None)


# Plotting the first set of subplots
for m in range(3):
    col2 = col[m]

    # Plot shaded error bar for R2_1
    plot_shaded_error(axes[0, 0], KK_array, R2_1[:, :, m], col2, label=f'Monkey {m}')

    # Plot shaded error bar for Entropy1
    plot_shaded_error(axes[0, 2], KK_array, Entropy1[:, :, m], col2)

    # Plot shaded error bar for FE1
    # Plot red dashed line at the minimum value
    mean_values = np.mean(FE1[:, :, m] / 1000, axis=1)
    axes[0, 1].axhline(y=np.min(mean_values), color='grey', linestyle='--', linewidth=0.7)
    plot_shaded_error(axes[0, 1], KK_array, FE1[:, :, m] / 1000, col2, label=f'Monkey {m}')

    # Plot shaded error bar for R2_2
    plot_shaded_error(axes[1, 0], np.arange(DD), R2_2[:, :, m], col2)

    # Plot shaded error bar for Entropy2
    plot_shaded_error(axes[1, 2], np.arange(DD), Entropy2[:, :, m], col2)

    # Plot shaded error bar for FE2
    plot_shaded_error(axes[1, 1], np.arange(DD), FE2[:, :, m] / 1000, col2)

# Set labels and adjust layout
axes[0, 0].set_ylabel('Explained var.', fontsize=fontsize)
axes[1, 0].set_ylabel('Explained var.', fontsize=fontsize)
axes[0, 1].set_ylabel('Free energy', fontsize=fontsize)
axes[1, 1].set_ylabel('Free energy', fontsize=fontsize)
axes[0, 2].set_ylabel('FO entropy', fontsize=fontsize)
axes[1, 2].set_ylabel('FO entropy', fontsize=fontsize)


# Set x-labels for both top and bottom plots
for ax in axes.flatten():
    ax.set_xlabel('No. of states', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize-2)  # Set tick labels fontsize to 8
    ax.grid(False)  # Turn off both x-axis and y-axis grids
    # Set spines' linewidth and color
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color('#333333')
    
# Set x-labels for the bottom plots
for ax in axes[1, :]:
    ax.set_xlabel('Temporal reg.', fontsize=fontsize)


plt.subplots_adjust(wspace=0.5, hspace=0.6)  # Adjust hspace for increased vertical space
plt.show()
