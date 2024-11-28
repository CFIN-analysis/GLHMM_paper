"""
This script is used in the context of the GLHMM paper, doi:
https://doi.org/10.48550/arXiv.2312.07151
This script concerns the comparison between GLHMM and standard HMM in the context of data analysis and behavioral
prediction on Local Field Potential (LFP) and spike data of rats performing an odor-memory task.
See the paper for details on the analyses.

The dataset is described in Shahbaba et al. (2022), and publicly available. See paper:https://doi.org/10.1038/s41467-022-28057-6

!!!Note!!!
The script assumes you already have downloaded the dataset, consisting of the continuous LFP, spike and behavioral data.
The raw data were preprocessed as part of a different project.
You can find the preprocessing script at https://github.com/LauraMasaracchia/state-conditioned_decoding/utils/preprocessing.py
The Preprocessing steps that are not included in this script regard:
 - PCA on the LFP data, only keep 1st component.
 - Filter the LFP 1st PC into power bands (keep only theta, 4-12 Hz).
 - Compute Power envelop the theta band
 - downsample data to 125Hz.

- Compute spike density from the spike data by means of a Gaussian kernel of width 10ms
- downsample data to 125Hz.

This script outputs figure 1 of the paper.

Requirements:
- GLHMM
The script assumes you have already correctly installed version 0.2.5 of the GLHMM toolbox.
You can download the toolbox using "pip install --user git+https://github.com/vidaurre/glhmm".
See documentation at: https://glhmm.readthedocs.io/en/latest/index.html

Other requirements to run this script:
- scikit-learn
- matplotlib
- numpy


Author: Laura Masaracchia
email: laurama@cfin.au.dk
Date: 22/11/2024

"""
# start with relevant imports

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from glhmm import glhmm, utils, statistics, preproc
from sklearn.decomposition import PCA,FastICA
from sklearn.linear_model import RidgeClassifierCV
import scipy.io
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------------------------------------------------
# ----------------------------define some useful functions---------------------------------------------
# -----------------------------------------------------------------------------------------------------

# plotting function
def plot_segments(ax, time, d, p_values, color):
    # Find indices of True values
    indices = np.where(p_values)[0]
    if len(indices) == 0:
        return  # No True values, return early        # Identify contiguous segments
    segments = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    if color =='black':
        # Adjust segments based on your criteria
        adjusted_segments = []
        for i, segment in enumerate(segments):
            if i == 0:
                # For the first segment, add the end value
                adjusted_segments.append(np.append(segment, segment[-1] + 1))
            else:
                # For other segments, add the start and end values
                adjusted_segments.append(np.insert(segment, 0, segment[0] - 1))
                adjusted_segments[-1] = np.append(adjusted_segments[-1], segment[-1] + 1)            # Plot each adjusted segment
        for segment in adjusted_segments:
            ax.plot(time[segment], d[segment], color=color, linewidth=3)
    else:
        for segment in segments:
            ax.plot(time[segment], d[segment], color=color, linewidth=3)    # Plotting the segments with color coding based on p-values


# perm test function
def perm_test(a1, a2, nperm=10000):
    """
    Perform a permutation test between two arrays a1 and a2.    Parameters:
    a1 (numpy.ndarray): The first input array of shape (T, N).
    a2 (numpy.ndarray): The second input array of shape (T, N).
    nperm (int, optional): The number of permutations. Default is 10000.    Returns:
    numpy.ndarray: A (T, 2) array of p-values.
    """    # Get the dimensions of the input arrays
    T, N = a1.shape    # Initialize the p-value array with ones
    pval = np.ones((T, 2))    # Initialize the base statistics array with zeros
    basestat = np.zeros((nperm, T))    # Compute the difference between the two input arrays
    a = a1 - a2    # Perform the permutation test
    for j in range(nperm):
        if j > 0:
            # Generate a random binary vector to decide which columns to flip
            pe = np.random.binomial(1, 0.5, N).astype(bool)            # Flip the selected columns of where pe is True
            a[:, pe] = -a[:, pe]        # Compute the mean of the difference array along the second axis, ignoring NaNs
        basestat[j, :] = np.nanmean(a, axis=1)    # Compute the p-values for the greater than or equal to comparison
    pval[:, 0] = np.sum(basestat >= basestat[0, :], axis=0) / (nperm + 1)    # Compute the p-values for the less than or equal to comparison
    pval[:, 1] = np.sum(basestat <= basestat[0, :], axis=0) / (nperm + 1)
    return pval, basestat


# function to compute class-balanced accuracy and output prediction tensor
def run_balanced_classification(X, y, n_bootstrap=100, return_prediction=False, return_coefficients=False):
    n_trials, n_units, n_time_points = X.shape
    # ------------ create a list of possible folds combining states and labels balance -------------------
    accuracy_stored = np.zeros(shape=(n_time_points, n_bootstrap))
    betas = np.zeros(shape=(n_units,n_time_points,n_bootstrap))
    if return_prediction:
        # initialize tensor with shape #trials, #time points, n_folds,
        # that you fill with a logical value for the right or wrong prediction
        pred_logik = np.empty(shape=(n_time_points,n_trials,n_bootstrap))
        pred_logik[:,:,:] = np.nan

    for t in range(n_time_points):
        # this code assumes that y is binary, 2 dimensional, i.e. can handle timevarying label
        X_time_point = X[:,:,t]
        y_time_point = y[:,t]

        if np.sum(y_time_point == 1) < np.sum(y_time_point == 0):
            smaller_class_nbr = np.sum(y_time_point == 1)
            smaller_label = 1
            bigger_label = 0
        else:
            smaller_class_nbr = np.sum(y_time_point == 0)
            smaller_label = 0
            bigger_label = 1

        # we will take 80% of these trials for training, and the same amount in the next
        trials_smallest_class = np.where(y_time_point == smaller_label)[0]  # smaller class trials indices
        trials_biggest_class = np.where(y_time_point == bigger_label)[0]  # bigger class trials indices

        # use kfolds only in the smallest batch
        training_nbr = int(0.8 * smaller_class_nbr)
        testing_nbr = int(0.2 * smaller_class_nbr)

        np.random.seed(10)
        print('timepoint t = %d' % t)
        if training_nbr<2:
            print('not enough training trials at time point t = %d. Skip t'%t)
            continue
        else:
            for k in range(n_bootstrap):
                # for every permutation shuffle the order of the trials,
                # get training and testing trials
                # same for both classes
                np.random.shuffle(trials_smallest_class)
                np.random.shuffle(trials_biggest_class)
                small_training_trials = trials_smallest_class[0:training_nbr]
                small_testing_trials = trials_smallest_class[training_nbr:training_nbr + testing_nbr]
                big_training_trials = trials_biggest_class[0:training_nbr]
                big_testing_trials = trials_biggest_class[training_nbr:training_nbr + testing_nbr]

                # concatenate trials from both classes and shuffle them
                train_index = np.concatenate((small_training_trials, big_training_trials))
                np.random.shuffle(train_index)
                test_index = np.concatenate((small_testing_trials, big_testing_trials))
                np.random.shuffle(test_index)

                # split everything into train and test
                # store test indices to use for gammas later
                X_train, X_test = X_time_point[train_index], X_time_point[test_index]
                y_train, y_test = y_time_point[train_index], y_time_point[test_index]

                # initialize and train model
                model = RidgeClassifierCV()
                model.fit(X_train, y_train)
                # get betas
                betas[:, t, k] = model.coef_

                # predict and compute accuracy
                y_pred = model.predict(X_test)
                accuracy_stored[t, k] = accuracy_score(y_test, y_pred)

                # store trial prediction
                # 1 if correctly predicted, 0 if wrong
                if return_prediction:
                    pred_logik[t,test_index,k] = (y_pred == y_test)*1.0

    if return_prediction:
        return accuracy_stored, pred_logik

    if return_coefficients:
        return accuracy_stored, betas

    else:
        return accuracy_stored


# -----------------------------------------------------------------------------------------------------
# ----------------------------------------- main script -----------------------------------------------
# -----------------------------------------------------------------------------------------------------
# these analyses are set to be performed on one rat at a time.
# change name of the rat for new analysis
# all rats names: 'Buchanan', 'Stella', 'Mitt', 'Barat', 'Superchris'

rat_name = 'Stella'
# change path to your folder
folder_name = '/Stella'

# ------------------------- set some decoding analysis parameters --------------------------------
n_perms = 100

# ------------------------------ set HMM analysis parameters -------------------------------
K = 4
n_rep_hmm = 10
hmm_options = {'cyc': 50, 'initcyc':15}

# -------------------------------------- load data ---------------------------------------------------
# ---------------------------- load behavioral data and extract important variables --------------

behavior_timings_file = '%s_behav_timing.mat' % rat_name.lower()
behav_timings = scipy.io.loadmat(os.path.join(folder_name, behavior_timings_file))

info_file = '%s_trial_info.npy' % rat_name.lower()
info_data = np.load(os.path.join(folder_name, info_file))

success_labels = info_data[:, 0]

poke_out_time = behav_timings['poke_out_time'] / 2  # for the resampling
trial_start_time = behav_timings['trial_subd'] / 2 # for the resampling

# ---------------------------- load spike density data -------------------------------------
units_density_continuous_name = 'units_density_continuous_125Hz.pkl'

with open(os.path.join(folder_name, units_density_continuous_name), 'rb') as fp:
    units_density_continuous = pickle.load(fp)

# ----------------------- load LFP data -----------------------------------------------
LFP_power_name = 'power_bands_continuous_data_125Hz.mat'
LFP_power = scipy.io.loadmat(os.path.join(folder_name, LFP_power_name))
LFP_power_resp = LFP_power['power_envelop_band']
theta_power = LFP_power_resp[:,1][:,np.newaxis]

# ----------------------------- analysis-specific preprocessing ------------------------------------------
# ------------------------ dimensionality reduction on the neural density --------------------------------
ica = FastICA(5)
units_decompose = ica.fit_transform(units_density_continuous)
tot_timepoints, n_units = units_decompose.shape
stand_units = preproc.preprocess_data(units_decompose, np.array([[0,tot_timepoints]]), post_standardise=True)[0]

# ---------------------- initialize where to store results -----------------------
result_glhmm_dict = {}
result_standard_hmm_dict = {}

# ---------------------------------------------------------------------------
# ------------------------------ set the HMMs ------------------------------

Y_glhmm = stand_units
X = preproc.preprocess_data(theta_power, np.array([[0,tot_timepoints]]), post_standardise=True)[0]
Y_standard = np.concatenate((stand_units, X), axis=1)


# the seed will ensure gamma is be the same for each rat, different per repetition
np.random.seed(1)

for rep in range(n_rep_hmm):
    result_glhmm_dict[rep] = {}
    result_standard_hmm_dict[rep] = {}

    # set the same gamma for both Hmms
    size_gamma0 = np.array([tot_timepoints])
    hmm0 = glhmm.glhmm(K=K)

    # Set Initial probabilities
    hmm0.Pi = np.full((1, K), 1 / (K))

    # Set Transition probabilities
    P = np.abs(np.random.randn(K, K))
    np.fill_diagonal(P, np.diag(P) + 0.5)
    P = P / np.sum(P, axis=1)[:, np.newaxis]
    hmm0.P = P

    gamma0 = glhmm.glhmm.sample_Gamma(hmm0, size=size_gamma0)

    # ---------------------- GLHMM -------------------------------
    # initialize
    LFP_spike_glhmm = glhmm.glhmm(model_beta='state',model_mean='no', K=K, covtype='shareddiag', dirichlet_diag=10000)

    print(LFP_spike_glhmm.hyperparameters)
    print('Training the GLHMM, rep %d, mouse %s'%(rep,rat_name))

    # train GLHMM on the LFP data to predict the units activity
    stc_glhmm, xi_glhmm, fe_glhmm = LFP_spike_glhmm.train(X=X,Y=Y_glhmm, Gamma=gamma0,options = hmm_options)

    # retrieve useful measures
    vpath_glhmm = LFP_spike_glhmm.decode(X=X, Y=Y_glhmm, viterbi=True)
    entropy_FO_glhmm = utils.get_FO_entropy(stc_glhmm,np.array([[0,tot_timepoints]]))
    #mean_glhmm = LFP_spike_glhmm.get_means()
    betas_glhmm = LFP_spike_glhmm.get_betas()
    cov_matrix_glhmm = LFP_spike_glhmm.get_covariance_matrix()

    result_glhmm_dict[rep] = {'stc': stc_glhmm,
                              'vpath': vpath_glhmm,
                              #'mean': mean_glhmm,
                              'betas': betas_glhmm,
                              'cov_matrix': cov_matrix_glhmm}

    # ---------------------- STANDARD HMM -------------------------------
    # initialize
    standard_hmm = glhmm.glhmm(model_beta='no', K=K, covtype='shareddiag', dirichlet_diag=10000)
    print(standard_hmm.hyperparameters)

    print('Training standard HMM, rep %d, mouse %s'%(rep, rat_name))
    stc_standard, xi_standard, fe_standard = standard_hmm.train(X=None, Y=Y_standard, Gamma=gamma0, options=hmm_options)
    vpath_standard = standard_hmm.decode(X=None,Y=Y_standard, viterbi=True)
    entropy_FO_standard_hmm = utils.get_FO_entropy(stc_standard, np.array([[0, tot_timepoints]]))
    #mean_standard =standard_hmm.get_means()
    cov_matrix_standard = standard_hmm.get_covariance_matrix()

    result_standard_hmm_dict[rep] = {'stc': stc_standard,
                              'vpath': vpath_standard,
                              #'mean': mean_standard,
                              'cov_matrix': cov_matrix_standard}

# ------------------------------------------------------------------------------------------------------------
# --------------------------------------- predict behaviour --------------------------------------------------
# perform decoding from the state time courses, including all repetitions
# first concatenate stc of all HMM runs
stc_glhmm_concat = np.zeros(shape=(tot_timepoints, K * n_rep_hmm))
stc_standard_concat = np.zeros(shape=(tot_timepoints, K * n_rep_hmm))

for rep in range(n_rep_hmm):
    stc_glhmm_concat[:, K * rep:K * (rep + 1)] = result_glhmm_dict[rep]['stc']
    stc_standard_concat[:, K * rep:K * (rep + 1)] = result_standard_hmm_dict[rep]['stc']

# predict trial outcome
y_regr = success_labels

# set regressors dimensions
n_trials = len(success_labels)
timepoints_per_trial = 501

# cut regressors data into trials
stc_glhmm_response_aligned = np.zeros(shape=(n_trials, K * n_rep_hmm, timepoints_per_trial))
stc_standard_response_aligned = np.zeros(shape=(n_trials, K * n_rep_hmm, timepoints_per_trial))

for i in range(n_trials):
    tmp = stc_glhmm_concat[int(poke_out_time[i, 0]) - 251:int(poke_out_time[i, 0]) + 250, :]
    stc_glhmm_response_aligned[i, :, :] = tmp.transpose(1, 0)
    tmp = stc_standard_concat[int(poke_out_time[i, 0]) - 251:int(poke_out_time[i, 0]) + 250, :]
    stc_standard_response_aligned[i, :, :] = tmp.transpose(1, 0)

# predict success from the stc of the different HMM
accuracy_concat_glhmm, pred_tensor_glhmm = run_balanced_classification(stc_glhmm_response_aligned, y_regr,
                                                                       n_bootstrap=n_perms, return_prediction=True)

accuracy_concat_standard, pred_tensor_standard = run_balanced_classification(stc_standard_response_aligned, y_regr,
                                                                             n_bootstrap=n_perms, return_prediction=True)

# -----------------------------------------------------------------------------------------------
# ------------------------------ plot results -----------------------------------------------------
# ------------------------------------------------------------------------------------------
# Initialize plot
n_cols = 4

fig, axes = plt.subplots(ncols=n_cols, figsize=(n_cols * 4.5, 3.5))
numticks = 5
# Common ticks
common_xticks = np.linspace(0, 1, numticks)  # Adjust this range as needed
common_yticks = np.linspace(0, 1, numticks)  # Adjust this range as needed
# Annotation labels
annotations = ['A', 'B', 'C', 'D']
xy =(-0.05, 1.1) # corrdinates of annotations
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cmap = matplotlib.colormaps['Set3']
colors = cmap.colors
labelsize= 14
annotsize = 18
textlabelsize = 16

ttrial, ntrials, nboots = pred_tensor_glhmm.shape
a_standard = np.zeros((ttrial, ntrials))
a_glhmm = np.zeros((ttrial, ntrials))
a_u_standard = np.zeros((ttrial, ntrials))
a_u_glhmm = np.zeros((ttrial, ntrials))
a_joint = np.zeros((ttrial, ntrials))
c_standard = np.zeros((ttrial, ntrials))
c_glhmm = np.zeros((ttrial, ntrials))
for ib in range(nboots):
    for t in range(ttrial):
        nn = ~np.isnan(pred_tensor_glhmm[t, :, ib])
        p1 = pred_tensor_glhmm[t, nn, ib]
        a_glhmm[t, nn] += p1
        c_glhmm[t, nn] += 1
        nn = ~np.isnan(pred_tensor_standard[t, :, ib])
        p2 = pred_tensor_standard[t, nn, ib]
        a_standard[t, nn] += p2
        c_standard[t, nn] += 1
        a_u_standard[t, nn] += np.logical_and(p2, np.logical_not(p1))
        a_u_glhmm[t, nn] += np.logical_and(p1, np.logical_not(p2))
        a_joint[t, nn] += np.logical_and(p1, p2)
    if not np.all(c_standard == c_glhmm):
        raise ValueError('wrong')    # Using np.divide with where parameter to handle division by zero
accuracy_s = np.divide(a_standard, c_standard, where=c_standard!=0, out=np.full_like(a_standard, np.nan))
accuracy_gl = np.divide(a_glhmm, c_glhmm, where=c_glhmm!=0, out=np.full_like(a_glhmm, np.nan))
accuracy_s_u = np.divide(a_u_standard, c_standard, where=c_standard!=0, out=np.full_like(a_u_standard, np.nan))
accuracy_gl_u = np.divide(a_u_glhmm, c_glhmm, where=c_glhmm!=0, out=np.full_like(a_u_glhmm, np.nan))
accuracy_j = np.divide(a_joint, c_glhmm, where=c_glhmm!=0, out=np.full_like(a_joint, np.nan))
p_val, bstats = perm_test(accuracy_gl, accuracy_s, 10000)
m1u = accuracy_s_u.T.flatten()
m1u = m1u[~np.isnan(m1u)].reshape(-1, ttrial)
m1 = accuracy_s.T.flatten()
m1 = m1[~np.isnan(m1)].reshape(-1, ttrial)
m2u = accuracy_gl_u.T.flatten()
m2u = m2u[~np.isnan(m2u)].reshape(-1, ttrial)
m2 = accuracy_gl.T.flatten()
m2 = m2[~np.isnan(m2)].reshape(-1, ttrial)
md12 = (accuracy_gl - accuracy_s).T.flatten()
md12 = md12[~np.isnan(md12)].reshape(-1, ttrial)
pca = PCA(n_components=1)
b = pca.fit_transform(md12)
b = np.argsort(b[:, 0])
md12 = md12[b, :]
mj12 = accuracy_j.T.flatten()
mj12 = mj12[~np.isnan(mj12)].reshape(-1, ttrial)

# ---------------------- Plotting the figure ----------------------------
# FIGURE A
ax = axes[0]
ax.plot(np.mean(m2, axis=0), color=colors[3], linewidth=3, label='Regression-based HMM' )
ax.plot(np.mean(m1, axis=0), default_colors[0], linewidth=3, label='Gaussian HMM')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(np.linspace(0, ttrial, numticks).round().astype(int))
xticks_display = np.linspace(-2, 2, numticks).round(2)
ax.set_xticklabels(xticks_display)
ax.set_xlim([0, ttrial])
#ax.set_xticks(np.linspace(-2, 2, numticks))
ax.set_yticks(np.linspace(0, 1, numticks))
#ax.set_xlim([-2, 2])
ax.set_xlabel('Time (s)', fontsize=textlabelsize)  # Set x label
ax.set_ylabel('Accuracy', fontsize=textlabelsize)  # Set y label
ax.annotate(annotations[0], xy=xy, xycoords='axes fraction', fontsize=annotsize, fontweight='bold', ha='right', va='center')
ax.tick_params(axis='both', labelsize=labelsize)  # Set tick label fontsize
ax.legend(loc='lower left', fontsize=labelsize)  # Customize location and fontsize

# FIGURE B
ax = axes[1]
d = np.mean(m2u, axis=0) - np.mean(m1u, axis=0)
alpha = 0.01
# accuracy_gl better than accuracy_s
p_values1 =statistics.pval_cluster_based_correction(bstats,p_val[:, 0],alpha=alpha)
# accuracy_s better than accuracy_gl
p_values2 =statistics.pval_cluster_based_correction(bstats,p_val[:, 1],alpha=alpha)
# Plotting the signal with color coding based on p-values
# Prepare time axis
time = np.linspace(0, len(d), len(d))
plot_segments(ax, time, d, (p_values1 > 0.01) & (p_values2 > 0.01), 'black')  # Both conditions (red)
plot_segments(ax, time, d, p_values1 < 0.01, colors[3])  # p_values1 < 0.01 (red)
plot_segments(ax, time, d, p_values2 < 0.01, default_colors[0])  # p_values2 < 0.01 (blue)
# Add a dashed horizontal line at zero
ax.axhline(y=0, color='gray', linewidth=2, linestyle='--')
ax.set_yticks(((np.linspace(np.min(d)-0.01, np.max(d)+0.01, numticks)).round(2)))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.set_xticks(np.linspace(-2, 2, numticks))
# ax.set_xlim([-2, 2])
ax.set_xticks(np.linspace(0, ttrial, numticks).round().astype(int))
xticks_display = np.linspace(-2, 2, numticks).round(2)
#ax.set_xlim([-2, 2])
ax.set_xticklabels(xticks_display)
ax.set_xlim([0, ttrial])
ax.set_xlabel('Time (s)', fontsize=textlabelsize)  # Set x label
ax.set_ylabel('Accuracy Diff', fontsize=textlabelsize)  # Set y label
ax.annotate(annotations[1], xy=xy, xycoords='axes fraction', fontsize=annotsize, fontweight='bold', ha='right', va='center')
ax.tick_params(axis='both', labelsize=labelsize)  # Set tick label fontsize

# FIGURE C
ax = axes[2]
bar_width = 1
ax.bar(np.arange(ttrial), np.mean(m1u, axis=0), width=bar_width, label='Mean m1u', color=default_colors[0]) # blue
ax.bar(np.arange(ttrial), np.mean(m2u, axis=0), bottom=np.mean(m1u, axis=0), width=bar_width, label='Mean m2u', color=colors[3])
ax.bar(np.arange(ttrial), np.mean(mj12, axis=0), bottom=np.mean(m1u, axis=0) + np.mean(m2u, axis=0), width=bar_width, label='Mean mj12', color=colors[5]) # yellow
ax.set_xticks(np.linspace(0, ttrial, numticks).round().astype(int))
xticks_display = np.linspace(-2, 2, numticks).round(2)
ax.set_xticklabels(xticks_display)
#ax.set_xlim([-2, 2])
ax.set_xlim(0, ttrial)
# ax.set_xticks(np.linspace(-2, 2, numticks))    ax.set_yticks(np.linspace(0, 1, numticks))
ax.set_xlabel('Time (s)', fontsize=textlabelsize)  # Set x label
ax.set_ylabel('Accuracy', fontsize=textlabelsize)  # Set y label
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.annotate(annotations[2], xy=xy, xycoords='axes fraction', fontsize=annotsize, fontweight='bold', ha='right', va='center')
ax.tick_params(axis='both', labelsize=labelsize)  # Set tick label fontsize

# FIGURE D
ax = axes[3]
im = ax.imshow(md12, aspect='auto', cmap="coolwarm")
ax.set_xticks(np.linspace(0, md12.shape[1], numticks).round().astype(int))
#ax.set_xticks(np.linspace(-2, 2, numticks))
xticks_display = np.linspace(-2, 2, numticks).round(2)
ax.set_xticklabels(xticks_display)
ax.set_yticks(np.linspace(0, md12.shape[0], numticks).round().astype(int))
ax.set_xlabel('Time (s)', fontsize=textlabelsize)  # Set x label
ax.set_ylabel('Trials', fontsize=textlabelsize)  # Set y label
ax.annotate(annotations[3], xy=xy, xycoords='axes fraction', fontsize=annotsize, fontweight='bold', ha='right', va='center')
ax.tick_params(axis='both', labelsize=labelsize)  # Set tick label fontsize
plt.tight_layout()
plt.show()
