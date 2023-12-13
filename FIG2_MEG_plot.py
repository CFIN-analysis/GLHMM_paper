"""
This script is used in the context of the GLHMM paper, doi: 
https://doi.org/10.48550/arXiv.2312.07151

The script plots analyses run on a MEG dataset regarding human subjects performing a visual -memory task. See the paper for details on the analyses.
The dataset is described in Myers et al. (2015), and publicly available. See paper: https://elifesciences.org/articles/09000
The script outputs figure 4 of the paper.

Requirements: 
- matplotlib
- numpy


!!!Note!!!
The script loads the analyses outputed by the script "MEG.py", so it assumes you already have the analyses results in .npy files in the sub-folder called 'out' (located in the same folder as this script).
Hence, before running this script, run the "MEG.py" file and make sure the results are correctly saved and available to you. Change the data path in the variable 'outdir' if needed.


Author: Sonsoles Alonso Martinez
email: sonsoles.alonso@cfin.au.dk
Date: 12/12/2023
"""

# imports
import matplotlib.pyplot as plt
import numpy as np

# --------- SET PATHS and PARAMETERS specific to the analysis --------------

# data directory
outdir = '/out/'


K = 10
mr = np.zeros((116, 2, 3))  # time x metric x constrain
sr = np.zeros((116, K, 3))  # time x metric x constrain

# Create colormaps for the upper and lower subplots
cmap1 = plt.get_cmap('YlOrBr', K)
cmap2 = plt.get_cmap('YlGnBu', K)
font_size = 8

# Define the range of colors to use, avoiding the darkest ones
color_range = [0.2, 0.6]
line_width = 2


# ---------------------- LOAD DATA -----------------------
for isub in range(1, 11):
    REG = np.load(outdir + 'MEG_reg_S' + str(isub) + '.npy')
    mr += REG[:, :, 0, :] / 10
    SER = np.load(outdir + 'MEG_ser_S' + str(isub) + '.npy')
    sr += SER[:, :, 0, :] / 10

    
# --------------------------- PLOTS -------------------------------

fig = plt.figure(figsize=(6,5))

# Create a 2x2 grid for subplots
grid = plt.GridSpec(2, 2, hspace=0.6)

# Create subplots
ax_upper_left = plt.subplot(grid[0, 0])
ax_upper_right = plt.subplot(grid[0, 1])
ax_lower_middle = plt.subplot(grid[1, :])

# Plot data in the upper subplots
for k in range(K):
    normed_color = (k / (K - 1)) * (color_range[1] - color_range[0]) + color_range[0]
    ax_upper_left.plot(sr[:, k, 0], linewidth=line_width, color=cmap1(normed_color))
    ax_upper_right.plot(sr[:, k, 1], linewidth=line_width, color=cmap2(normed_color))

# Plot data in the lower_middle subplot
ax_lower_middle.plot(mr[:, 0, 0], linewidth=line_width + 2, color=cmap1(5))
ax_lower_middle.plot(mr[:, 0, 1], linewidth=line_width + 2, color=cmap2(5))


# Set the xlim for all subplots
for ax in [ax_upper_left, ax_upper_right, ax_lower_middle]:
    ax.set_xlim((1, 116))
    ax.tick_params(axis='both', labelsize=font_size-2)  # Set tick label font size to 8

# Set the ylim for the upper subplots
ax_upper_left.set_ylim((0.075, 0.15))
ax_upper_right.set_ylim((0, 0.4))

# Calculate the width of ax_upper_left
width = ax_upper_left.get_position().xmax - ax_upper_left.get_position().xmin

# Set the position of ax_lower_middle to be in the middle of the last row with the same width
ax_lower_middle.set_position([0.34, ax_lower_middle.get_position().y0, width, ax_lower_middle.get_position().height])


# Add labels and titles
ax_upper_left.set_title('Unconstrained solution' , fontsize=font_size ,fontweight='bold')
ax_upper_left.set_xlabel('Time' , fontsize=font_size)
ax_upper_left.set_ylabel('State evoked-response' , fontsize=font_size)
ax_upper_right.set_title('Constrained solution' , fontsize=font_size,fontweight='bold')
ax_upper_right.set_xlabel('Time', fontsize=font_size)
ax_lower_middle.set_title('Reaction time prediction' , fontsize=font_size, fontweight='bold')
ax_lower_middle.set_xlabel('Time' , fontsize=font_size)
ax_lower_middle.set_ylabel('Root squared error' , fontsize=font_size)
ax_lower_middle.legend(['unconstrained', 'constrained'], loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=font_size-2)

# Add pannel labels outside the subplots
fig.text(0.03, 0.94, 'A', fontsize=font_size+2, fontweight='bold')
fig.text(0.03, 0.46, 'B', fontsize=font_size+2, fontweight='bold')

plt.show()

