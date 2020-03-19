# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import math
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-path', help='path to folder contains experiments folders', default='C:\\Users\\user\\Documents\\Papers\\Affect-based\\Experiments\\gamma_searching\\negative', type=str)
parser.add_argument('--cov_t', '-cov_t', help='coverage threshold. to cope with too short segments', default=50, type=int)
parser.add_argument('--format', '-format', help='format to plot graph. choose from [png,svg]', default='png', type=str)
args = parser.parse_args()


# list of all experiments folders
exp_folders = [name for name in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, name))]
exp_folders = [os.path.join(args.path, f) for f in exp_folders]

exp_data = {}
for exp_folder in exp_folders:

    print("processing folder {}...".format(exp_folder))
    # list of all recordings data_folders
    data_folders = [name for name in os.listdir(exp_folder) if os.path.isdir(os.path.join(exp_folder, name))]
    data_folders = [os.path.join(exp_folder, f) for f in data_folders]

    durations, coverages = [], []
    duration_sub = 0.0
    for folder in data_folders:
        # open evaluation file
        eval_txt = pd.read_csv(os.path.join(folder, 'airsim_rec.txt'), sep='\t')

        # read file
        for i in range(0, eval_txt.shape[0], 1):

            if i > 0 and float(eval_txt.iloc[i-1][['StartingPoint']]) < float(eval_txt.iloc[i][['StartingPoint']]):
                duration_sub = float(eval_txt.iloc[i][['Duration']])

            if (i < eval_txt.shape[0] - 1 and float(eval_txt.iloc[i][['Coverage']]) > float(eval_txt.iloc[i+1][['Coverage']])) or (i == eval_txt.shape[0] - 1):
                # if the given coverage is higher than the threshold
                if float(eval_txt.iloc[i][['Coverage']]) > args.cov_t:
                    coverages.append(float(eval_txt.iloc[i][['Coverage']]))
                    durations.append(float(eval_txt.iloc[i][['Duration']])-duration_sub)
    
    # store coverage data for experiment
    durations_mean = np.array(durations).mean()
    coverages_mean = np.array(coverages).mean()#*10.7639 - constant to convert from m^2 to ft^2
    sessions = np.array(coverages).shape[0]
    coverages_ste = np.array(coverages).std() / math.sqrt(sessions)
    exp_data[float(exp_folder.split('_')[-1])] = [durations_mean,coverages_mean,coverages_ste]

print('done')

# sort data dict
ord_data = collections.OrderedDict(sorted(exp_data.items()))

# plot the results
plt.plot(list(ord_data.keys()), [v[1] for k,v in ord_data.items()])
plt.fill_between(list(ord_data.keys()), np.array([v[1] for k,v in ord_data.items()]) + 1.96 * np.array([v[2] for k,v in ord_data.items()]),\
                                        np.array([v[1] for k,v in ord_data.items()]) - 1.96 * np.array([v[2] for k,v in ord_data.items()]), alpha=0.5)
plt.legend(['Coverage'])
plt.xlabel('gamma')
plt.ylabel('Coverage')
plt.title('Coverage per gamma')
plt.grid(True)
plt.savefig(os.path.join(args.path,"coverage_gamma.{}".format(args.format)))