import os
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-path', help='path to experiment folder containing recordings folders', default='C:\\Users\\user\\Documents\\experiments\\data_collection_recordings\\straight', type=str)
parser.add_argument('--cov_t', '-cov_t', help='coverage threshold. to cope with too short segments', default=50, type=int)
args = parser.parse_args()

# list of all recordings data_folders
data_folders = [name for name in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, name))]
data_folders = [os.path.join(args.path, f) for f in data_folders]

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

durations_mean = np.array(durations).mean()
coverages_mean = np.array(coverages).mean()#*10.7639 - constant to convert from m^2 to ft^2
sessions = np.array(coverages).shape[0]

print("Durations mean: {}. Coverages mean: {}. Sessions: {}.".format(durations_mean,coverages_mean,sessions))