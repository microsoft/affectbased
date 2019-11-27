import os
import math
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-path', help='experiments folder path', default='C:\\Users\\user\\Documents\\experiments\\data_collection_recordings\\con_rgb', type=str)
parser.add_argument('--title', '-title', help='text to put in the title of the graph', default='Sketch-to-Image Translation', type=str)
parser.add_argument('--num_episodes', '-num_episodes', help='number of episodes to present graph for', default=50, type=int)
args = parser.parse_args()

# list of all methods data folders
methods_folders = [name for name in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, name))]
methods_folders = [os.path.join(args.path, f) for f in methods_folders]

loss_data = {}

for method_folder in methods_folders:

    # list of all experiments data folders inside method
    exp_folders = [name for name in os.listdir(method_folder) if os.path.isdir(os.path.join(method_folder, name))]
    exp_folders = [os.path.join(method_folder, f) for f in exp_folders]

    # initiate loss data for method
    method_name = method_folder.split('\\')[-1]
    loss_np = np.zeros((len(exp_folders),args.num_episodes), dtype=np.float64)

    print("Processing method {}...".format(method_name))

    for i, exp_folder in enumerate(exp_folders):

        # list of all episodes data folders inside the experiment
        episodes_folders = [name for name in os.listdir(exp_folder) if os.path.isdir(os.path.join(exp_folder, name))]
        episodes_folders = [os.path.join(exp_folder, f) for f in episodes_folders]

        for episode_folder in episodes_folders:

            # open log file and add loss data
            logfile_df = pd.read_csv(os.path.join(episode_folder, "log.txt"), sep='\t')
            episode_idx = int(episode_folder.split("\\")[-1])
            loss_np[i,episode_idx] = logfile_df.iloc[40]['TestLoss']
    
    # add loss values to dictionary by mean over episodes
    loss_np_meaned = np.mean(loss_np, axis=0)
    loss_np_std = 1.96 * np.std(loss_np, axis=0) / math.sqrt(len(exp_folders))
    loss_data[method_name] = [loss_np_meaned, loss_np_std]

# plot graph
keys_ordered = ['random','straight','il','il_curious_6.0']
labels_ordered = ['Random','Straight','Imitation learning','Ours']

for i, key in enumerate(keys_ordered):
    plt.plot(range(args.num_episodes), loss_data[key][0], label=labels_ordered[i])
    plt.fill_between(range(args.num_episodes), loss_data[key][0]+loss_data[key][1], loss_data[key][0]-loss_data[key][1], alpha=0.5)

plt.legend(loc='upper right')
plt.xlabel('Episodes')
plt.ylabel('Test loss')
plt.title(args.title)
plt.grid(True)
plt.savefig(os.path.join(args.path,"{}.svg".format(args.title)))

