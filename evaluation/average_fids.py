from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.append('.')
import os
import glob
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fids_path', '-fids_path', help='path to folder containing FID values files', default='C:\\Users\\user\\Documents\\experiments\\generating_samples\\con_rgb\\fids', type=str)
args = parser.parse_args()

fids = {}
for filename in glob.glob(os.path.join(args.fids_path,"*.txt")):
    extracted_fn = filename.split('.txt')[-2].split('\\')[-1].split('/')[-1]
    method = ''.join(extracted_fn.split('_')[:-1])
    exp_idx = int(extracted_fn.split('_')[-1])
    log_file = open(filename,"r")
    fid_val = float(log_file.read())

    if method not in fids:
        fids[method] = np.zeros((30))
    fids[method][exp_idx] = fid_val

for key, val in fids.items():
    print("{} - mean: {}, std: {}".format(key,val.mean(),val.std()))
