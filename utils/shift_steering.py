import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--recordings_dir', '-recordings_dir', help='path to raw data folder', default='C:\\Users\\user\\Documents\\AirSim\\shifted_l2.0', type=str)
parser.add_argument('--shifting', '-shifting', help='amount of steering to shift', default=1.0, type=str)
args = parser.parse_args()

data_file = open(os.path.join(args.recordings_dir, 'airsim_rec.txt'),"r+")
data_file_shifted = open(os.path.join(args.recordings_dir, 'airsim_rec_shifted.txt'),"w+")

for i, line in enumerate(data_file.readlines()):
    line_list = line.split('\t')
    if i > 0:
        shifted_steering = min(max(float(line_list[-2]) + args.shifting, -1.0),1.0)
        line_list[-2] = str(shifted_steering)
    line = "\t".join(line_list)
    data_file_shifted.write(line)

data_file.close()
data_file_shifted.close()