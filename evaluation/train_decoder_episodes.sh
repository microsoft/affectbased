#!/bin/bash

method="random"
task="con2rgb"
gpu="1"
data_file="cv/randompose_rgb_contours/randompose_rgb_contours_2000.h5"
src_path="/home/user/experiments/data_collection_recordings/frame_res"
dest_path="/home/user/experiments/data_collection_recordings/con_rgb_frozen"

for i in {23..29}; do
	for j in {00..49}; do
		model_path=$src_path"/"$method"/"$i"/"$j"/vaemodel40.ckpt"
		output_path=$dest_path"/"$method"/"$i"/"$j"/"
		#echo $model_path
		#echo $output_path
		python methods/vae/train_decoder.py --model_path $model_path --dset_file ../../Data/$data_file --task $task --cp_interval 40 --trainable_layers 0 --output_dir $output_path --gpu $gpu;
	done
done
