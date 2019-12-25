#!/bin/bash

method="straight"
src_path="/home/user/experiments/data_collection_recordings"
dest_path="/home/user/experiments/data_collection_recordings/frame_res"

files=( $src_path"/"$method/* )

for i in {20..29}; do
	files_shuffled=( $(shuf -e "${files[@]}") )
	files_to_train=""
	
	for j in {00..49}; do
		files_to_train=$files_to_train${files_shuffled[10#$j]}";"
		python methods/vae/train.py --data_file $files_to_train --output_dir $dest_path"/"$method"/"$i"/"$j;
	done
done
