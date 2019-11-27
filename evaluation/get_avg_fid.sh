#!/bin/bash

task="con_rgb"
src_path="/home/user/experiments/generating_samples/models"
method_path="/home/user/experiments/generating_samples/"$task
models_path="/home/user/experiments/data_collection_recordings"

data_path=$method_path"/inputs"
output_path=$method_path"/fids"

method_folders=( $src_path"/"$task/* )

for method in "${method_folders[@]}"; do
	method_splitted=(${method//// })
	method_name=${method_splitted[-1]}
	for i in {00..29}; do
		model_path=$method"/"$i"/10/vaemodel40.ckpt"
		log_path=$output_path"/"$method_name"_"$i".txt"
		python evaluation/get_fid.py --model_path $model_path --images_path $data_path --log_path $log_path --task $task
	done
done
