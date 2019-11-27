import os
import h5py
import math
import argparse
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import glob
import random
from PIL import Image

def prepare_h5_file_for_imitation_model(dset_folder, res, buffer_size, images_gap):

    # list of all recordings data_folders
    data_folders = [name for name in os.listdir(dset_folder) if os.path.isdir(os.path.join(dset_folder, name))]
    data_folders = [os.path.join(dset_folder, f) for f in data_folders]

    images_all = []
    labels_all = []
    for folder in data_folders:
        print('Reading data from {0}...'.format(folder))
        # check if the file was shifted
        if os.path.isfile(os.path.join(folder, 'airsim_rec_shifted.txt')):
            current_df = pd.read_csv(os.path.join(folder, 'airsim_rec_shifted.txt'), sep='\t')
        else:
            current_df = pd.read_csv(os.path.join(folder, 'airsim_rec.txt'), sep='\t')

        for i in range(((buffer_size - 1)*images_gap), current_df.shape[0], 1):

            # store sequence of images names
            images = []
            start_idx = i - ((buffer_size - 1)*images_gap)
            for j in range(start_idx, i+images_gap, images_gap):
                im = Image.open(os.path.join(os.path.join(folder, 'images'), current_df.iloc[j]['ImageFile']))
                if res > 0:
                    im = im.resize((res,res), Image.BILINEAR)
                im_np = np.array(im, dtype=np.float32) / 255.0
                images.append(im_np)
            images_np = np.array(images, dtype=np.float32).transpose(1,2,0)

            # store label
            current_label = float(current_df.iloc[i][['Steering']])
            current_label = [int((current_label + 1.0) * 2.0)] # value => class index
            current_label = np.array(current_label, dtype=np.float32)

            images_all.append(images_np)
            labels_all.append(current_label)

    images_np = np.array(images_all, dtype=np.float32)
    labels_np = np.array(labels_all, dtype=np.float32)

    # write h5 file to the same location
    dest_filename = os.path.basename(os.path.normpath(dset_folder))
    with h5py.File(os.path.join(dset_folder,"{}.h5".format(dest_filename)), "w") as f:
        f.create_dataset("images", data=images_np, dtype=images_np.dtype)
        f.create_dataset("labels", data=labels_np, dtype=labels_np.dtype)

def prepare_h5_file_for_visceral_model(dset_file, res, grayscale=False, vel=False, cov_size=0):

    # open dataset text files
    dataset_txt = open(dset_file,"r")

    # extract data from train text file
    images, labels = [], []
    if vel == True:
        velocities = []

    for i, line in enumerate(dataset_txt):
        # get image
        image_path = line.split("\t")[0]
        im = Image.open(image_path).resize((res,res), Image.BILINEAR)
        if grayscale:
            im = im.convert("L")
        else:
            im = im.convert("RGB")
        image_np = np.array(im, dtype=np.float32) / 255.0

        # get cov image
        if cov_size > 0:
            cov_path = image_path.replace("img__","cov__").replace("\\images\\", "\\covs\\")
            cov_im = Image.open(cov_path).resize((cov_size,cov_size), Image.BILINEAR)
            cov_np = np.array(cov_im, dtype=np.float32) / 255.0
            if grayscale:
                image_np[:cov_size,:cov_size] = cov_np
            else:
                cov_np = np.repeat(np.expand_dims(cov_np, axis=-1), repeats=3, axis=-1)
                image_np[:cov_size,:cov_size,:] = cov_np

        images.append(image_np)

        # get label as [positive_value, negative_value]
        pos_emotion = float(line.split("\t")[1])
        neg_emotion = float(line.split("\t")[2].split("\n")[0])
        label_np = np.array([pos_emotion,neg_emotion], dtype=np.float32)
        labels.append(label_np)

        # add velocity to h5 file
        if vel:
            velocity_val = float(line.split("\t")[3].split("\n")[0])
            velocity_np = np.array([velocity_val], dtype=np.float32)
            velocities.append(velocity_np)

        if i % 1000 == 0:
            print("processing {} images".format(i))

    # define trainset numpy
    images_np = np.array(images, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.float32)
    if vel:
        velocities_np = np.array(velocities, dtype=np.float32)

    # write h5 file to the same location
    dest_filename = dset_file.split('/')[-1].split('\\')[-1].split('.')[0]
    if cov_size > 0:
        dest_filename = dest_filename + "_cov"
    dest_dir = os.path.dirname(os.path.normpath(dset_file))
    with h5py.File(os.path.join(dest_dir,"{}.h5".format(dest_filename)), "w") as f:
        f.create_dataset("images", data=images_np, dtype=images_np.dtype)
        f.create_dataset("labels", data=labels_np, dtype=labels_np.dtype)
        if vel:
            f.create_dataset("velocities", data=velocities_np, dtype=velocities_np.dtype)

def prepare_h5_for_frame_restoration(dset_folder, res):
    
    # list of all recordings data_folders
    data_folders = [name for name in os.listdir(dset_folder) if os.path.isdir(os.path.join(dset_folder, name))]
    data_folders = [os.path.join(dset_folder, f) for f in data_folders]

    images_all = []
    for folder in data_folders:
        print('Reading data from {0}...'.format(folder))
        current_df = pd.read_csv(os.path.join(folder, 'airsim_rec.txt'), sep='\t')

        for i in range(0, current_df.shape[0], 1):
            # store image
            im = Image.open(os.path.join(os.path.join(folder, 'images'), current_df.iloc[i]['ImageFile']))
            if res > 0:
                im = im.resize((res,res), Image.BILINEAR)
            im_np = np.array(im, dtype=np.float32) / 255.0
            images_all.append(im_np)

    images_np = np.array(images_all, dtype=np.float32)
    if len(images_np.shape) == 3:
        images_np = np.expand_dims(images_np, axis=-1)

    # write h5 file to the same location
    dest_filename = os.path.basename(os.path.normpath(dset_folder))
    with h5py.File(os.path.join(dset_folder,"{}.h5".format(dest_filename)), "w") as f:
        f.create_dataset("images", data=images_np, dtype=images_np.dtype)

def prepare_h5_for_frames_prediction(dset_folder, res, frames):
    
    # list of all recordings data_folders
    data_folders = [name for name in os.listdir(dset_folder) if os.path.isdir(os.path.join(dset_folder, name))]
    data_folders = [os.path.join(dset_folder, f) for f in data_folders]

    images_all = []
    for folder in data_folders:
        print('Reading data from {0}...'.format(folder))
        current_df = pd.read_csv(os.path.join(folder, 'airsim_rec.txt'), sep='\t')

        for i in range(0, current_df.shape[0], 1):
            # store image
            im = Image.open(os.path.join(os.path.join(folder, 'images'), current_df.iloc[i]['ImageFile']))
            if res > 0:
                im = im.resize((res,res), Image.BILINEAR)
            im_np = np.array(im, dtype=np.float32) / 255.0
            images_all.append(im_np)

    images_np = np.array(images_all, dtype=np.float32)
    if len(images_np.shape) == 3:
        images_np = np.expand_dims(images_np, axis=-1)

    # write h5 file to the same location
    dest_filename = os.path.basename(os.path.normpath(dset_folder))
    with h5py.File(os.path.join(dset_folder,"{}.h5".format(dest_filename)), "w") as f:
        f.create_dataset("images", data=images_np, dtype=images_np.dtype)

def prepare_h5_for_frame_restoration_trial_separated(dset_folder, res, cov_t):
    
    # list of all recordings data_folders
    data_folders = [name for name in os.listdir(dset_folder) if os.path.isdir(os.path.join(dset_folder, name))]
    data_folders = [os.path.join(dset_folder, f) for f in data_folders]

    images_all = []
    images_set = []
    for folder in data_folders:
        print('Reading data from {0}...'.format(folder))
        current_df = pd.read_csv(os.path.join(folder, 'airsim_rec.txt'), sep='\t')

        for i in range(0, current_df.shape[0], 1):

            # store image
            im = Image.open(os.path.join(os.path.join(folder, 'images'), current_df.iloc[i]['ImageFile']))
            if res > 0:
                im = im.resize((res,res), Image.BILINEAR)
            im_np = np.array(im, dtype=np.float32) / 255.0
            images_set.append(im_np)

            if (i < current_df.shape[0] - 1 and float(current_df.iloc[i][['Coverage']]) > float(current_df.iloc[i+1][['Coverage']])) or (i == current_df.shape[0] - 1):
                # if the given coverage is higher than the threshold
                if float(current_df.iloc[i][['Coverage']]) > cov_t:
                    images_all.append(images_set)
                images_set = []

    # print statistics
    images_num = np.array([len(x) for x in images_all])
    print("Numer of images in session:")
    print("Min: {}, max: {}, mean: {}.".format(images_num.min(),images_num.max(),images_num.mean()))

    dest_filename = os.path.basename(os.path.normpath(dset_folder))
    for i, images_set in enumerate(images_all):
        
        print("Saving {}_{}.h5...".format(dest_filename, i))
        images_set_np = np.array(images_set, dtype=np.float32)
        if len(images_set_np.shape) == 3:
            images_set_np = np.expand_dims(images_set_np, axis=-1)
        
        # write h5 file to the same location
        with h5py.File(os.path.join(dset_folder,"{}_{}.h5".format(dest_filename, i)), "w") as f:
            f.create_dataset("images", data=images_set_np, dtype=images_set_np.dtype)


def prepare_h5_for_frame_prediction(dset_folder, res, images_gap):
    
    # list of all recordings data_folders
    data_folders = [name for name in os.listdir(dset_folder) if os.path.isdir(os.path.join(dset_folder, name))]
    data_folders = [os.path.join(dset_folder, f) for f in data_folders]

    images_all = []
    target_images_all = []
    for folder in data_folders:
        print('Reading data from {0}...'.format(folder))
        current_df = pd.read_csv(os.path.join(folder, 'airsim_rec.txt'), sep='\t')

        for i in range(0, current_df.shape[0] - images_gap, 1):
            # store images
            im = Image.open(os.path.join(os.path.join(folder, 'images'), current_df.iloc[i]['ImageFile']))
            if res > 0:
                im = im.resize((res,res), Image.BILINEAR)
            im_np = np.array(im, dtype=np.float32) / 255.0
            images_all.append(im_np)

            # store future image
            im = Image.open(os.path.join(os.path.join(folder, 'images'), current_df.iloc[i+images_gap]['ImageFile']))
            if res > 0:
                im = im.resize((res,res), Image.BILINEAR)
            im_np = np.array(im, dtype=np.float32) / 255.0
            target_images_all.append(im_np)            

    images_np = np.array(images_all, dtype=np.float32)
    target_images_np = np.array(target_images_all, dtype=np.float32)
    if len(images_np.shape) == 3:
        images_np = np.expand_dims(images_np, axis=-1)
        target_images_np = np.expand_dims(target_images_np, axis=-1)

    # write h5 file to the same location
    dest_filename = os.path.basename(os.path.normpath(dset_folder))
    with h5py.File(os.path.join(dset_folder,"{}_prediction_{}.h5".format(dest_filename,images_gap)), "w") as f:
        f.create_dataset("images", data=images_np, dtype=images_np.dtype)
        f.create_dataset("target_images", data=target_images_np, dtype=target_images_np.dtype)

def prepare_h5_for_depth_estimation(dset_folder, res, num_imgs=100000):
    
    # list of all recordings data_folders
    data_folders = [name for name in os.listdir(dset_folder) if os.path.isdir(os.path.join(dset_folder, name))]
    data_folders = [os.path.join(dset_folder, f) for f in data_folders]

    images_all = []
    depths_all = []
    for folder in data_folders:
        print('Reading data from {0}...'.format(folder))
        current_df = pd.read_csv(os.path.join(folder, 'airsim_rec.txt'), sep='\t')

        for i in range(0, current_df.shape[0], 1):
            image_path = os.path.join(os.path.join(folder, 'images'), current_df.iloc[i]['ImageFile'])

            # store images
            im = Image.open(image_path)
            if res > 0:
                im = im.resize((res,res), Image.BILINEAR)
            im_np = np.array(im, dtype=np.float32) / 255.0
            images_all.append(im_np)

            depth_im = Image.open(image_path.replace("\\im_","\\depth_").replace("/im_","/depth_"))
            if res > 0:
                depth_im = depth_im.resize((res,res), Image.BILINEAR)
            depth_np = np.array(depth_im, dtype=np.float32) / 255.0
            depths_all.append(depth_np)            

    images_np = np.array(images_all, dtype=np.float32)
    depths_np = np.array(depths_all, dtype=np.float32)
    if len(images_np.shape) == 3:
        images_np = np.expand_dims(images_np, axis=-1)
    if len(depths_np.shape) == 3:
        depths_np = np.expand_dims(depths_np, axis=-1)

    # shuffle images and labels in the same order
    p = np.random.permutation(images_np.shape[0])
    images_np = images_np[p]
    depths_np = depths_np[p]

    # trim data if asked to
    if images_np.shape[0] > num_imgs:
        images_np = images_np[:num_imgs,:,:,:]
        depths_np = depths_np[:num_imgs,:,:,:]

    # write h5 file to the same location
    dest_filename = os.path.basename(os.path.normpath(dset_folder))
    with h5py.File(os.path.join(dset_folder,"{}_{}.h5".format(dest_filename,num_imgs)), "w") as f:
        f.create_dataset("images", data=images_np, dtype=images_np.dtype)
        f.create_dataset("depths", data=depths_np, dtype=depths_np.dtype)

def prepare_h5_for_semantic_segmentation(dset_folder, res, num_imgs=100000):
    
    # list of all recordings data_folders
    data_folders = [name for name in os.listdir(dset_folder) if os.path.isdir(os.path.join(dset_folder, name))]
    data_folders = [os.path.join(dset_folder, f) for f in data_folders]

    images_all = []
    segs_all = []
    for folder in data_folders:
        print('Reading data from {0}...'.format(folder))
        current_df = pd.read_csv(os.path.join(folder, 'airsim_rec.txt'), sep='\t')

        for i in range(0, current_df.shape[0], 1):
            image_path = os.path.join(os.path.join(folder, 'images'), current_df.iloc[i]['ImageFile'])

            # store images
            im = Image.open(image_path)
            if res > 0:
                im = im.resize((res,res), Image.BILINEAR)
            im_np = np.array(im, dtype=np.float32) / 255.0
            images_all.append(im_np)

            seg_im = Image.open(image_path.replace("\\im_","\\seg_").replace("/im_","/seg_"))
            if res > 0:
                seg_im = seg_im.resize((res,res), Image.BILINEAR)
            seg_np = np.array(seg_im, dtype=np.float32) / 255.0
            segs_all.append(seg_np)            

    images_np = np.array(images_all, dtype=np.float32)
    segs_np = np.array(segs_all, dtype=np.float32)
    if len(images_np.shape) == 3:
        images_np = np.expand_dims(images_np, axis=-1)
    if len(segs_np.shape) == 3:
        segs_np = np.expand_dims(segs_np, axis=-1)

    # shuffle images and labels in the same order
    p = np.random.permutation(images_np.shape[0])
    images_np = images_np[p]
    segs_np = segs_np[p]

    # trim data if asked to
    if images_np.shape[0] > num_imgs:
        images_np = images_np[:num_imgs,:,:,:]
        segs_np = segs_np[:num_imgs,:,:,:]

    # write h5 file to the same location
    dest_filename = os.path.basename(os.path.normpath(dset_folder))
    with h5py.File(os.path.join(dset_folder,"{}_{}.h5".format(dest_filename,num_imgs)), "w") as f:
        f.create_dataset("images", data=images_np, dtype=images_np.dtype)
        f.create_dataset("segs", data=segs_np, dtype=segs_np.dtype)

def prepare_h5_for_rgb_creation(dset_folder, res, num_imgs=100000):
    
    # list of all recordings data_folders
    data_folders = [name for name in os.listdir(dset_folder) if os.path.isdir(os.path.join(dset_folder, name))]
    data_folders = [os.path.join(dset_folder, f) for f in data_folders]

    images_all = []
    conts_all = []
    for folder in data_folders:
        print('Reading data from {0}...'.format(folder))
        current_df = pd.read_csv(os.path.join(folder, 'airsim_rec.txt'), sep='\t')

        for i in range(0, current_df.shape[0], 1):
            image_path = os.path.join(os.path.join(folder, 'images'), current_df.iloc[i]['ImageFile'])

            # store images
            im = Image.open(image_path)
            if res > 0:
                im = im.resize((res,res), Image.BILINEAR)
            im_np = np.array(im, dtype=np.float32) / 255.0
            images_all.append(im_np)

            # open segmentation and grayscale it
            seg_im = Image.open(image_path.replace("\\im_","\\seg_").replace("/im_","/seg_")).convert('L')
            if res > 0:
                seg_im = seg_im.resize((res,res), Image.BILINEAR)
            seg_np = np.array(seg_im, dtype=np.uint8)

            # create contours and save them
            thresh = 170
            _, thresh = cv2.threshold(seg_np,thresh,255,0)
            contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

            cont_im = np.zeros(seg_np.shape)
            for contour in contours:
                cv2.drawContours(cont_im,contour.astype('int'),-1,(255,255,255),2)
            cv2.imwrite(image_path.replace("\\im_","\\cont_").replace("/im_","/cont_"),cont_im)
            conts_all.append(cont_im / 255.0)

            if i % 200 == 0:
                print("{}/{} images added".format(i,current_df.shape[0]))

    images_np = np.array(images_all, dtype=np.float32)
    conts_np = np.array(conts_all, dtype=np.float32)
    if len(images_np.shape) == 3:
        images_np = np.expand_dims(images_np, axis=-1)
    if len(conts_np.shape) == 3:
        conts_np = np.expand_dims(conts_np, axis=-1)

    # shuffle images and labels in the same order
    p = np.random.permutation(images_np.shape[0])
    images_np = images_np[p]
    conts_np = conts_np[p]

    # trim data if asked to
    if images_np.shape[0] > num_imgs:
        images_np = images_np[:num_imgs,:,:,:]
        conts_np = conts_np[:num_imgs,:,:,:]

    # write h5 file to the same location
    dest_filename = os.path.basename(os.path.normpath(dset_folder))+'_contours'
    with h5py.File(os.path.join(dset_folder,"{}_{}.h5".format(dest_filename,num_imgs)), "w") as f:
        f.create_dataset("rgbs", data=images_np, dtype=images_np.dtype)
        f.create_dataset("images", data=conts_np, dtype=conts_np.dtype)

def prepare_h5_for_flow_estimation(dset_folder, res, num_imgs=100000):
    
    # list of all recordings data_folders
    data_folders = [name for name in os.listdir(dset_folder) if os.path.isdir(os.path.join(dset_folder, name))]
    data_folders = [os.path.join(dset_folder, f) for f in data_folders]

    images_all = []
    nexts_all = []
    flows_all = []
    for folder in data_folders:
        print('Reading data from {0}...'.format(folder))
        current_df = pd.read_csv(os.path.join(folder, 'airsim_rec.txt'), sep='\t')

        for i in range(0, current_df.shape[0], 1):
            image_path = os.path.join(os.path.join(folder, 'images'), current_df.iloc[i]['ImageFile'])

            # store source image
            im = Image.open(image_path)
            if res > 0:
                im = im.resize((res,res), Image.BILINEAR)
            im_np = np.array(im, dtype=np.float32) / 255.0
            images_all.append(im_np)

            # store next image
            next_im = Image.open(image_path.replace("\\im_","\\next_").replace("/im_","/next_"))
            if res > 0:
                next_im = next_im.resize((res,res), Image.BILINEAR)
            next_np = np.array(next_im, dtype=np.float32) / 255.0
            nexts_all.append(next_np)

            # store flow map numpy
            flow_im = np.load(image_path.replace("\\im_","\\flow_").replace("/im_","/flow_").replace(".png",".npy"))
            flows_all.append(flow_im)

    images_np = np.array(images_all, dtype=np.float32)
    nexts_np = np.array(nexts_all, dtype=np.float32)
    flows_np = np.array(flows_all, dtype=np.float32)
    if len(images_np.shape) == 3:
        images_np = np.expand_dims(images_np, axis=-1)
    if len(nexts_np.shape) == 3:
        nexts_np = np.expand_dims(nexts_np, axis=-1)
    if len(flows_np.shape) == 3:
        flows_np = np.expand_dims(flows_np, axis=-1)
    
    # shuffle images and labels in the same order
    p = np.random.permutation(images_np.shape[0])
    images_np = images_np[p]
    nexts_np = nexts_np[p]
    flows_np = flows_np[p]

    # trim data if asked to
    if images_np.shape[0] > num_imgs:
        images_np = images_np[:num_imgs,:,:,:]
        nexts_np = nexts_np[:num_imgs,:,:,:]
        flows_np = flows_np[:num_imgs,:,:,:]

    # write h5 file to the same location
    dest_filename = os.path.basename(os.path.normpath(dset_folder))
    with h5py.File(os.path.join(dset_folder,"{}_{}.h5".format(dest_filename,num_imgs)), "w") as f:
        f.create_dataset("images", data=images_np, dtype=images_np.dtype)
        f.create_dataset("nexts", data=nexts_np, dtype=nexts_np.dtype)
        f.create_dataset("flows", data=flows_np, dtype=flows_np.dtype)

def prepare_h5_for_frame_prediction(dset_folder, res, num_imgs=100000):
    
    # list of all recordings data_folders
    data_folders = [name for name in os.listdir(dset_folder) if os.path.isdir(os.path.join(dset_folder, name))]
    data_folders = [os.path.join(dset_folder, f) for f in data_folders]

    images_all = []
    nexts_all = []
    for folder in data_folders:
        print('Reading data from {0}...'.format(folder))
        current_df = pd.read_csv(os.path.join(folder, 'airsim_rec.txt'), sep='\t')

        for i in range(0, current_df.shape[0], 1):
            image_path = os.path.join(os.path.join(folder, 'images'), current_df.iloc[i]['ImageFile'])

            # store source image
            im = Image.open(image_path)
            if res > 0:
                im = im.resize((res,res), Image.BILINEAR)
            im_np = np.array(im, dtype=np.float32) / 255.0
            images_all.append(im_np)

            # store next image
            next_im = Image.open(image_path.replace("\\im_","\\next_").replace("/im_","/next_"))
            if res > 0:
                next_im = next_im.resize((res,res), Image.BILINEAR)
            next_np = np.array(next_im, dtype=np.float32) / 255.0
            nexts_all.append(next_np)

    images_np = np.array(images_all, dtype=np.float32)
    nexts_np = np.array(nexts_all, dtype=np.float32)
    if len(images_np.shape) == 3:
        images_np = np.expand_dims(images_np, axis=-1)
    if len(nexts_np.shape) == 3:
        nexts_np = np.expand_dims(nexts_np, axis=-1)
    
    # shuffle images and labels in the same order
    p = np.random.permutation(images_np.shape[0])
    images_np = images_np[p]
    nexts_np = nexts_np[p]

    # trim data if asked to
    if images_np.shape[0] > num_imgs:
        images_np = images_np[:num_imgs,:,:,:]
        nexts_np = nexts_np[:num_imgs,:,:,:]

    # write h5 file to the same location
    dest_filename = os.path.basename(os.path.normpath(dset_folder))
    with h5py.File(os.path.join(dset_folder,"{}_{}.h5".format(dest_filename,num_imgs)), "w") as f:
        f.create_dataset("images", data=images_np, dtype=images_np.dtype)
        f.create_dataset("nexts", data=nexts_np, dtype=nexts_np.dtype)

def prepare_h5_from_images_folder(dset_folder, res):

    images_list = []
    for filename in glob.glob(os.path.join(dset_folder,"*.png")):
        im = Image.open(filename).resize((res,res),Image.BILINEAR)
        im = np.array(im) / 255.0
        images_list.append(im)
    
    images_np = np.array(images_list, dtype=np.float32)
    if len(images_np.shape) == 3:
        images_np = np.expand_dims(images_np, axis=-1)

    # write h5 file to the same location
    dest_filename = os.path.basename(os.path.normpath(dset_folder))
    with h5py.File(os.path.join(dset_folder,"{}.h5".format(dest_filename)), "w") as f:
        f.create_dataset("images", data=images_np, dtype=images_np.dtype)

def get_starting_points(points_path):

    # filter starting points
    points_txt = open(points_path,"r")
    starting_points = []
    for line in points_txt:
        point_vec = [float(line.split(" ")[0][2:]),float(line.split(" ")[1][2:]),float(line.split(" ")[2][2:])] # point vector - (x,y,z)
        orientation_vec = [float(line.split(" ")[3][2:]),float(line.split(" ")[4][2:]),float(line.split(" ")[5][2:])] # orientation vector - (pitch,yaw,roll)
        starting_points.append([point_vec,orientation_vec])

    org_starting_point = starting_points[0]

    # convert starting points to relative
    rel_points = []
    for point in starting_points:

        rel_point = (np.array(point) - np.array(org_starting_point))
        rel_point[0] /= 100.0
        rel_point[1][0] = math.radians(rel_point[1][0])
        rel_point[1][1] = math.radians(rel_point[1][1]) - math.pi/2
        rel_point[1][2] = math.radians(rel_point[1][2])
        rel_points.append(rel_point)
    
    return rel_points

def get_random_navigable_point(log_path):
    
    if log_path != "":
        log_file = open(log_path,"r")

        random_point_str = None
        for line in reversed(list(log_file)):
            if "RandomPoint - " in line.rstrip():
                random_point_str = line.rstrip()
                break
        
        # starting point in maze env
        org_starting_point = [-290.000, 10050.000, 250.000]
        #org_starting_point = [340.000, 840.000, 32.000]
        point_vec = [float(random_point_str.split(" ")[-3].split("=")[-1]),float(random_point_str.split(" ")[-2].split("=")[-1]),float(random_point_str.split(" ")[-1].split("=")[-1])]

        rel_point = (np.array(point_vec) - np.array(org_starting_point))
        rel_point /= 100.0

        orientation = [0.0, np.random.uniform()*2*math.pi, 0.0]

        return rel_point, orientation
    else: #-0.5*math.pi
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

def load_dataset_for_imitation_model(dset_file, num_imgs, batch_size):

    # load h5 file
    dataset_dict = h5py.File(dset_file, 'r')

    # get dataset as numpy
    images_dataset = np.asarray(dataset_dict['images'], dtype=np.float32)
    labels_dataset = np.asarray(dataset_dict['labels'], dtype=np.int)

    # shuffle images and labels in the same order
    p = np.random.permutation(images_dataset.shape[0])
    images_dataset = images_dataset[p]
    labels_dataset = labels_dataset[p]

    # trim data if asked to
    if images_dataset.shape[0] > num_imgs:
        images_dataset = images_dataset[:num_imgs,:,:,:]
        labels_dataset = labels_dataset[:num_imgs,:]

    # convert to tf format dataset and prepare batches
    test_split = int(images_dataset.shape[0] * 0.1)
    train_ds = tf.data.Dataset.from_tensor_slices((images_dataset[:-test_split,:,:,:],labels_dataset[:-test_split,:])).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((images_dataset[-test_split:,:,:,:],labels_dataset[-test_split:,:])).batch(batch_size)

    return train_ds, test_ds

def load_dataset_for_visceral_model(dset_file, batch_size, pred_method, vel):

    # load h5 file
    dataset_dict = h5py.File(dset_file, 'r')

    # get dataset as numpy
    images_dataset = np.asarray(dataset_dict['images'], dtype=np.float32)
    labels_dataset = np.asarray(dataset_dict['labels'], dtype=np.float32)
    if vel:
        velocities_dataset = np.asarray(dataset_dict['velocities'], dtype=np.float32) / 20.0
        velocities_dataset = np.clip(velocities_dataset, -1.0, 1.0)

    if len(images_dataset.shape) < 4:
        images_dataset = np.expand_dims(images_dataset, axis=-1)

    if pred_method == 'pos':
        labels_dataset = labels_dataset[:,:1]
    elif pred_method == 'neg':
        labels_dataset = labels_dataset[:,1:]

    # shuffle images and labels in the same order
    p = np.random.permutation(images_dataset.shape[0])
    images_dataset = images_dataset[p]
    labels_dataset = labels_dataset[p]
    if vel:
        velocities_dataset = velocities_dataset[p]

    # convert to tf format dataset and prepare batches
    test_split = int(images_dataset.shape[0] * 0.1)
    if vel:
        train_ds = tf.data.Dataset.from_tensor_slices((images_dataset[:-test_split,:,:,:],velocities_dataset[:-test_split,:],labels_dataset[:-test_split,:])).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((images_dataset[-test_split:,:,:,:],velocities_dataset[-test_split:,:],labels_dataset[-test_split:,:])).batch(batch_size)
    else:      
        train_ds = tf.data.Dataset.from_tensor_slices((images_dataset[:-test_split,:,:,:],labels_dataset[:-test_split,:])).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((images_dataset[-test_split:,:,:,:],labels_dataset[-test_split:,:])).batch(batch_size)

    return train_ds, test_ds

def load_dataset_for_frame_restoration(dset_file, num_imgs, batch_size, random_sample=True):

    # parser h5 files
    h5_files = dset_file.split(";")
    images_dataset = []
    for h5_file in h5_files:
        if h5_file != "":
            # load h5 file
            h5_dict = h5py.File(h5_file, 'r')
            # get dataset as numpy
            images_dataset.append(np.asarray(h5_dict['images'], dtype=np.float32))

    # combine all numpy arrays into one
    images_dataset = np.concatenate(images_dataset, axis=0)

    # shuffle images and labels in the same order
    if random_sample:
        p = np.random.permutation(images_dataset.shape[0])
        images_dataset = images_dataset[p]

    # trim data if asked to
    if images_dataset.shape[0] > num_imgs:
        images_dataset = images_dataset[:num_imgs,:,:,:]

    # shuffle again
    p = np.random.permutation(images_dataset.shape[0])
    images_dataset = images_dataset[p]

    # convert to tf format dataset and prepare batches
    test_split = int(images_dataset.shape[0] * 0.1)
    train_ds = tf.data.Dataset.from_tensor_slices((images_dataset[:-test_split,:,:,:],images_dataset[:-test_split,:,:,:])).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((images_dataset[-test_split:,:,:,:],images_dataset[-test_split:,:,:,:])).batch(batch_size)

    return train_ds, test_ds

def load_dataset_for_training_decoder(dset_file, num_imgs, batch_size, method):

    # load h5 file
    dataset_dict = h5py.File(dset_file, 'r')

    # get dataset as numpy
    images_dataset = np.asarray(dataset_dict['images'], dtype=np.float32)
    predictions_dataset = np.asarray(dataset_dict[method], dtype=np.float32)
    if method == 'rgbs' and images_dataset.shape[-1] == 1:
        images_dataset = np.repeat(images_dataset, 3, axis=-1)

    # trim data if asked to
    if images_dataset.shape[0] > num_imgs:

        # shuffle images and labels in the same order
        p = np.random.permutation(images_dataset.shape[0])
        images_dataset = images_dataset[p]
        predictions_dataset = predictions_dataset[p]

        images_dataset = images_dataset[:num_imgs,:,:,:]
        predictions_dataset = predictions_dataset[:num_imgs,:,:,:]

    # convert to tf format dataset and prepare batches
    test_split = int(images_dataset.shape[0] * 0.1)

    train_ds = tf.data.Dataset.from_tensor_slices((images_dataset[:-test_split,:,:,:],predictions_dataset[:-test_split,:,:,:])).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((images_dataset[-test_split:,:,:,:],predictions_dataset[-test_split:,:,:,:])).batch(batch_size)

    return train_ds, test_ds

def flow_to_img(flow, normalize=True, flow_mag_max=None):
    """Convert flow to viewable image, using color hue to encode flow vector orientation, and color saturation to
    encode vector length. This is similar to the OpenCV tutorial on dense optical flow, except that they map vector
    length to the value plane of the HSV color model, instead of the saturation plane, as we do here.
    Args:
        flow: optical flow
        normalize: Normalize flow to 0..255
        flow_mag_max: Max flow to map to 255
    Returns:
        img: viewable representation of the dense optical flow in RGB format
        flow_avg: optionally, also return average flow magnitude
    Ref:
        - OpenCV 3.0.0-dev documentation » OpenCV-Python Tutorials » Video Analysis »
        https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    """
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))

    # A couple times, we've gotten NaNs out of the above...
    nans = np.isnan(flow_magnitude)
    if np.any(nans):
        nans = np.where(nans)
        flow_magnitude[nans] = 0.

    # Normalize
    hsv[..., 0] = flow_angle * 180 / np.pi / 2
    if normalize is True:
        if flow_mag_max is None:
            hsv[..., 1] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        else:
            hsv[..., 1] = flow_magnitude * 255 / flow_mag_max
    else:
        hsv[..., 1] = flow_magnitude
    hsv[..., 2] = 255
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return img

def create_dataset(data_dir, batch_size, train_size=0, label_type='image'):

    # get train dataset 
    train_dataset = h5py.File(os.path.join(data_dir, 'train.h5'), 'r')
    train_images_dataset = np.asarray(train_dataset['image']) / 255.0
    train_labels_dataset = np.asarray(train_dataset['label']) / 255.0
    if train_size > 0:
        train_images_dataset = train_images_dataset[:train_size,:,:,:]
        train_labels_dataset = train_labels_dataset[:train_size,:,:,:]

    # get test dataset
    test_dataset = h5py.File(os.path.join(data_dir, 'test.h5'), 'r')
    test_images_dataset = np.asarray(test_dataset['image']) / 255.0
    test_labels_dataset = np.asarray(test_dataset['label']) / 255.0

    # get only first image from the input sequence
    x_train = np.expand_dims(train_images_dataset[:,0,:,:], axis=1)
    x_test = np.expand_dims(test_images_dataset[:,0,:,:], axis=1)
    if label_type == 'image':
        y_train = np.expand_dims(train_images_dataset[:,-1,:,:], axis=1)
        y_test = np.expand_dims(test_images_dataset[:,-1,:,:], axis=1)
    else: # label_type = 'depth'
        y_train = train_labels_dataset
        y_test = test_labels_dataset

    # convert data format
    x_train = x_train.transpose(0, 2, 3, 1) # NCHW => NHWC
    y_train = y_train.transpose(0, 2, 3, 1) # NCHW => NHWC
    x_test = x_test.transpose(0, 2, 3, 1) # NCHW => NHWC
    y_test = y_test.transpose(0, 2, 3, 1) # NCHW => NHWC

    # convert data type to float32
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # convert to tf format dataset and prepare batches
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return train_ds, test_ds

def add_velocity_to_data_file(dset_file):
    
    # open dataset text files
    dataset_txt = open(dset_file,"r")
    dset_file_dir = os.path.dirname(dset_file)
    dset_file_name = "".join(os.path.basename(dset_file).split('.')[:-1])
    dataset_txt_velocity = open(os.path.join(dset_file_dir,dset_file_name+"_vel.txt"),"w")

    for i, line in enumerate(dataset_txt):
        im_dir = os.path.dirname(os.path.dirname(line.split('\t')[0]))
        im_name = os.path.basename(line.split('\t')[0])
        
        # get velocity for image
        airsim_rec = pd.read_csv(os.path.join(im_dir, 'airsim_rec.txt'), sep='\t')
        velocity_val = float(airsim_rec.loc[airsim_rec['ImageFile'] == im_name]['Speed'])

        # write line to the new txt file
        line_with_vel = line.split('\n')[0] + '\t' + str(velocity_val) + '\n'
        dataset_txt_velocity.write(line_with_vel)

        if i % 1000 == 0:
            print("processing {} lines".format(i))
    
    dataset_txt.close()
    dataset_txt_velocity.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dset_file', '-dset_file', help='path to dataset file', default='C:\\Users\\user\\Documents\\Data\\VisceralMachines\\TrainingFiles\\data_mazeonly_frame_2s_window_noclipping2_07_12_19.txt', type=str)
    parser.add_argument('--dset_dir', '-dset_dir', help='path to dataset folder', default='C:\\Users\\user\\Documents\\Data\\cv\\randompose_rgb_segmentation', type=str)
    parser.add_argument('--res', '-res', help='destination resolution for images in the cooked data. if 0, do nothing', default=64, type=int)
    parser.add_argument('--buffer_size', '-buffer_size', help='number of images in one sample', default=4, type=int)
    parser.add_argument('--images_gap', '-images_gap', help='number of images in one sample', default=1, type=int)
    parser.add_argument('--img_type', '-img_type', help='type of image from [rgb, depth], depth and grayscale are the same', default='depth', type=str)
    parser.add_argument('--label_type', '-label_type', help='type of image from [action, depth, segmentation]', default='segmentation', type=str)
    parser.add_argument('--cov_t', '-cov_t', help='coverage threshold. to cope with too short segments', default=100, type=int)
    args = parser.parse_args()

    #prepare_h5_file_for_visceral_model(args.dset_file, args.res, cov_size=20)
    #prepare_h5_file_for_imitation_model(args.dset_dir, args.res, args.buffer_size, args.images_gap)
    #add_velocity_to_data_file(args.dset_file)
    #prepare_h5_for_frame_restoration(args.dset_dir, args.res)
    #prepare_h5_for_frame_restoration_trial_separated(args.dset_dir, args.res, args.cov_t)
    #prepare_h5_for_frame_prediction(args.dset_dir, args.res, args.images_gap)
    #prepare_h5_for_depth_estimation(args.dset_dir, args.res, num_imgs=2000)
    #prepare_h5_for_semantic_segmentation(args.dset_dir, args.res, num_imgs=2000)
    prepare_h5_for_rgb_creation(args.dset_dir, args.res, num_imgs=2000)
    #prepare_h5_for_flow_estimation(args.dset_dir, args.res, num_imgs=2000)
    #prepare_h5_for_frame_prediction(args.dset_dir, args.res, num_imgs=2000)
    #prepare_h5_from_images_folder(args.dset_dir, args.res)