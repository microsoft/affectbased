# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.append('.')
import os
import glob
import numpy as np
import cv2
import h5py
import scipy
import argparse
import tensorflow as tf
from methods.vae.model import VAEModel
from PIL import Image
import shutil
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', '-model_path', help='path to model', default='C:\\Users\\user\\Documents\\experiments\\generating_samples\\models\\depth_est\\il\\00\\10\\vaemodel40.ckpt', type=str)
parser.add_argument('--images_path', '-images_path', help='image file path', default='C:\\Users\\user\\Documents\\experiments\\generating_samples\\inputs', type=str)
parser.add_argument('--log_path', '-log_path', help='path for the log file', default='C:\\Users\\user\\Documents\\experiments\\generating_samples\\outputs\\log.txt', type=str)
parser.add_argument('--task', '-task', help='the task to train on. choose from [frame_res,depth,segmentation,con_rgb]', default='con_rgb', type=str)
parser.add_argument('--n_z', '-n_z', help='size of the each one of the parameters [mean,stddev] in the latent space', default=8, type=int)
parser.add_argument('--vae_res', '-vae_res', help='destination resolution for images for the vae', default=64, type=int)
parser.add_argument('--v3_res', '-v3_res', help='destination resolution for images for the inception model', default=299, type=int)
args = parser.parse_args()

# tf functions
@tf.function
def predict_image(images):
    imgs, _, _, _ = model(images)
    return imgs

@tf.function
def get_features(preds, gts):
    preds_features = inception_model(preds)
    gts_features = inception_model(gts)
    return preds_features, gts_features

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Load model
if args.task == 'depth':
    model = VAEModel(n_z=args.n_z, out_channels=1)
else: # segmentation or con_rgb
    model = VAEModel(n_z=args.n_z, out_channels=3)
model.load_weights(args.model_path)

# Load inception model
inception_model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', weights='imagenet')

# Load images
images_np_list, gt_np_list = [], []
for filename in glob.glob(os.path.join(args.images_path,"*.png")):
    # Load input image
    im = Image.open(filename).resize((args.vae_res,args.vae_res),Image.BILINEAR)
    im_np = np.array(im) / 255.0
    images_np_list.append(im_np)
    # Load GT image
    if args.task == 'depth':
        im = Image.open(filename.replace("\\im_","\\depth_").replace("/im_","/depth_").replace("\\inputs\\","\\gts\\").replace("/inputs/","/gts/")).resize((args.v3_res,args.v3_res),Image.BILINEAR)
    elif args.task == 'segmentation':
        im = Image.open(filename.replace("\\im_","\\seg_").replace("/im_","/seg_").replace("\\inputs\\","\\gts\\").replace("/inputs/","/gts/")).resize((args.v3_res,args.v3_res),Image.BILINEAR)
    elif args.task == 'con_rgb':
        im = Image.open(filename.replace("\\cont","\\im_").replace("/cont_","/im_").replace("\\inputs\\","\\gts\\").replace("/inputs/","/gts/")).resize((args.v3_res,args.v3_res),Image.BILINEAR)        
    else: # args.task == 'frame_res'
        im = Image.open(filename).resize((args.v3_res,args.v3_res),Image.BILINEAR)
    im_np = np.array(im)
    gt_np_list.append(im_np)

images_np = np.array(images_np_list, dtype=np.float32)
gt_np = np.array(gt_np_list, dtype=np.float32)
if len(images_np.shape) == 3:
    images_np = np.expand_dims(images_np, axis=-1)
    if args.task == 'con_rgb':
        images_np = np.repeat(images_np, 3, axis=-1)
if len(gt_np.shape) == 3:
    gt_np = np.expand_dims(gt_np, axis=-1)

# get predicted outcomes
preds = predict_image(images_np)
preds_np = preds.numpy() * 255.0

# calculate frechet inception distance (FID)
if preds_np.shape[-1] == 1:
    preds_np = np.repeat(preds_np, 3, axis=-1)
if gt_np.shape[-1] == 1:
    gt_np = np.repeat(gt_np, 3, axis=-1)

# get inception features for both groups
# the loop is because the GPU can't handle all of them at once
preds_features = np.zeros((preds_np.shape[0], 2048))
gts_features = np.zeros((preds_np.shape[0], 2048))
for i in range(preds_np.shape[0]):
    pred_np_i = np.expand_dims(cv2.resize(preds_np[i], (args.v3_res,args.v3_res)), axis=0)
    gt_np_i = np.expand_dims(gt_np[i], axis=0)

    pred_np_i = tf.keras.applications.inception_v3.preprocess_input(pred_np_i)
    gt_np_i = tf.keras.applications.inception_v3.preprocess_input(gt_np_i)

    preds_features_i, gts_features_i = get_features(pred_np_i, gt_np_i)
    
    preds_features[i] = preds_features_i[0]
    gts_features[i] = gts_features_i[0]
    
    if i%200 == 0:
        print('{}/{} features done'.format(i,preds_np.shape[0]))

# calculate mean and covariance statistics
mu1, sigma1 = preds_features.mean(axis=0), np.cov(preds_features, rowvar=False)
mu2, sigma2 = gts_features.mean(axis=0), np.cov(gts_features, rowvar=False)

# calculate sum squared difference between means
ssdiff = np.sum((mu1 - mu2)**2.0)

# calculate sqrt of product between cov
covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))

# check and correct imaginary numbers from sqrt
if np.iscomplexobj(covmean):
    covmean = covmean.real

# calculate and print score
fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

# write to txt file
log_file = open(args.log_path,"w") 
log_file.write("{}".format(fid))
log_file.close()
#print('FID is: {}'.format(fid))
