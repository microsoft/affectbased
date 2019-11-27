from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.append('.')
import os
import glob
import numpy as np
import cv2
import h5py
import argparse
import tensorflow as tf
from methods.vae.model import VAEModel
from PIL import Image
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--il_path', '-il_path', help='path to imitation learning model', default='C:\\Users\\user\\Documents\\models\\test_frozen\\vaemodel40.ckpt', type=str)
parser.add_argument('--ilcur_path', '-ilcur_path', help='path to curiosity driven model', default='C:\\Users\\user\\Documents\\models\\test\\vaemodel40.ckpt', type=str)
parser.add_argument('--images_path', '-images_path', help='image file path', default='C:\\Users\\user\\Documents\\experiments\\generating_samples\\con_rgb\\inputs', type=str)
parser.add_argument('--task', '-task', help='the task to train on. choose from [depth,segmentation,con_rgb]', default='con_rgb', type=str)
parser.add_argument('--n_z', '-n_z', help='size of the each one of the parameters [mean,stddev] in the latent space', default=8, type=int)
parser.add_argument('--res', '-res', help='destination resolution for images in the cooked data. if 0, do nothing', default=64, type=int)
parser.add_argument('--output_dir', '-output_dir', help='path to output directory', default='C:\\Users\\user\\Documents\\experiments\\generating_samples\\con_rgb\\samples_frozen', type=str)
args = parser.parse_args()

# tf function for prediction
@tf.function
def predict_image(image):
    il_img, _, _, _ = il_model(image)
    ilcur_img, _, _, _ = ilcur_model(image)
    return il_img, ilcur_img

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# check if output folder exists
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

# Load models
if args.task == 'depth':
    il_model = VAEModel(n_z=args.n_z, out_channels=1)
    ilcur_model = VAEModel(n_z=args.n_z, out_channels=1)
else: # segmentation or con_rgb
    il_model = VAEModel(n_z=args.n_z, out_channels=3)
    ilcur_model = VAEModel(n_z=args.n_z, out_channels=3)

il_model.load_weights(args.il_path)
ilcur_model.load_weights(args.ilcur_path)

# Load images
images_np_list, input_paths, gt_paths = [], [], []
for filename in glob.glob(os.path.join(args.images_path,"*.png")):
    # open input image
    im = Image.open(filename).resize((args.res,args.res),Image.BILINEAR)
    im_np = np.array(im) / 255.0
    images_np_list.append(im_np)
    input_paths.append(filename)
    if args.task == 'depth':
        gt_paths.append(filename.replace("\\im_","\\depth_").replace("/im_","/depth_").replace("\\inputs\\","\\gts\\").replace("/inputs/","/gts/"))
    elif args.task == 'segmentation':
        gt_paths.append(filename.replace("\\im_","\\seg_").replace("/im_","/seg_").replace("\\inputs\\","\\gts\\").replace("/inputs/","/gts/"))
    else: # args.task == 'con_rgb'
        gt_paths.append(filename.replace("\\cont_","\\im_").replace("/cont_","/im_").replace("\\inputs\\","\\gts\\").replace("/inputs/","/gts/"))
        
images_np = np.array(images_np_list, dtype=np.float32)
if len(images_np.shape) == 3:
    images_np = np.expand_dims(images_np, axis=-1)
if images_np.shape[-1] == 1:
    images_np = np.repeat(images_np, 3, axis=-1)

# inference
il_preds, ilcur_preds = predict_image(images_np)

# save predictions to output folder with input and ground truth
il_preds_np = il_preds.numpy()
ilcur_preds_np = ilcur_preds.numpy()
if il_preds_np.shape[-1] == 1:
    il_preds_np = il_preds_np.squeeze(axis=-1)
    ilcur_preds_np = ilcur_preds_np.squeeze(axis=-1)
    
for i in range(images_np.shape[0]):

    # save input image
    input_im = cv2.resize(cv2.imread(input_paths[i]), (args.res,args.res))
    cv2.imwrite(os.path.join(args.output_dir, "{0:04d}_input.png".format(i)), input_im)

    # save ground truth
    gt_im = cv2.resize(cv2.imread(gt_paths[i]), (args.res,args.res))
    cv2.imwrite(os.path.join(args.output_dir, "{0:04d}_gt.png".format(i)), gt_im)

    # save predictions images        
    Image.fromarray(np.uint8(il_preds_np[i]*255)).save(os.path.join(args.output_dir, "{0:04d}_il.png".format(i)))
    Image.fromarray(np.uint8(ilcur_preds_np[i]*255)).save(os.path.join(args.output_dir, "{0:04d}_ilcur.png".format(i)))

    if i % 200 == 0:
        print('{}/{} images done'.format(i, images_np.shape[0]))

