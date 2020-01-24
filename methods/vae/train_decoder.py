# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.append('.')
import os
import h5py
import argparse
import tensorflow as tf
from PIL import Image
from model import VAEModel
from utils import import_data

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', '-model_path', help='model file path', default='C:\\Users\\user\\Documents\\experiments\\data_collection_recordings\\frame_res\\il_curious_6.0\\00\\49\\vaemodel40.ckpt', type=str)
parser.add_argument('--dset_file', '-dset_file', help='path to dataset file', default='C:\\Users\\user\\Documents\\Data\\cv\\randompose_rgb_segmentation\\randompose_rgb_segmentation_contours_2000.h5', type=str)
parser.add_argument('--output_dir', '-output_dir', help='path to output folder', default='C:\\Users\\user\\Documents\\models\\test', type=str)
parser.add_argument('--task', '-task', help='the task to train on. choose from [depth,segmentation,con2rgb]', default='depth', type=str)
parser.add_argument('--cp_interval', '-cp_interval', help='interval for checkpoint saving', default=20, type=int)
parser.add_argument('--num_imgs', '-num_imgs', help='number of samples in the dataset', default=50000, type=int)
parser.add_argument('--batch_size', '-batch_size', help='number of samples in one minibatch', default=32, type=int)
parser.add_argument('--epochs', '-epochs', help='number of epochs to train the model', default=40, type=int)
parser.add_argument('--n_z', '-n_z', help='size of the each one of the parameters [mean,stddev] in the latent space', default=8, type=int)
parser.add_argument('--trainable_layers', '-trainable_layers', help='number of trainable decoding layers', default=5, type=int)
parser.add_argument('--res', '-res', help='destination resolution for images in the cooked data. if 0, do nothing', default=64, type=int)
parser.add_argument('--gpu', '-gpu', help='gpu number to train on', default='0', type=str)
args = parser.parse_args()

# tf function to train
@tf.function
def train_generative(images, labels):
    with tf.GradientTape() as tape:
        predictions, means, stddev, _ = model(images)
        loss = tf.reduce_mean(tf.keras.losses.MSE(labels, predictions))
        loss_z = -0.5*tf.reduce_mean(tf.reduce_sum((1+stddev-tf.math.pow(means, 2)-tf.math.exp(stddev)), axis=1))

        total_loss = loss + loss_z
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)                                                                                        

@tf.function
def train(images, labels):
    with tf.GradientTape() as tape:
        predictions, _, _, _ = model(images)
        loss = tf.reduce_mean(tf.keras.losses.MSE(labels, predictions))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)     

# tf function to test
@tf.function
def test(images, labels):
    predictions, _, _, _ = model(images)
    loss = tf.reduce_mean(tf.keras.losses.MSE(labels, predictions))

    test_loss(loss)

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

# check if output folder exists
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

# get train and test datasets
if args.task == 'depth':
    train_ds, test_ds = import_data.load_dataset_for_training_decoder(args.dset_file, args.num_imgs, args.batch_size, 'depths')
elif args.task == 'segmentation':
    train_ds, test_ds = import_data.load_dataset_for_training_decoder(args.dset_file, args.num_imgs, args.batch_size, 'segs')
else: #args.task == 'con2rgb'
    train_ds, test_ds = import_data.load_dataset_for_training_decoder(args.dset_file, args.num_imgs, args.batch_size, 'rgbs')

# prepare boolean vars for decoding layers
trainable_layers = [True]*5
frozen_layers = 5 - args.trainable_layers
for i in range(frozen_layers):
    trainable_layers[i] = False

# create model, loss and optimizer
if args.task == 'depth':
    model = VAEModel(n_z=args.n_z, trainable_encoder=False, trainable_decoder=trainable_layers, out_channels=1)
    model.load_weights(args.model_path)
elif args.task == 'segmentation':
    model = VAEModel(n_z=args.n_z, trainable_encoder=False, trainable_decoder=trainable_layers, out_channels=3)
    model.load_weights(args.model_path)
else: # args.task == 'con2rgb'
    model = VAEModel(n_z=args.n_z, trainable_encoder=True, trainable_decoder=trainable_layers, out_channels=3)
    model.load_decoder_weights_only(args.model_path)
optimizer = tf.keras.optimizers.Adam()

# create text log file
log_file = open(os.path.join(args.output_dir,"log.txt"),"w") 
log_file.write("Epoch\tTrainLoss\tTestLoss\n")

# define metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')
metrics_writer = tf.summary.create_file_writer(args.output_dir)

print('Start training...')
for epoch in range(args.epochs):

    if args.task == 'con2rgb':  
        for images, labels in train_ds:
            train_generative(images, labels)
    else:
        for images, labels in train_ds:
            train(images, labels)

    for test_images, test_labels in test_ds:
        test(test_images, test_labels)

    # write to tensorboard log file
    with metrics_writer.as_default():
        tf.summary.scalar('Train loss', train_loss.result(), step=epoch)
        tf.summary.scalar('Test loss', test_loss.result(), step=epoch)

    # write to txt log file
    log_file.write("{}\t{}\t{}\n".format(epoch+1, train_loss.result(), test_loss.result()))

    # save model
    if (epoch+1) % args.cp_interval == 0 and epoch > 0:
        print('Saving weights to {}'.format(args.output_dir))
        model.save_weights(os.path.join(args.output_dir, "vaemodel{}.ckpt".format(epoch+1)))
    
    print('Epoch {}, Loss: {}, Test Loss: {}'.format(epoch+1, train_loss.result(), test_loss.result()))

log_file.close()