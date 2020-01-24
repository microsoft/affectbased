# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.append('.')
import os
import numpy as np
import h5py
import argparse
import tensorflow as tf
from utils import import_data
from model import VisceralModel

parser = argparse.ArgumentParser()
parser.add_argument('--dset_file', '-dset_file', help='path to raw data folder', default='C:\\Users\\user\\Documents\\Data\\VisceralMachines\\TrainingFiles\\data_mazeonly_frame_2s_window_noclipping2_07_12_19_cov.h5', type=str)
parser.add_argument('--output_dir', '-output_dir', help='path to output folder', default='C:\\Users\\user\\Documents\\models\\visceral_models\\visceral_model_84_reg_both_norm_2s_noclipping2_cov', type=str)
parser.add_argument('--pred', '-pred', help='prediction method. choose from [pos, neg, both]', default='both', type=str)
parser.add_argument('--batch_size', '-batch_size', help='number of samples in one minibatch', default=32, type=int)
parser.add_argument('--epochs', '-epochs', help='number of epochs to train the model', default=40, type=int)
parser.add_argument('--res', '-res', help='destination resolution for images in the cooked data', default=84, type=int)
parser.add_argument('--gpu', '-gpu', help='gpu number to train on', default='0', type=str)
args = parser.parse_args()

# tf functions
@tf.function
def train(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = tf.reduce_mean(tf.keras.losses.MSE(labels, predictions))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)

@tf.function
def test(images, labels):
    predictions = model(images)
    loss = tf.reduce_mean(tf.keras.losses.MSE(labels, predictions))

    test_loss(loss)

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

# get train and test datasets
train_ds, test_ds = import_data.load_dataset_for_visceral_model(args.dset_file, args.batch_size, args.pred)

# create model, loss and optimizer
if args.pred == 'both':
    num_outputs = 2
else:
    num_outputs = 1
model = VisceralModel(num_outputs=num_outputs)
optimizer = tf.keras.optimizers.Adam()

# define metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')
metrics_writer = tf.summary.create_file_writer(args.output_dir)

# check if output folder exists
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

# train
train_counter = 0
test_counter = 0
print('Start training...')
for epoch in range(args.epochs):
      
    for images, labels in train_ds:
        train(images, labels)
        train_counter += 1
        with metrics_writer.as_default():
            tf.summary.scalar('Train loss', train_loss.result(), step=train_counter)
    
    for test_images, test_labels in test_ds:
        test(test_images, test_labels)
        test_counter += 1
        with metrics_writer.as_default():
            tf.summary.scalar('Test loss', test_loss.result(), step=test_counter)

    # save model
    if (epoch+1) % 5 == 0 and epoch > 0:
        print('Saving weights to {}'.format(args.output_dir))
        model.save_weights(os.path.join(args.output_dir, "vismodel{}.ckpt".format(epoch+1)))
    
    print('Epoch {}, Loss: {}, Test Loss: {}'.format(epoch+1, train_loss.result(), test_loss.result()))