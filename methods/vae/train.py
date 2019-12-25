from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.append('.')
import os
import numpy as np
import h5py
import argparse
import tensorflow as tf
from model import VAEModel
from utils import import_data

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', '-data_file', help='path to raw data folder, use ; for multiple data files', default='C:\\Users\\user\\Documents\\experiments\\data_collection_recordings\\il_curious_6.0\\il_curious_6.0.h5', type=str)
parser.add_argument('--output_dir', '-output_dir', help='path to output folder', default='C:\\Users\\user\\Documents\\models\\test_residual_01', type=str)
parser.add_argument('--batch_size', '-batch_size', help='number of samples in one minibatch', default=32, type=int)
parser.add_argument('--num_imgs', '-num_imgs', help='number of images to train on', default=50000, type=int)
parser.add_argument('--random_sample', '-random_sample', dest='random_sample', help='if to randomly sample num_imgs from batch', action='store_true')
parser.add_argument('--epochs', '-epochs', help='number of epochs to train the model', default=40, type=int)
parser.add_argument('--cp_interval', '-cp_interval', help='interval for checkpoint saving', default=20, type=int)
parser.add_argument('--n_z', '-n_z', help='size of the each one of the parameters [mean,stddev] in the latent space', default=8, type=int)
parser.add_argument('--res', '-res', help='destination resolution for images in the cooked data. if 0, do nothing', default=64, type=int)
parser.add_argument('--grayscale', '-grayscale', dest='grayscale', help='choose for training on grayscale images', action='store_true')
parser.add_argument('--gpu', '-gpu', help='gpu number to train on', default='0', type=str)
args = parser.parse_args()

@tf.function
def compute_loss(y, y_pred, means, stddev):

    # copute reconstruction loss
    recon_loss = tf.reduce_mean(tf.keras.losses.MSE(y, y_pred))

    # copute KL loss: D_KL(Q(z|X,y) || P(z|X))
    kl_loss = -0.5*tf.reduce_mean(tf.reduce_sum((1+stddev-tf.math.pow(means, 2)-tf.math.exp(stddev)), axis=1))

    return recon_loss, kl_loss

# tf function to train
@tf.function
def train(images, labels):
    with tf.GradientTape() as tape:
        predictions, means, stddev, _ = model(images)
        recon_loss, kl_loss = compute_loss(labels, predictions, means, stddev)
        loss = recon_loss + kl_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_rec_loss(recon_loss)
    train_kl_loss(kl_loss)

# tf function to test
@tf.function
def test(images, labels):
    predictions, means, stddev, _ = model(images)
    recon_loss, kl_loss = compute_loss(labels, predictions, means, stddev)

    test_rec_loss(recon_loss)
    test_kl_loss(kl_loss)

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

# get train and test datasets
train_ds, test_ds = import_data.load_dataset_for_frame_restoration(args.data_file, args.num_imgs, args.batch_size, random_sample=args.random_sample)

# create model, loss and optimizer
model = VAEModel(n_z=args.n_z, out_channels=3)
optimizer = tf.keras.optimizers.Adam()

# define metrics
train_rec_loss = tf.keras.metrics.Mean(name='train_rec_loss')
train_kl_loss = tf.keras.metrics.Mean(name='train_kl_loss')
test_rec_loss = tf.keras.metrics.Mean(name='test_rec_loss')
test_kl_loss = tf.keras.metrics.Mean(name='test_kl_loss')
metrics_writer = tf.summary.create_file_writer(args.output_dir)

# check if output folder exists
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

# train
print('Start training...')
for epoch in range(args.epochs):

    for images, labels in train_ds:
        train(images, labels)

    for test_images, test_labels in test_ds:
        test(test_images, test_labels)
    
    with metrics_writer.as_default():
        tf.summary.scalar('Train reconstruction loss', train_rec_loss.result(), step=epoch)
        tf.summary.scalar('Train KL loss', train_kl_loss.result(), step=epoch)
        tf.summary.scalar('Test reconstruction loss', test_rec_loss.result(), step=epoch)
        tf.summary.scalar('Test KL loss', test_kl_loss.result(), step=epoch)

    # save model
    if (epoch+1) % args.cp_interval == 0 and epoch > 0:
        print('Saving weights to {}'.format(args.output_dir))
        model.save_weights(os.path.join(args.output_dir, "vaemodel{}.ckpt".format(epoch+1)))
    
    print('Epoch {}, Loss: {}, KL-Loss: {}, Test Loss: {}, Test KL-Loss: {}'.format(epoch+1, train_rec_loss.result(), train_kl_loss.result(), test_rec_loss.result(), test_kl_loss.result()))