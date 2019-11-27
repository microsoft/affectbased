from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.append('.')
import os
import numpy as np
import h5py
import argparse
from model import ILModel
from utils import import_data
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', '-model_path', help='model file path if fine tune an existing model', default='C:\\Users\\user\\Documents\\models\\imitation_4images_nocov\\model40.ckpt', type=str)
parser.add_argument('--data_file', '-data_file', help='path to raw data folder', default='C:\\Users\\user\\Documents\\Data\\gs_4images_nocov_b99\\gs_4images_nocov_b99.h5', type=str)
parser.add_argument('--output_dir', '-output_dir', help='path to output folder', default='C:\\Users\\user\\Documents\\models\\imitation_4images_nocov_b99_finetuned', type=str)
parser.add_argument('--num_imgs', '-num_imgs', help='number of images to train on', default=50000, type=int)
parser.add_argument('--num_actions', '-num_actions', help='number of actions for the model to perdict', default=5, type=int)
parser.add_argument('--batch_size', '-batch_size', help='number of samples in one minibatch', default=32, type=int)
parser.add_argument('--epochs', '-epochs', help='number of epochs to train the model', default=11, type=int)
args = parser.parse_args()

# tf function to train
@tf.function
def train(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

# tf function to test
@tf.function
def test(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

if __name__ == "__main__":

    # allow growth is possible using an env var in tf2.0
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # load train and test datasets
    train_ds, test_ds = import_data.load_dataset_for_imitation_model(args.data_file, args.num_imgs, args.batch_size)

    # create model, loss and optimizer
    model = ILModel()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    # load model if asked
    if args.model_path != "":
        model.load_weights(args.model_path)

    # define metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
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
                tf.summary.scalar('Train accuracy', train_accuracy.result()*100, step=train_counter)

        for test_images, test_labels in test_ds:
            test(test_images, test_labels)
            test_counter += 1
            with metrics_writer.as_default():
                tf.summary.scalar('Test loss', test_loss.result(), step=test_counter)
                tf.summary.scalar('Test accuracy', test_accuracy.result()*100, step=test_counter)
        
        # save model
        if epoch % 5 == 0 and epoch > 0:
            print('Saving weights to {}'.format(args.output_dir))
            model.save_weights(os.path.join(args.output_dir, "model{}.ckpt".format(epoch)))
        
        print('Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'.format(epoch+1, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))