from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.append('.')
import os
import numpy as np
import h5py
import argparse
import tensorflow as tf
from model import VAEModel
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-path', help='model file path', default='C:\\Users\\user\\Documents\\models\\test_mse_clear\\vaemodel40.ckpt', type=str)
parser.add_argument('--n_z', '-n_z', help='size of the each one of the parameters [mean,stddev] in the latent space', default=8, type=int)
parser.add_argument('--img', '-img', help='image file path', default='C:\\Users\\user\\Documents\\Data\\cv\\randompose_rgb_segmentation\\recordings_01\\images\\cont_15715113413696584.png', type=str)
parser.add_argument('--res', '-res', help='destination resolution for images in the cooked data. if 0, do nothing', default=64, type=int)
parser.add_argument('--show', '-show', help='choose what to do from [predict, inter]', default='predict', type=str)
parser.add_argument('--method', '-method', help='choose what to do from [restoration, depth, con2rgb]', default='restoration', type=str)
parser.add_argument('--grayscale', '-grayscale', dest='grayscale', help='choose for training on grayscale images', action='store_true')
args = parser.parse_args()


# tf function for prediction
@tf.function
def predict_image(image, inter=None):
    return model(image, inter)

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Load the model
model = VAEModel(n_z=args.n_z)
model.load_weights(args.path)

# Load image
im = Image.open(args.img)
if args.res > 0:
    im = im.resize((args.res, args.res), Image.ANTIALIAS)
input_img = (np.expand_dims(np.array(im), axis=0) / 255.0).astype(np.float32)
if len(input_img.shape) < 4:
    input_img = np.expand_dims(np.array(input_img), axis=-1)
    input_img = np.repeat(input_img, 3, axis=-1)

if args.show == 'inter': # interpolate over the latent space

    latent_array = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])

    if not args.grayscale:
        output_im = np.zeros((args.res*latent_array.shape[0],args.res*args.n_z,3))
    else:
        output_im = np.zeros((args.res*latent_array.shape[0],args.res*args.n_z))

    for i in range(args.n_z):

        for j in range(latent_array.shape[0]):
            # predict next image
            latent_vector = np.zeros((1, args.n_z)).astype(np.float32)
            latent_vector[0][i] = latent_array[j]
            prediction, _, _, _ = predict_image(input_img, inter=latent_vector)

            # add predicted image to sequence
            predicted_image = prediction.numpy().squeeze(axis=0)*255
            if args.grayscale:
                predicted_image = predicted_image.squeeze(axis=-1)

            if not args.grayscale:
                output_im[j*args.res:(j+1)*args.res,i*args.res:(i+1)*args.res,:] = predicted_image
            else:
                output_im[j*args.res:(j+1)*args.res,i*args.res:(i+1)*args.res] = predicted_image

    # present sequence of images
    predicted_image = Image.fromarray(np.uint8(output_im))
    predicted_image.show()

else:

    # predict next image
    prediction, _, _, z = predict_image(input_img)

    # present mean values
    print("z values:")
    print(z.numpy())

    # present predicted image
    predicted_image = prediction.numpy().squeeze(axis=0)
    if args.method == 'depth':
        predicted_image = predicted_image.squeeze(axis=-1)
        predicted_image = Image.fromarray(np.uint8(predicted_image*255))
    else: # args.method == 'restoration or con2rgb'
        predicted_image = Image.fromarray(np.uint8(predicted_image*255))
    predicted_image.show()
