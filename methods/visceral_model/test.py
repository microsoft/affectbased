# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
sys.path.append('.')
import airsim
import os
import numpy as np
import argparse
import tensorflow as tf
from model import VisceralModel
from utils.coverage_map import CoverageMap
from PIL import Image
import cv2
import time
import keyboard

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', '-model_path', help='model file path', default='C:\\Users\\user\\Documents\\models\\visceral_models\\visceral_model_84_reg_both_norm_2s_noclipping2_newdata\\vismodel40.ckpt', type=str)
parser.add_argument('--debug', '-debug', dest='debug', help='debug mode, present camera on screen', action='store_true')
parser.add_argument('--res', '-res', help='destination resolution for images in the cooked data. if 0, do nothing', default=84, type=int)
parser.add_argument('--mean_size', '-mean_size', help='number of values for mean', default=20, type=int)
args = parser.parse_args()

# tf function to test
@tf.function
def predict_image(image):
    return model(image)

if __name__ == "__main__":

    # allow growth is possible using an env var in tf2.0
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # Load model
    model = VisceralModel(num_outputs=2)
    model.load_weights(args.model_path)

    # connect to the AirSim simulator 
    client = airsim.CarClient()
    client.confirmConnection()
    car_controls = airsim.CarControls()
    
    pos_list, neg_list = [0]*args.mean_size, [0]*args.mean_size
    Image_requests = [airsim.ImageRequest("VisceralCamera_m", airsim.ImageType.Scene, False, False)]
    print('Running model')

    while(True):

        start_time = time.time()

        # get and filter input image from airsim
        responses = client.simGetImages(Image_requests)
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        if img1d.size > 1:
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))
            image = Image.fromarray(img2d)
            image_np = np.array(image.resize((args.res, args.res)))
            image_np = image_np[:,:,:3]
        else:
            image_np = np.zeros((args.res,args.res, 3)).astype(float)

        # present state image if debug mode is on
        if args.debug:
            cv2.imshow('Camera', image_np)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # convert input to [1,res,res,3]
        image_np = image_np / 255.0
        input_img = np.expand_dims(image_np, axis=0).astype(np.float32)

        # predict signal
        prediction = predict_image(input_img)

        # compute predicted x and dx
        pos_t = prediction.numpy()[0][0]
        neg_t = prediction.numpy()[0][1]
        end_time = time.time()

        pos_list.append(pos_t)
        neg_list.append(neg_t)
        del pos_list[0]
        del neg_list[0]

        # show prediction
        print("pos: {}, neg: {}, pos mean: {}, neg mean: {}, fps: {}".format(str(pos_t)[:5], 
                                                                             str(neg_t)[:5], 
                                                                             str(np.array(pos_list).mean())[:5], 
                                                                             str(np.array(neg_list).mean())[:5], 
                                                                             str(1/(end_time-start_time))[:5]))

        if keyboard.is_pressed('q'):  # if key 'q' is pressed
            sys.exit()