import sys
sys.path.append('.')
import airsim
import os
import numpy as np
import argparse
import tensorflow as tf
from model import ILModel
from PIL import Image
import cv2
import keyboard

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-path', help='model file path', default='C:\\Users\\user\\Documents\\models\\imitation_4images\\model40.ckpt', type=str)
parser.add_argument('--debug', '-debug', dest='debug', help='debug mode, present front camera on screen', action='store_true')
args = parser.parse_args()

# tf functions
@tf.function
def predict_action(image):
    return model(image)

if __name__ == "__main__":

    # allow growth is possible using an env var in tf2.0
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # Load imitation model
    model = ILModel()
    model.load_weights(args.path)

    # connect to the AirSim simulator 
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()

    # actions list
    actions = [-1.0, -0.5, 0.0, 0.5, 1.0]
    prev_action = 0.0
    x_pt = 0.0

    # initiate buffer
    buffer = np.zeros((4, 84, 84), dtype=np.float32)

    # let the car start driving
    car_controls.throttle = 0.5
    car_controls.steering = 0
    client.setCarControls(car_controls)

    print('Running model')

    Image_requests = [airsim.ImageRequest("RCCamera", airsim.ImageType.Scene, False, False)]

    while(True):

        # get images from AirSim
        responses = client.simGetImages(Image_requests)
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        if img1d.size > 1:
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))
            image = Image.fromarray(img2d).convert('L')
            image_np = np.array(image.resize((84, 84)))
        else:
            image_np = np.zeros((84,84)).astype(float)

        # present state image if debug mode is on
        if args.debug:
            cv2.imshow('navigation map', image_np)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # append to buffer
        image_np = image_np / 255.0
        buffer[:-1] = buffer[1:]
        buffer[-1] = image_np
        
        # convert input to [1,84,84,4]
        input_img = np.expand_dims(buffer, axis=0).transpose(0, 2, 3, 1).astype(np.float32)

        # predict next action based on model decision
        action_values = predict_action(input_img)
        car_controls.steering = actions[np.argmax(action_values)]
        
        # send action to AirSim
        client.setCarControls(car_controls)

        print('steering = {}, actions_p = {}'.format(car_controls.steering, np.around(action_values,decimals=3)[0]))
            
        if keyboard.is_pressed('q'):  # if key 'q' is pressed
            if args.store != "":
                airsim_rec.close()
            sys.exit()