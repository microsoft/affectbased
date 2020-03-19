# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
sys.path.append('.')
import airsim
import os
import argparse
import numpy as np
import time
import datetime
import cv2
from utils.coverage_map import CoverageMap
from utils.import_data import get_random_navigable_point
from methods.imitation_learning.model import ILModel
from methods.visceral_model.model import VisceralModel
import tensorflow as tf
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--il_path', '-il_path', help='imitation learning model file path', default='C:\\Users\\user\\Documents\\Papers\\Affect-based\\Models\\imitation_4images_nocov\\model40.ckpt', type=str)
parser.add_argument('--vis_path', '-vis_path', help='visceral model file path. wont load if empty', default='C:\\Users\\user\\Documents\\Papers\\Affect-based\\Models\\visceral_models\\visceral_model_84_reg_both_norm_2s_noclipping2_newdata\\vismodel40.ckpt', type=str)
parser.add_argument('--output_dir', '-output_dir', help='output folder to put the text file in', default='C:\\Users\\user\\Documents\\Papers\\Affect-based\\Experiments\\il_03', type=str)
parser.add_argument('--res', '-res', help='destination resolution for images to be stored', default=84, type=int)
parser.add_argument('--duration', '-duration', help='driving duration for the experiment', default=2000, type=int)
parser.add_argument('--store', '-store', help='type of images to store. choose from [none, rgb]', default='rgb', type=str)
parser.add_argument('--alpha', '-alpha', help='multiplication factor for the imitation model probabilities', default=1.0, type=float)
parser.add_argument('--gamma', '-gamma', help='multiplication factor for adding positive value', default=0.0, type=float)
parser.add_argument('--beta', '-beta', help='multiplication factor for adding negative value', default=0.0, type=float)
parser.add_argument('--delta', '-delta', help='multiplication factor for adding random value', default=0.0, type=float)
parser.add_argument('--epsilon', '-epsilon', help='multiplication factor for adding straight value', default=0.0, type=float)
parser.add_argument('--log_path', '-log_path', help='path to simulation log file', default='C:\\Users\\user\\Documents\\Unreal Projects\\Maze\\Saved\\Logs\\Car_Maze.log', type=str)
#parser.add_argument('--log_path', '-log_path', help='path to simulation log file', default='', type=str)
parser.add_argument('--debug', '-debug', dest='debug', help='debug mode, present fron camera on screen', action='store_true')
args = parser.parse_args()

# tf functions
@tf.function
def predict_vis_values_and_il(image_l, image_lm, image_m, image_rm, image_r, image_il):
    return vismodel(image_l), vismodel(image_lm), vismodel(image_m), vismodel(image_rm), vismodel(image_r), model(image_il)

# convert airsim response into a required numpy image
def get_image(response, res, grayscale=False):

    image_l_np = None
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
    if img1d.size > 1:
        img2d = np.reshape(img1d, (response.height, response.width, 3))
        img2d = img2d[:,:,:3]
        image = Image.fromarray(img2d)
        image_np = np.array(image.resize((res, res)))
        if grayscale:
            image_l = image.convert('L')
            image_l_np = np.array(image_l.resize((res, res)))
    else:
        image_np = np.zeros((res,res,3)).astype(float)
        if grayscale:
            image_l_np = np.zeros((res,res)).astype(float)

    return image_np, image_l_np


if __name__ == "__main__":

    # allow growth is possible using an env var in tf2.0
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # check if output folder exists
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Load the model
    model = ILModel()
    model.load_weights(args.il_path)

    # Load visceral model
    vismodel = VisceralModel(num_outputs=2)
    vismodel.load_weights(args.vis_path)

    # connect to the AirSim simulator
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()

    # create coverage map and connect to client
    covMap = CoverageMap(map_size=64000, scale_ratio=20, state_size=6000, input_size=20, height_threshold=0.9, reward_norm=30, paint_radius=15)
    covMap.set_client(client=client)

    # create experiments directories and meta-data file
    experiment_dir = os.path.join(args.output_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(experiment_dir)
    airsim_rec = open(os.path.join(experiment_dir,"airsim_rec.txt"),"w")
    if args.store != "none":
        images_dir = os.path.join(experiment_dir, 'images')
        os.makedirs(images_dir)
        airsim_rec.write("TimeStamp\tStartingPoint\tDuration\tCoverage\tPOS_X\tPOS_Y\tPOS_Z\tRPM\tSpeed\tSteering\tImageFile\n")
    else:
        airsim_rec.write("TimeStamp\tStartingPoint\tDuration\tCoverage\tPOS_X\tPOS_Y\tPOS_Z\tRPM\tSpeed\tSteering\n")
    
    # actions list
    actions = [-1.0, -0.5, 0.0, 0.5, 1.0]

    # initiate buffer
    buffer = np.zeros((4, 84, 84), dtype=np.float32)

    # get random navigable point
    point, orientation = get_random_navigable_point(args.log_path)
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(point[0], point[1], point[2]), 
                                                airsim.to_quaternion(orientation[0], orientation[2], orientation[1])), True)

    # let the car start driving
    car_controls.throttle = 0.5
    car_controls.steering = 0
    client.setCarControls(car_controls)

    # starting point number
    start_time = time.time()
    start_idx = 0
    counter = 0

    print('Running model')
    image_requests = [airsim.ImageRequest("RCCamera", airsim.ImageType.Scene, False, False)]
    image_requests.append(airsim.ImageRequest("VisceralCamera_l", airsim.ImageType.Scene, False, False))
    image_requests.append(airsim.ImageRequest("VisceralCamera_lm", airsim.ImageType.Scene, False, False))
    image_requests.append(airsim.ImageRequest("VisceralCamera_m", airsim.ImageType.Scene, False, False))
    image_requests.append(airsim.ImageRequest("VisceralCamera_rm", airsim.ImageType.Scene, False, False))
    image_requests.append(airsim.ImageRequest("VisceralCamera_r", airsim.ImageType.Scene, False, False))

    while(True):

        time_stamp = int(time.time()*10000000)

        # get images data from AirSim
        responses = client.simGetImages(image_requests)

        # preprocess image for imitation learning model
        image_rgb_np, image_np = get_image(responses[0], args.res, grayscale=True)

        # preprocess images for visceral model
        visl_np, _ = get_image(responses[1], args.res)
        vislm_np, _ = get_image(responses[2], args.res)
        vism_np, _ = get_image(responses[3], args.res)
        visrm_np, _ = get_image(responses[4], args.res)
        visr_np, _ = get_image(responses[5], args.res)

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

        # convert visceral input to [1,84,84,1]
        visl_np = np.expand_dims(visl_np, axis=0).astype(np.float32) / 255.0
        vislm_np = np.expand_dims(vislm_np, axis=0).astype(np.float32) / 255.0
        vism_np = np.expand_dims(vism_np, axis=0).astype(np.float32) / 255.0
        visrm_np = np.expand_dims(visrm_np, axis=0).astype(np.float32) / 255.0
        visr_np = np.expand_dims(visr_np, axis=0).astype(np.float32) / 255.0

        # predict emotions and actions
        emotions_l, emotions_lm, emotions_m, emotions_rm, emotions_r, action_values = predict_vis_values_and_il(visl_np, vislm_np, vism_np, visrm_np, visr_np, input_img)

        # drive according to the selected policy
        pos_l, pos_lm, pos_m, pos_rm, pos_r = emotions_l.numpy()[0][0], emotions_lm.numpy()[0][0], emotions_m.numpy()[0][0], emotions_rm.numpy()[0][0], emotions_r.numpy()[0][0]
        neg_l, neg_lm, neg_m, neg_rm, neg_r = emotions_l.numpy()[0][1], emotions_lm.numpy()[0][1], emotions_m.numpy()[0][1], emotions_rm.numpy()[0][1], emotions_r.numpy()[0][1]
        curiosity_vals = np.array([pos_l, pos_lm, pos_m, pos_rm, pos_r])
        negative_vals = np.array([neg_l, neg_lm, neg_m, neg_rm, neg_r])
        random_vals = np.random.uniform(low=0.0, high=1.0, size=5)
        straight_vals = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        car_controls.steering = actions[np.argmax(args.alpha * action_values.numpy() + args.gamma * curiosity_vals + args.beta * negative_vals + args.delta * random_vals + args.epsilon * straight_vals)]
        
        end_time = time.time()

        # send action to AirSim
        client.setCarControls(car_controls)

        # get coverage data
        _, _, coverage = covMap.get_state_from_pose()

        # track duration and save image
        duration = round(time.time() - start_time,4)
        if args.store == "rgb":
            cv2.imwrite(os.path.join(images_dir, "im_{}.png".format(time_stamp)), image_rgb_np)

        # get position and car state
        client_pose = client.simGetVehiclePose()
        car_state = client.getCarState()

        # write meta-date to text file
        if args.store != "none":
            airsim_rec.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(time_stamp,start_idx,duration,coverage,client_pose.position.x_val,client_pose.position.y_val,client_pose.position.z_val,car_state.rpm,car_state.speed,car_controls.steering,"im_{}.png".format(time_stamp)))
        else:
            airsim_rec.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(time_stamp,start_idx,duration,coverage,client_pose.position.x_val,client_pose.position.y_val,client_pose.position.z_val,car_state.rpm,car_state.speed,car_controls.steering))
        
        counter += 1
        if counter % 1000 == 0:
            print('{} steps recorded'.format(counter))

        # reset the car if collided
        if abs(car_state.speed) < 0.02:
            
            print('trial = {}, coverage = {}, duration = {}'.format(start_idx, coverage, duration))

            # clean buffer
            buffer = np.zeros((4, 84, 84), dtype=np.float32)

            # clear coverage
            covMap.reset()

            # quit if reached the required duration
            if int(duration) > args.duration:
                airsim_rec.close()
                sys.exit()

            # get random navigable point
            start_idx += 1
            point, orientation = get_random_navigable_point(args.log_path)
            client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(point[0], point[1], point[2]), 
                                                airsim.to_quaternion(orientation[0], orientation[2], orientation[1])), True)
