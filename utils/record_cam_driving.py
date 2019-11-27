import sys
sys.path.append('.')
import airsim
import os
import numpy as np
import pdb
import time
import datetime
import pprint
import keyboard  # using module keyboard
import math
from PIL import Image
from coverage_map import CoverageMap
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--camera', '-camera', help='type of the camera. choose from [rgb, depth, grayscale]', default='grayscale', type=str)
parser.add_argument('--dest_dir', '-dest_dir', help='destination folder path', default='C:\\Users\\user\\Documents\\AirSim', type=str)
parser.add_argument('--cov_size', '-cov_size', help='size of the coverage map in input. 0 if none', default=0, type=int)
args = parser.parse_args()

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

# create coverage map and connect to client
if args.cov_size > 0:
    covMap = CoverageMap(map_size=64000, scale_ratio=20, state_size=6000, input_size=args.cov_size, height_threshold=0.9, reward_norm=30, paint_radius=15)
    covMap.set_client(client=client)

# create experiments directories and meta-data file
experiment_dir = os.path.join(args.dest_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
images_dir = os.path.join(experiment_dir, 'images')
os.makedirs(images_dir)
airsim_rec = open(os.path.join(experiment_dir,"airsim_rec.txt"),"w") 
airsim_rec.write("TimeStamp\tPOS_X\tPOS_Y\tPOS_Z\tRPM\tSpeed\tSteering\tImageFile\n") 

# let the car start driving
car_controls.throttle = 0.5
car_controls.steering = 0
client.setCarControls(car_controls)

# actions list
actions = [-1.0, -0.5, 0, 0.5, 1.0]
actions_idx = 2

idx = 0
while True:

    time_stamp = int(time.time()*10000000)

    # change steering according to keyboard
    if keyboard.is_pressed('a'):
        actions_idx = 0
    if keyboard.is_pressed('w'):
        actions_idx = 1
    if keyboard.is_pressed('s'):
        actions_idx = 2
    if keyboard.is_pressed('e'):
        actions_idx = 3
    if keyboard.is_pressed('d'):
        actions_idx = 4

    car_controls.steering = actions[actions_idx]
    client.setCarControls(car_controls)
    print("steering: {}".format(actions[actions_idx]))

    if args.camera == 'depth':
        # get depth image from airsim
        responses = client.simGetImages([airsim.ImageRequest("RCCamera", airsim.ImageType.DepthPerspective, True, False)])
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255/np.maximum(np.ones(img1d.size), img1d)
        if img1d.size > 1:
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
            image = Image.fromarray(img2d)
            image_np = np.array(image.resize((84, 84)).convert('L')) 
        else:
            image_np = np.zeros((84,84)).astype(float)
    else: # args.camera = 'rgb' or 'grayscale'
        # get image from AirSim
        image_response = client.simGetImages([airsim.ImageRequest("RCCamera", airsim.ImageType.Scene, False, False)])[0]
        img1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
        if img1d.size > 1:
            img2d = np.reshape(img1d, (image_response.height, image_response.width, 3))
            image = Image.fromarray(img2d)
            if args.camera == 'grayscale':
                image = image.convert('L')
            image_np = np.array(image.resize((84, 84)))
            
        else:
            if args.camera == 'grayscale':
                image_np = np.zeros((84,84)).astype(float)
            else:
                image_np = np.zeros((84,84,3)).astype(float)

    # get coverage image
    if args.cov_size > 0:
        cov_image, reward, _ = covMap.get_state_from_pose()
        #print("reward: {}".format(reward))

        # combine both inputs
        if args.camera == 'depth' or args.camera == 'grayscale':
            image_np[:cov_image.shape[0],:cov_image.shape[1]] = cov_image
        else:
            cov_rgb_image = np.expand_dims(cov_image, axis=2)
            cov_rgb_image = np.repeat(cov_rgb_image, 3, axis=2)
            image_np[:cov_rgb_image.shape[0],:cov_rgb_image.shape[1],:] = cov_rgb_image

    # save the image
    im = Image.fromarray(np.uint8(image_np))
    im.save(os.path.join(images_dir, "im_{}.png".format(time_stamp)))
	
    # get position and car state
    client_pose = client.simGetVehiclePose()
    car_state = client.getCarState()

    # write meta-date to text file
    airsim_rec.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(time_stamp,client_pose.position.x_val,client_pose.position.y_val,client_pose.position.z_val,car_state.rpm,car_state.speed,car_controls.steering,"im_{}.png".format(time_stamp))) 
	
    # present state image
    cv2.imshow('navigation map', image_np)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if keyboard.is_pressed('q'):  # if key 'q' is pressed
        airsim_rec.close()
        cv2.destroyAllWindows()
        quit()
	
    idx += 1
    #time.sleep(0.2)