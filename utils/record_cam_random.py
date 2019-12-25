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
import argparse
import cv2
from utils.import_data import get_random_navigable_point

parser = argparse.ArgumentParser()
parser.add_argument('--camera', '-camera', help='type of the camera. choose from [rgb, depth, segmentation, pair]', default='pair', type=str)
parser.add_argument('--dest_dir', '-dest_dir', help='destination folder path', default='C:\\Users\\user\\Documents\\AirSim', type=str)
parser.add_argument('--log_path', '-log_path', help='path to simulation log file', default='C:\\Users\\user\\Documents\\Unreal Projects\\Maze\\Saved\\Logs\\Car_Maze.log', type=str)
parser.add_argument('--res', '-res', help='destination resolution for images to be stored', default=84, type=int)
parser.add_argument('--debug', '-debug', dest='debug', help='debug mode, present fron camera on screen', action='store_true')
args = parser.parse_args()

# connect to the AirSim simulator
client = airsim.VehicleClient()
client.confirmConnection()

# create experiments directories and meta-data file
experiment_dir = os.path.join(args.dest_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
images_dir = os.path.join(experiment_dir, 'images')
os.makedirs(images_dir)
airsim_rec = open(os.path.join(experiment_dir,"airsim_rec.txt"),"w") 
airsim_rec.write("TimeStamp\tImageFile\n") 

Image_requests = [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
if args.camera == 'depth':
    Image_requests.append(airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False))
elif args.camera == 'segmentation':
    Image_requests.append(airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False))

idx = 0
while True:

    # get random navigable point
    point, orientation = get_random_navigable_point(args.log_path)
    point[2] = -0.7
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(point[0], point[1], point[2]), 
                                                airsim.to_quaternion(orientation[0], orientation[2], orientation[1])), True)

    time_stamp = int(time.time()*10000000)

    # get images data from AirSim
    responses = client.simGetImages(Image_requests)

    # filter scene image
    img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
    if img1d.size > 1:
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))
        image = Image.fromarray(img2d)
        image_np = np.array(image.resize((args.res, args.res)))
    else:
        image_np = np.zeros((args.res,args.res,3)).astype(float)

    # filter depth map
    if args.camera == 'depth':
        img1d = np.array(responses[1].image_data_float, dtype=np.float)
        img1d = 255/np.maximum(np.ones(img1d.size), img1d)
        if img1d.size > 1:
            img2d = np.reshape(img1d, (responses[1].height, responses[1].width))
            image = Image.fromarray(img2d)
            depth_np = np.array(image.resize((args.res, args.res)).convert('L')) 
        else:
            depth_np = np.zeros((args.res,args.res)).astype(float)
    elif args.camera == 'segmentation':
        img1d = np.fromstring(responses[1].image_data_uint8, dtype=np.uint8)
        if img1d.size > 1:
            img2d = np.reshape(img1d, (responses[1].height, responses[1].width, 3))
            image = Image.fromarray(img2d)
            seg_np = np.array(image.resize((args.res, args.res))) 
        else:
            seg_np = np.zeros((args.res,args.res,3)).astype(float)

    # save the images
    cv2.imwrite(os.path.join(images_dir, "im_{}.png".format(time_stamp)), image_np)
    if args.camera == 'depth':
        cv2.imwrite(os.path.join(images_dir, "depth_{}.png".format(time_stamp)), depth_np)
    elif args.camera == 'segmentation':
        cv2.imwrite(os.path.join(images_dir, "seg_{}.png".format(time_stamp)), seg_np)

    # write meta-date to text file
    airsim_rec.write("{}\t{}\n".format(time_stamp,"im_{}.png".format(time_stamp)))

    # save consecutive image for flow estimation dataset
    if args.camera == 'pair': 
        step_vector = [math.cos(orientation[1]) / 10.0, math.sin(orientation[1]) / 0.5, 0] 
        point += step_vector # perform short step in the same direction of the orientation
        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(point[0], point[1], point[2]), 
                                                airsim.to_quaternion(orientation[0], orientation[2], orientation[1])), True)

        # get images data from AirSim
        responses = client.simGetImages(Image_requests)

        # filter scene image
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        if img1d.size > 1:
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))
            image = Image.fromarray(img2d)
            image_np = np.array(image.resize((args.res, args.res)))
        else:
            image_np = np.zeros((args.res,args.res,3)).astype(float)

        # save the image
        cv2.imwrite(os.path.join(images_dir, "next_{}.png".format(time_stamp)), image_np)

    # present state image
    if args.debug:
        cv2.imshow('image', image_np)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if keyboard.is_pressed('q'):  # if key 'q' is pressed
        airsim_rec.close()
        cv2.destroyAllWindows()
        quit()
	
    idx += 1

    if idx % 200 == 0:
        print("{} images recorded".format(idx))