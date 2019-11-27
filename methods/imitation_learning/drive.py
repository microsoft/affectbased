import sys
sys.path.append('.')
import airsim
import os
import numpy as np
import argparse
import tensorflow as tf
from model import ILModel
from methods.visceral_model.model import VisceralModel
from PIL import Image
from utils.coverage_map import CoverageMap
import cv2
import time
import datetime
import keyboard

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-path', help='model file path', default='C:\\Users\\user\\Documents\\models\\imitation_4images\\model40.ckpt', type=str)
parser.add_argument('--vis_path', '-vis_path', help='visceral model file path. wont load if empty', default='C:\\Users\\user\\Documents\\models\\visceral_models\\visceral_model_84_reg_both_norm_2s_noclipping2\\vismodel40.ckpt', type=str)
parser.add_argument('--cov', '-cov', help='size of the coverage map in the input. 0 for no coverage', default=20, type=int)
parser.add_argument('--safe', '-safe', dest='safe', help='add safety option to the model using visceral model', action='store_true')
parser.add_argument('--curious', '-curious', dest='curious', help='add curiosity option to the model using visceral model', action='store_true')
parser.add_argument('--debug', '-debug', dest='debug', help='debug mode, present fron camera on screen', action='store_true')
parser.add_argument('--store', '-store', help='store images to experiment folder. choose from [regular, depth, segmentation]. leave empty if dont', default='', type=str)
args = parser.parse_args()

# tf functions
@tf.function
def predict_action(image):
    return model(image)

@tf.function
def predict_values(image, vis_input):
    return model(image), vismodel(vis_input)

if __name__ == "__main__":

    # allow growth is possible using an env var in tf2.0
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    if args.store != "":
        # create experiments directories
        experiment_dir = os.path.join('C:\\Users\\user\\Documents\\AirSim', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        images_dir = os.path.join(experiment_dir, 'images')
        os.makedirs(images_dir)
        # create txt file
        airsim_rec = open(os.path.join(experiment_dir,"airsim_rec.txt"),"w")
        airsim_rec.write("TimeStamp\tPOS_X\tPOS_Y\tPOS_Z\tRPM\tSpeed\tSteering\tImageFile\n") 

    # Load imitation model
    model = ILModel()
    model.load_weights(args.path)

    # Load visceral model
    if args.safe or args.curious:
        vismodel = VisceralModel(num_outputs=2)
        vismodel.load_weights(args.vis_path)

    # connect to the AirSim simulator 
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()

    # create coverage map and connect to client
    if args.cov > 0:
        covMap = CoverageMap(map_size=64000, scale_ratio=20, state_size=6000, input_size=args.cov, height_threshold=0.9, reward_norm=30, paint_radius=15)
        covMap.set_client(client=client)

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
    if args.store == 'depth':
        Image_requests.append(airsim.ImageRequest("RCCamera", airsim.ImageType.DepthPerspective, True, False))
    if args.store == 'segmentation':
        Image_requests.append(airsim.ImageRequest("RCCamera", airsim.ImageType.Segmentation, False, False))
    if args.safe or args.curious:
        Image_requests.append(airsim.ImageRequest("0", airsim.ImageType.Scene, False, False))

    while(True):

        start_time = time.time()
        time_stamp = int(start_time*10000000)

        # get images from AirSim
        responses = client.simGetImages(Image_requests)
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        if img1d.size > 1:
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))
            image = Image.fromarray(img2d).convert('L')
            image_np = np.array(image.resize((84, 84)))
        else:
            image_np = np.zeros((84,84)).astype(float)
            
        if args.store == 'depth':
            # get depth image from airsim
            img1d = np.array(responses[1].image_data_float, dtype=np.float)
            img1d = 255/np.maximum(np.ones(img1d.size), img1d)
            if img1d.size > 1:
                img2d = np.reshape(img1d, (responses[1].height, responses[1].width))
                image = Image.fromarray(img2d)
                depth_np = np.array(image.resize((84, 84)).convert('L')) 
            else:
                depth_np = np.zeros((84,84)).astype(float)

        if args.store == 'segmentation':
            # get segmentation image from AirSim
            img1d = np.fromstring(responses[1].image_data_uint8, dtype=np.uint8)
            if img1d.size > 1:
                img2d = np.reshape(img1d, (responses[1].height, responses[1].width, 3))
                image = Image.fromarray(img2d).convert('L')
                seg_np = np.array(image.resize((84, 84)))
            else:
                seg_np = np.zeros((84,84)).astype(float)

        if args.safe or args.curious:
            # get image for visceral model
            img1d = np.fromstring(responses[-1].image_data_uint8, dtype=np.uint8)
            if img1d.size > 1:
                img2d = np.reshape(img1d, (responses[-1].height, responses[-1].width, 3))
                image = Image.fromarray(img2d)
                vis_np = np.array(image.resize((84, 84)))
            else:
                vis_np = np.zeros((84,84,3)).astype(float)

        # get coverage image
        if args.cov > 0:
            cov_image, _, _ = covMap.get_state_from_pose()

        # store images if requested
        if args.store != "":
            # save grayscaled image
            im = Image.fromarray(np.uint8(image_np))
            im.save(os.path.join(images_dir, "im_{}.png".format(time_stamp)))

            # save depth image
            if args.store == 'depth':
                depth_im = Image.fromarray(np.uint8(depth_np))
                depth_im.save(os.path.join(images_dir, "depth_{}.png".format(time_stamp)))

            # save segmentation image
            if args.store == 'segmentation':
                seg_im = Image.fromarray(np.uint8(seg_np))
                seg_im.save(os.path.join(images_dir, "seg_{}.png".format(time_stamp)))

        # combine both inputs
        if args.cov > 0:
            image_np[:cov_image.shape[0],:cov_image.shape[1]] = cov_image

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

        # predict action values
        if args.safe or args.curious:
            vis_input = np.expand_dims(vis_np, axis=0).astype(np.float32) / 255.0
            action_values, emotions = predict_values(input_img, vis_input)
            pos_t = emotions.numpy()[0][0]
            neg_t = emotions.numpy()[0][1]

            # apply safety driven decision
            if args.safe:
                if neg_t < -0.1:
                    action_values = action_values.numpy()
                    action_values[:,1:-1] = 0.0
            
            car_controls.steering = actions[np.argmax(action_values)] 

            # apply curiosity driven decision
            if args.curious:
                if pos_t > 0.12:
                    car_controls.steering = 0.0

        else:
            action_values = predict_action(input_img)
            car_controls.steering = actions[np.argmax(action_values)]
        
        end_time = time.time()

        # send action to AirSim
        client.setCarControls(car_controls)

        # store it if requested
        if args.store != "":
            # get position and car state
            client_pose = client.simGetVehiclePose()
            car_state = client.getCarState()

            # write meta-date to text file
            airsim_rec.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(time_stamp,client_pose.position.x_val,client_pose.position.y_val,client_pose.position.z_val,car_state.rpm,car_state.speed,car_controls.steering,"im_{}.png".format(time_stamp)))
        
        if args.safe or args.curious:
            print('steering = {}, pos = {}, neg = {}, fps = {}, actions_p = {}'.format(car_controls.steering, str(pos_t)[:5], str(neg_t)[:5], str(1/(end_time-start_time))[:5], np.around(action_values,decimals=3)[0]))
        else:
            print('steering = {}, actions_p = {}'.format(car_controls.steering, np.around(action_values,decimals=3)[0]))
            
        if keyboard.is_pressed('q'):  # if key 'q' is pressed
            if args.store != "":
                airsim_rec.close()
            sys.exit()