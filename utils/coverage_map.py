import sys
sys.path.append('.')
import airsim
import cv2
import numpy as np
import math
import time
from PIL import Image
import copy

class CoverageMap:

    # initiate coverage map
    def __init__(self, map_size, scale_ratio, state_size, input_size, height_threshold, reward_norm, reward_count=1, paint_radius=5):

        self.map_size = map_size # map size to be in the shape of [map_size, map_size], in centimeters
        self.scale_ratio = scale_ratio # scale ratio to be used to reduce the map size and increase performance
        self.state_size = int(state_size / self.scale_ratio) # state size to be in the shape of [state_size, state_size], in centimeters
        self.input_size = input_size # final input size to be in the shape of [input_size, input_size], in centimeters
        self.height_threshold = height_threshold # height_threshold, in meters
        self.saved_state = np.zeros((self.input_size, self.input_size)) # save state in each iteration
        self.saved_reward = 0 # save reward in each iteration
        self.reward_norm = reward_norm # factor to normalize the reward
        self.reward_count = reward_count
        self.reward_list = [1.0] * self.reward_count
        self.paint_radius = paint_radius

        # prepare clean coverage map
        self.cov_map = np.zeros((int(self.map_size / self.scale_ratio), int(self.map_size / self.scale_ratio)))
        self.total_mileage = 0

    # clear coverage map and state
    def reset(self):

        self.saved_state = np.zeros((self.input_size, self.input_size))
        self.saved_reward = 0
        self.cov_map = np.zeros((int(self.map_size / self.scale_ratio), int(self.map_size / self.scale_ratio)))
        self.reward_list = [1.0] * self.reward_count
        self.total_mileage = 0

    # set airsim client, to get position, orientation and lidar data
    def set_client(self, client):

        self.client = client

    # get the entire map, rescaled to the input size
    def get_map_scaled(self):

        # resize coverage image using PIL
        im = Image.fromarray(np.uint8(self.cov_map))
        im = im.resize((self.input_size,self.input_size), Image.BILINEAR)
        binary_map = np.array(im)

        # make it binary
        idxs = np.where(binary_map > 0.0)
        binary_map[idxs] = 255.0

        return binary_map

    # get state image from the coverage map using only the location
    def get_state_from_pose(self):
        
        # get car position, convert it to be relative to the world in centimeters, and convert it according to scale ratio
        client_pose = self.client.simGetVehiclePose()
        pose = [int(round(((self.map_size / 2) - (client_pose.position.x_val * 100.0))/self.scale_ratio)), 
                int(round(((self.map_size / 2) + (client_pose.position.y_val * 100.0))/self.scale_ratio)),
                int(round(((self.map_size / 2) + (client_pose.position.z_val * 100.0))/self.scale_ratio))]

        # get car orientation
        angles = airsim.to_eularian_angles(client_pose.orientation)


        # paint and get number of new pixels for reward computation
        new_pixels = 0
        for i in range(-self.paint_radius, self.paint_radius):
            for j in range(-self.paint_radius, self.paint_radius):
                if math.sqrt(math.pow(i,2)+math.pow(j,2)) <= self.paint_radius and self.cov_map[pose[0]+i,pose[1]+j] == 0.0:
                    self.cov_map[pose[0]+i,pose[1]+j] = 255.0
                    new_pixels += 1

        # extract state from nav map
        x_range = (int(pose[0] - self.state_size/2), int(pose[0] + self.state_size/2))
        y_range = (int(pose[1] - self.state_size/2), int(pose[1] + self.state_size/2))
        state = self.cov_map[x_range[0]:x_range[1],y_range[0]:y_range[1]]
        
        # scale using PIL
        im = Image.fromarray(np.uint8(state))
        im = im.resize((self.input_size*2, self.input_size*2), Image.ANTIALIAS)

        # rotate according to the orientation
        im = im.rotate(math.degrees(angles[2]))
        state = np.array(im)
        
        # extract half of the portion to receive state in input size, save it for backup
        self.saved_state = state[int(state.shape[0]/2 - state.shape[0]/4):int(state.shape[0]/2 + state.shape[0]/4), 
                        int(state.shape[1]/2 - state.shape[1]/4):int(state.shape[1]/2 + state.shape[1]/4)]
        
        # compute reward
        self.saved_reward = min(new_pixels / self.reward_norm, 1.0)
        self.reward_list.append(self.saved_reward)
        del self.reward_list[0]

        self.total_mileage += new_pixels * ((self.scale_ratio*self.scale_ratio) / (100*100))
        return self.saved_state, sum(self.reward_list) / len(self.reward_list), round(self.total_mileage, 4)

    # get state image from the coverage map using offline pose
    def get_state_from_offline_pose(self, pose):
        
        # get car position, convert it to be relative to the world in centimeters, and convert it according to scale ratio
        pos = [int(round(((self.map_size / 2) - (pose.position.x_val * 100.0))/self.scale_ratio)), 
                int(round(((self.map_size / 2) + (pose.position.y_val * 100.0))/self.scale_ratio)),
                int(round(((self.map_size / 2) + (pose.position.z_val * 100.0))/self.scale_ratio))]

        # get car orientation
        angles = airsim.to_eularian_angles(pose.orientation)

        # paint and get number of new pixels for reward computation
        new_pixels = 0
        for i in range(-self.paint_radius, self.paint_radius):
            for j in range(-self.paint_radius, self.paint_radius):
                if math.sqrt(math.pow(i,2)+math.pow(j,2)) <= self.paint_radius and self.cov_map[pos[0]+i,pos[1]+j] == 0.0:
                    self.cov_map[pos[0]+i,pos[1]+j] = 255.0
                    new_pixels += 1

        # extract state from nav map
        x_range = (int(pos[0] - self.state_size/2), int(pos[0] + self.state_size/2))
        y_range = (int(pos[1] - self.state_size/2), int(pos[1] + self.state_size/2))
        state = self.cov_map[x_range[0]:x_range[1],y_range[0]:y_range[1]]
        
        # scale using PIL
        im = Image.fromarray(np.uint8(state))
        im = im.resize((self.input_size*2, self.input_size*2), Image.ANTIALIAS)

        # rotate according to the orientation
        im = im.rotate(math.degrees(angles[2]))
        state = np.array(im)
        
        # extract half of the portion to receive state in input size, save it for backup
        self.saved_state = state[int(state.shape[0]/2 - state.shape[0]/4):int(state.shape[0]/2 + state.shape[0]/4), 
                        int(state.shape[1]/2 - state.shape[1]/4):int(state.shape[1]/2 + state.shape[1]/4)]
        
        # compute reward
        self.saved_reward = min(new_pixels / self.reward_norm, 1.0)
        self.reward_list.append(self.saved_reward)
        del self.reward_list[0]

        self.total_mileage += new_pixels * ((self.scale_ratio*self.scale_ratio) / (100*100))
        return self.saved_state, sum(self.reward_list) / len(self.reward_list), round(self.total_mileage, 4)

    # get state image from the coverage map
    def get_state_from_lidar(self):

        # get car position, convert it to be relative to the world in centimeters, and convert it according to scale ratio
        client_pose = self.client.simGetVehiclePose()
        pose = [int(round(((self.map_size / 2) - (client_pose.position.x_val * 100.0))/self.scale_ratio)), 
                int(round(((self.map_size / 2) + (client_pose.position.y_val * 100.0))/self.scale_ratio)),
                int(round(((self.map_size / 2) + (client_pose.position.z_val * 100.0))/self.scale_ratio))]

        # get car orientation
        angles = airsim.to_eularian_angles(client_pose.orientation)

        # get lidar data
        lidarData = self.client.getLidarData(lidar_name='LidarSensor1', vehicle_name='Car')
        if (len(lidarData.point_cloud) < 3):
            print("\tNo points received from Lidar data")
            return self.saved_state, self.saved_reward
        else:
            
            # reshape array of floats to array of [X,Y,Z]
            points = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0]/3), 3))

            # trim high points
            idxs = np.where(points[:,2] < self.height_threshold)[0]
            points_trimmed = np.delete(points, idxs, axis=0)

            # rotate it according to the current orientation
            rot_angle = angles[2] + math.pi
            rot_matrix = np.array([[-math.cos(rot_angle), -math.sin(rot_angle), 0],
                                    [math.sin(rot_angle), -math.cos(rot_angle), 0],
                                    [0, 0, 1]])
            points_trimmed = np.dot(points_trimmed, rot_matrix)

            # convert it to be relative to the world in centimeters, z axis is not relevant 
            points_trimmed[:,0] = np.subtract(pose[0],np.rint(points_trimmed[:,0] * 100.0 / self.scale_ratio))
            points_trimmed[:,1] = np.add(pose[1],np.rint(points_trimmed[:,1] * 100.0 / self.scale_ratio))
            points_trimmed = points_trimmed.astype(int)

            # paint selected indexes, and sum new pixels
            new_pixels = 0
            for i in range(points_trimmed.shape[0]):
                if self.cov_map[points_trimmed[i][0],points_trimmed[i][1]] == 0:
                    self.cov_map[points_trimmed[i][0],points_trimmed[i][1]] = 255
                    new_pixels += 1
            
            # extract state from nav map
            x_range = (int(pose[0] - self.state_size/2), int(pose[0] + self.state_size/2))
            y_range = (int(pose[1] - self.state_size/2), int(pose[1] + self.state_size/2))
            state = self.cov_map[x_range[0]:x_range[1],y_range[0]:y_range[1]]
            
            # scale using PIL
            im = Image.fromarray(np.uint8(state))
            im = im.resize((self.input_size*2, self.input_size*2), Image.ANTIALIAS)

            # rotate according to the orientation
            im = im.rotate(math.degrees(angles[2]))
            state = np.array(im)
            
            # extract half of the portion to receive state in input size, save it for backup
            self.saved_state = state[int(state.shape[0]/2 - state.shape[0]/4):int(state.shape[0]/2 + state.shape[0]/4), 
                            int(state.shape[1]/2 - state.shape[1]/4):int(state.shape[1]/2 + state.shape[1]/4)]
            
            # compute reward
            self.saved_reward = min(new_pixels / self.reward_norm, 1.0)
            self.reward_list.append(self.saved_reward)
            del self.reward_list[0]

            return self.saved_state, sum(self.reward_list) / len(self.reward_list)

def get_depth_image(client):

    responses = client.simGetImages([airsim.ImageRequest("RCCamera", airsim.ImageType.DepthPerspective, True, False)])
    return transform_depth_input(responses)


def transform_depth_input(responses):
    img1d = np.array(responses[0].image_data_float, dtype=np.float)

    if img1d.size > 1:

        img1d = 255/np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert('L')) 

        return im_final

    return np.zeros((84,84)).astype(float)

if __name__ == "__main__":

    # connect to AirSim 
    client = airsim.CarClient()
    client.confirmConnection()
    
    # create coverage map and connect to client
    covMap = CoverageMap(map_size=12000, scale_ratio=20, state_size=6000, input_size=20, height_threshold=0.9, reward_norm=30, paint_radius=15)
    covMap.set_client(client=client)

    # start free run session
    i = 1
    fps_sum = 0
    while True:
        startTime = time.time()

        # get state and show it on screen
        state, reward = covMap.get_state_from_pose()
        depth_im = get_depth_image(client)
        depth_im[:state.shape[0],:state.shape[1]] = state

        cv2.imshow('navigation map (q to exit)', depth_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        endTime = time.time()

        # present fps or reward
        fps_sum += (1/(endTime-startTime))
        print("reward: {}".format(reward))

        i+=1
        
    cv2.destroyAllWindows()
