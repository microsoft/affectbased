import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import matplotlib.image as mpimg 
import seaborn as sns; sns.set()

parser = argparse.ArgumentParser()
parser.add_argument('--bg_img', '-bg_img', help='path to background image', default='C:\\Users\\user\\Documents\\experiments\\heatmap\\map_navigable.png', type=str)
parser.add_argument('--txt_path', '-txt_path', help='path to txt file containing coordinates', default='C:\\Users\\user\\Documents\\experiments\\data_collection_recordings\\il_curious_6.0\\recordings_01\\airsim_rec.txt', type=str)
parser.add_argument('--dest_path', '-dest_path', help='path to save the heatmap image', default='C:\\Users\\user\\Documents\\experiments\\heatmap\\heatmap_ilcur_raw.png', type=str)
args = parser.parse_args()

# open dataframe and remove unnecessary columns
df = pd.read_csv(args.txt_path, sep='\t')
df = df.drop(columns=['TimeStamp', 'StartingPoint', 'Duration', 'Coverage', 'POS_Z', 'RPM', 'Speed', 'Steering', 'ImageFile'])

# normalize
df['POS_X'] -= df['POS_X'].min()
df['POS_Y'] -= df['POS_Y'].min()
df['POS_X'] /= 5.0
df['POS_Y'] /= 5.0
df = df.astype(int)

# create numpy grid to plot later
cmap_np = np.zeros((df['POS_X'].max(), df['POS_Y'].max()))
for i in range(df.shape[0]):
    cmap_np[int(df.iloc[i][['POS_X']])-1,int(df.iloc[i][['POS_Y']])-1] += 1

# flip and generate heatmap
cmap_np = np.swapaxes(cmap_np, 0, 1)
plt.imshow(cmap_np, cmap='hot', interpolation='bicubic')
plt.axis('off')

# load background image
bg_img = Image.open(args.bg_img)

# convert heatmap to PIL image
buffer = io.StringIO()
cmap_canvas = plt.get_current_fig_manager().canvas
cmap_canvas.draw()
cmap_img = Image.frombytes('RGB', cmap_canvas.get_width_height(), cmap_canvas.tostring_rgb())
plt.close()
w, h = cmap_img.size
cmap_img.save(args.dest_path)
