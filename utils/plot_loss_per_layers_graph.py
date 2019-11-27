import os
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', '-csv_path', help='path to csv file', default='C:\\Users\\user\\Documents\\models\\segmentation_layers_loss\\results_test.csv', type=str)
parser.add_argument('--flipped', '-flipped', dest='flipped', help='choose to flip between epochs and layers on x axis', action='store_true')
args = parser.parse_args()

if args.flipped:
    # prepare lists for ploting
    steps = [4,3,2,1]
    loss_10epochs, loss_20epochs, loss_30epochs, loss_40epochs = [], [], [], []
    columns = ['4 layers','3 layers','2 layers','1 layers']

    current_df = pd.read_csv(args.csv_path)
    for i in range(0, len(columns), 1):
        loss_10epochs.append(float(current_df.iloc[9][[columns[i]]]))
        loss_20epochs.append(float(current_df.iloc[19][[columns[i]]]))
        loss_30epochs.append(float(current_df.iloc[29][[columns[i]]]))
        loss_40epochs.append(float(current_df.iloc[39][[columns[i]]]))

    # plot the results
    plt.plot(steps, loss_10epochs)
    plt.plot(steps, loss_20epochs)
    plt.plot(steps, loss_30epochs)
    plt.plot(steps, loss_40epochs)
    plt.xticks([1,2,3,4])
    plt.legend(['10 epochs','20 epochs','30 epochs','40 epochs'])
    plt.xlabel('Trainable layers')
    plt.ylabel('Loss')
    plt.title('Loss per trainable layers')

else:
    # prepare lists for ploting
    steps, loss_4layers, loss_3layers, loss_2layers, loss_1layers = [], [], [], [], []

    current_df = pd.read_csv(args.csv_path)
    for i in range(0, current_df.shape[0], 1):
        steps.append(int(current_df.iloc[i][['Step']]))
        loss_4layers.append(math.log(float(current_df.iloc[i][['4 layers']])))
        loss_3layers.append(math.log(float(current_df.iloc[i][['3 layers']])))
        loss_2layers.append(math.log(float(current_df.iloc[i][['2 layers']])))
        loss_1layers.append(math.log(float(current_df.iloc[i][['1 layers']])))

    # plot the results
    plt.plot(steps, loss_4layers)
    plt.plot(steps, loss_3layers)
    plt.plot(steps, loss_2layers)
    plt.plot(steps, loss_1layers)
    plt.legend(['4 trainable layers','3 trainable layers','2 trainable layers','1 trainable layers'])
    plt.xlabel('Epochs')
    plt.ylabel('Log loss')
    plt.title('Loss per epochs')
    
plt.grid(True)
plt.show()
#plt.savefig(os.path.join(os.path.dirname(args.csv_path),"losses_layers.svg"))
