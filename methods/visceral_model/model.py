import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Concatenate

# model definition class
class VisceralModel(Model):
    def __init__(self, num_outputs=2):
        super(VisceralModel, self).__init__()

        # Convolution stack
        self.conv1 = Conv2D(filters=32, kernel_size=5, strides=2, activation='relu')
        self.bn1 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv2 = Conv2D(filters=48, kernel_size=4, strides=2, activation='relu')
        self.bn2 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv3 = Conv2D(filters=64, kernel_size=4, strides=2, activation='relu')
        self.bn3 = BatchNormalization(momentum=0.9, epsilon=1e-5)

        # FC stack
        self.flatten = Flatten()
        self.d1 = Dense(2048, activation='relu')
        self.bn5 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.d2 = Dense(num_outputs)

    def call(self, x, x2=None):

        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        
        x = self.flatten(x)
        x = self.bn5(self.d1(x))
        x = self.d2(x)

        return x