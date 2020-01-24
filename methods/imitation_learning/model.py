# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# model definition class
class ILModel(Model):
  def __init__(self, num_actions=5):
    super(ILModel, self).__init__()
    self.conv1 = Conv2D(filters=16, kernel_size=8, strides=4, activation='relu')
    self.conv2 = Conv2D(filters=32, kernel_size=4, strides=2, activation='relu')
    self.conv3 = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(units=256, activation='relu')
    self.d2 = Dense(units=num_actions, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)