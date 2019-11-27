import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.activations import softplus, relu
from tensorflow.keras.backend import random_normal
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Lambda, Conv2DTranspose, Reshape

# model definition class
class VAEModel(Model):
    def __init__(self, n_z, stddev_epsilon=1e-6, final_activation='sigmoid', trainable_encoder=True, trainable_decoder=[True]*5, in_channels=3, out_channels=3):
        super(VAEModel, self).__init__()
    
        self.n_z = n_z
        self.stddev_epsilon = stddev_epsilon
        self.final_activation = final_activation
        self.trainable_encoder = trainable_encoder
        self.trainable_decoder = trainable_decoder
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder architecture
        if self.in_channels == 3:
            self.conv1 = Conv2D(filters=64, kernel_size=4, strides=2, trainable=self.trainable_encoder)
        else:
            self.conv1_clean = Conv2D(filters=64, kernel_size=4, strides=2, trainable=self.trainable_encoder)
        self.bne1 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=self.trainable_encoder)
        self.conv2 = Conv2D(filters=128, kernel_size=4, strides=2, trainable=self.trainable_encoder)
        self.bne2 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=self.trainable_encoder)
        self.conv3 = Conv2D(filters=256, kernel_size=4, strides=2, trainable=self.trainable_encoder)
        self.bne3 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=self.trainable_encoder)
        self.flatten = Flatten()
        self.fce1 = Dense(units=1024, trainable=self.trainable_encoder)
        self.bne4 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=self.trainable_encoder)
        self.fce2 = Dense(units=2*self.n_z, trainable=self.trainable_encoder)

        # Latent space
        self.mean_params = Lambda(lambda x: x[:, :self.n_z])
        self.stddev_params = Lambda(lambda x: x[:, self.n_z:])

        # Decoder architecture
        self.fcd1 = Dense(units=1024, trainable=self.trainable_decoder[0])
        self.bnd1 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=self.trainable_decoder[0])
        self.fcd2 = Dense(units=128 * 7 * 7, trainable=self.trainable_decoder[1])
        self.bnd2 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=self.trainable_decoder[1])
        self.reshape = Reshape((7, 7, 128))
        self.deconv1 = Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='valid', trainable=self.trainable_decoder[2])
        self.bnd3 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=self.trainable_decoder[2])
        self.deconv2 = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', trainable=self.trainable_decoder[3])
        self.bnd4 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=self.trainable_decoder[3])
        if self.out_channels == 3:
            self.deconv3 = Conv2DTranspose(filters=self.out_channels, kernel_size=4, strides=2, padding='same', activation=self.final_activation, trainable=self.trainable_decoder[4])
        else:
            self.deconv3_clean = Conv2DTranspose(filters=self.out_channels, kernel_size=4, strides=2, padding='same', activation=self.final_activation, trainable=self.trainable_decoder[4])

    def call(self, x, inter=None):

        # Encoding
        if self.in_channels == 3:
            x1 = relu(self.bne1(self.conv1(x)))
        else:
            x1 = relu(self.bne1(self.conv1_clean(x)))
        x2 = relu(self.bne2(self.conv2(x1)))
        x3 = relu(self.bne3(self.conv3(x2)))
        x3 = self.flatten(x3)
        x4 = relu(self.bne4(self.fce1(x3)))
        x5 = self.fce2(x4)
        means = self.mean_params(x5)
        stddev = tf.math.exp(0.5*self.stddev_params(x5))
        eps = random_normal(tf.shape(stddev))

        # Decoding
        z = means + eps * stddev
        if inter is not None:
            z = tf.keras.layers.add([z,inter])
        x6 = relu(self.bnd1(self.fcd1(z)))
        x7 = relu(self.bnd2(self.fcd2(x6)))
        x7 = self.reshape(x7)
        x8 = relu(self.bnd3(self.deconv1(x7)))
        x9 = relu(self.bnd4(self.deconv2(x8)))
        if self.out_channels == 3:
            x10 = self.deconv3(x9)
        else:
            x10 = self.deconv3_clean(x9)

        return x10, means, stddev, z

    def load_encoder_weights_only(self, weights_path):

        # loading weights
        self.load_weights(weights_path)

        # Reloading decoder architecture
        self.fcd1 = Dense(units=1024, trainable=self.trainable_decoder[0])
        self.bnd1 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=self.trainable_decoder[0])
        self.fcd2 = Dense(units=128 * 7 * 7, trainable=self.trainable_decoder[1])
        self.bnd2 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=self.trainable_decoder[1])
        self.reshape = Reshape((7, 7, 128))
        self.deconv1 = Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='valid', trainable=self.trainable_decoder[2])
        self.bnd3 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=self.trainable_decoder[2])
        self.deconv2 = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', trainable=self.trainable_decoder[3])
        self.bnd4 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=self.trainable_decoder[3])
        if self.out_channels == 3:
            self.deconv3 = Conv2DTranspose(filters=self.out_channels, kernel_size=4, strides=2, padding='same', activation=self.final_activation, trainable=self.trainable_decoder[4])
        else:
            self.deconv3_clean = Conv2DTranspose(filters=self.out_channels, kernel_size=4, strides=2, padding='same', activation=self.final_activation, trainable=self.trainable_decoder[4])

    
    def load_decoder_weights_only(self, weights_path):

        # loading weights
        self.load_weights(weights_path)

        # Reloading encoder architecture
        if self.in_channels == 3:
            self.conv1 = Conv2D(filters=64, kernel_size=4, strides=2, trainable=self.trainable_encoder)
        else:
            self.conv1_clean = Conv2D(filters=64, kernel_size=4, strides=2, trainable=self.trainable_encoder)
        self.bne1 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=self.trainable_encoder)
        self.conv2 = Conv2D(filters=128, kernel_size=4, strides=2, trainable=self.trainable_encoder)
        self.bne2 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=self.trainable_encoder)
        self.conv3 = Conv2D(filters=256, kernel_size=4, strides=2, trainable=self.trainable_encoder)
        self.bne3 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=self.trainable_encoder)
        self.flatten = Flatten()
        self.fce1 = Dense(units=1024, trainable=self.trainable_encoder)
        self.bne4 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=self.trainable_encoder)
        self.fce2 = Dense(units=2*self.n_z, trainable=self.trainable_encoder)

