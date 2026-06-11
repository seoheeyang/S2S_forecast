import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, Dropout,
    UpSampling2D, LayerNormalization, Cropping2D
)

def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))

class PretrainAutoencoder(Model):
    def __init__(self, xdim, ydim, zdim,
                 filter1=64, filter2=128,
                 drop_rate=0.2,
                 training=True):
        super().__init__()
        self.xdim, self.ydim, self.zdim = int(xdim), int(ydim), int(zdim)
        self.filter1, self.filter2 = int(filter1), int(filter2)
        self.drop_rate = float(drop_rate)
        self.training_flag = bool(training)

        inp = keras.Input(shape=(self.xdim, self.ydim, self.zdim), name="cnn_input")

        # =========================
        # Encoder = CNN trunk 
        # =========================
        x = Conv2D(filters=self.filter1, kernel_size=3, strides=(2, 2), padding="same",
                   kernel_initializer="glorot_uniform", name="conv1")(inp)
        x = BatchNormalization(name="bn1")(x)
        x = Activation("tanh", name="tanh1")(x)

        x = Conv2D(filters=self.filter2, kernel_size=3, strides=(2, 2), padding="same",
                   kernel_initializer="glorot_uniform", name="conv2")(x)
        x = BatchNormalization(name="bn2")(x)
        x = Activation("tanh", name="tanh2")(x)

        if self.drop_rate > 0:
            x = Dropout(self.drop_rate, name="drop")(x)

        # trunk output = feature map
        self.encoder_trunk = keras.Model(inp, x, name="cnn_trunk")

        # =========================
        # Decoder (reconstruction)
        # =========================
        d = UpSampling2D(size=(2, 2), name="up1")(x)
        d = Conv2D(self.filter1, kernel_size=3, padding="same", name="up_conv1")(d)
        d = LayerNormalization(epsilon=1e-6, name="up_ln1")(d)
        d = Activation(gelu, name="up_act1")(d)

        d = UpSampling2D(size=(2, 2), name="up2")(d)
        d = Conv2D(self.zdim, kernel_size=3, padding="same", name="up_conv2")(d)
        d = Activation("linear", name="decoded")(d)

        conv2_h = int(np.ceil(self.xdim / 4))
        conv2_w = int(np.ceil(self.ydim / 4))
        recon_h0 = conv2_h * 4
        recon_w0 = conv2_w * 4
        crop_h = recon_h0 - self.xdim
        crop_w = recon_w0 - self.ydim
        if crop_h > 0 or crop_w > 0:
            d = Cropping2D(cropping=((0, crop_h), (0, crop_w)), name="cropping_fix")(d)

        self.autoencoder = keras.Model(inp, d, name="AE_full")

    def call(self, inputs, training=False):
        return self.autoencoder(inputs, training=training)

    def save_encoder_weights(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        self.encoder_trunk.save_weights(os.path.join(dir_path, "pretrained_encoder.weights.h5"))
        self.autoencoder.save_weights(os.path.join(dir_path, "autoencoder_full.weights.h5"))
        print(f"Weights saved to {dir_path}")

