import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Dropout, Flatten, Dense

# 2025.11._______________________________________________________________________
# This code is an improved and extended model developed by Seohee H. Yang and Seok-Woo Son, based on the MAML (Model-Agnostic Meta-Learning) framework provided by Oh and Ham (2024), originally introduced in: #Oh, S. H., & Ham, Y. G. (2024). Few-shot learning for Korean winter temperature forecasts. npj Climate and Atmospheric Science, 7(1), 279.
(Original MAML repository: https://github.com/XXXXXXXX)

# To enhance extreme value prediction performance, the ExtremeCast methodology proposed in the following study was additionally integrated into this model:
# Xu, W., Chen, K., Han, T., Chen, H., Ouyang, W., & Bai, L. (2024).
# ExtremeCast: Boosting Extreme Value Prediction for Global Weather Forecast.
arXiv preprint arXiv:2402.01295.
#_________________________________________________________________________________
# ----- Exloss -----Seohee H. Yang
SCALE = 100.0/81.0

def make_exloss_with_fixed_quantiles(q10_const, q90_const, scale=SCALE):
    q10_const = tf.constant(q10_const, dtype=tf.float32)
    q90_const = tf.constant(q90_const, dtype=tf.float32)
    def exloss(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        cond_low  = tf.logical_and(y_pred >= y_true, y_true <  q10_const)
        cond_high = tf.logical_and(y_pred <= y_true, y_true >  q90_const)
        s = tf.where(tf.logical_or(cond_low, cond_high),
                     tf.cast(scale, y_pred.dtype), tf.ones_like(y_pred))
        err = s * (y_pred - y_true)
        return tf.reduce_mean(tf.square(err))
    return exloss

# ----- Model -----
class MAMLNets(Model):

    def __init__(self, shot, xdim, ydim, zdim,
                 filter1, filter2, update, training=True,
                 stage=2,                # 1:MSE, 2:Exloss    Seohee H. Yang
                 q10=None, q90=None,     # Global quantile of Exloss Seohee H. Yang
                 inference_noise_std=0.1, n_samples=50):
        super(MAMLNets, self).__init__()

        self.xdim, self.ydim, self.zdim = xdim, ydim, zdim
        self.support_samples = shot
        self.filter1, self.filter2 = filter1, filter2
        self.update = update
        self.training = training

        self.stage = stage
        self.loss_mse = keras.losses.MeanSquaredError()
        self.loss_ex = make_exloss_with_fixed_quantiles(q10, q90) if (stage == 2 and q10 is not None and q90 is not None) else None

        # ExBooster
        self.inference_noise_std = inference_noise_std
        self.n_samples = n_samples

        self.initializer = tf.initializers.GlorotUniform()
        self.inner_optimizer = tf.keras.optimizers.Adam()

        inputs = keras.Input(shape=(self.xdim, self.ydim, self.zdim))
        x = Conv2D(strides=(2,2), filters=self.filter1, kernel_size=3, padding='same',
                   kernel_initializer=self.initializer)(inputs)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)

        x = Conv2D(filters=self.filter2, strides=(2,2), kernel_size=3, padding='same',
                   kernel_initializer=self.initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)

        x = Dropout(0.2)(x)
        x = Flatten()(x)
        outputs = Dense(1)(x)

        self.forward = tf.keras.Model(inputs=inputs, outputs=outputs, trainable=self.training)

    # Loss function selector #Seohee H. Yang
    def _loss(self, y_true, y_pred):
        if self.stage == 2 and self.loss_ex is not None:
            return self.loss_ex(y_true, y_pred)
        return self.loss_mse(y_true, y_pred)

    # ---- encoder weights I/O ----
    def save_encoder(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        # build once
        _ = self.forward(tf.zeros([1, self.xdim, self.ydim, self.zdim]))
        self.forward.save_weights(os.path.join(dir_path, 'cnn_encoder.weights.h5'))

    def load_encoder(self, dir_path):
        path = os.path.join(dir_path, 'cnn_encoder.weights.h5')
        _ = self.forward(tf.zeros([1, self.xdim, self.ydim, self.zdim]))
        self.forward.load_weights(path)

    # Scalar prediction helper
    def _predict_scalar(self, x, training_flag=None):
        if training_flag is None:
            training_flag = self.training
        y = self.forward(x, training=training_flag)
        return tf.reshape(y, [-1])

    def call(self, inp_support, lab_support, inp_query, lab_query):
        inp_support = tf.reshape(inp_support, [self.support_samples, self.xdim, self.ydim, self.zdim])
        lab_support = tf.reshape(lab_support, [self.support_samples])
        inp_query   = tf.reshape(inp_query,   [-1, self.xdim, self.ydim, self.zdim])
        lab_query   = tf.reshape(lab_query,   [-1])

        em_loss, em_pred = [], []

        # inner step 1
        with tf.GradientTape() as train_tape:
            pred = self._predict_scalar(inp_support)
            inner_loss = self._loss(lab_support, pred)
        grads = train_tape.gradient(inner_loss, self.forward.trainable_variables)
        self.inner_optimizer.apply_gradients(zip(grads, self.forward.trainable_variables))

        # query eval
        query_pred = self._predict_scalar(inp_query)
        outer_loss = self._loss(lab_query, query_pred)
        em_loss.append(outer_loss); em_pred.append(query_pred)

        # additional inner updates #Seohee H. Yang
        for _ in range(self.update-1):
            with tf.GradientTape() as train_tape:
                pred = self._predict_scalar(inp_support)
                inner_loss = self._loss(lab_support, pred)
            grads = train_tape.gradient(inner_loss, self.forward.trainable_variables)
            self.inner_optimizer.apply_gradients(zip(grads, self.forward.trainable_variables))

            query_pred = self._predict_scalar(inp_query)
            outer_loss = self._loss(lab_query, query_pred)
            em_loss.append(outer_loss); em_pred.append(query_pred)

        loss = tf.reduce_mean(em_loss[-1])
        predictions = tf.squeeze(em_pred[-1])
        return predictions, loss

    # Stage-4: m-sample median  Seohee H. Yang
    def exbooster_scalar(self, x):
        base = self._predict_scalar(x, training_flag=False)  # [B]
        base = tf.reshape(base, [1, -1])                    # [1, B]
        base = tf.tile(base, [self.n_samples, 1])           # [m, B]
        noise = tf.random.normal(
            tf.shape(base),
            stddev=self.inference_noise_std,
            dtype=base.dtype
        )                                                   # [m, B]
        preds = base + noise                                # [m, B]

        preds = tf.sort(preds, axis=0)                          # [m, B]
        median = preds[self.n_samples // 2]                     # [B]
        return median
