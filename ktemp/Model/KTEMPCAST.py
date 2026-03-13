import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Layer, Conv2D, BatchNormalization, Activation, Dropout,
    Dense, GlobalAveragePooling2D
)

# -------------------------
# Activation
# -------------------------
def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))

# -------------------------
# ExLoss (z-score space)
# -------------------------
def make_exloss_with_fixed_quantiles(q10_const, q90_const, scale=100/81):
    q10_const = tf.constant(float(q10_const), dtype=tf.float32)
    q90_const = tf.constant(float(q90_const), dtype=tf.float32)
    scale = tf.constant(float(scale), dtype=tf.float32)

    def exloss(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        missed_cold = tf.logical_and(y_pred >= y_true, y_true < q10_const)
        missed_warm = tf.logical_and(y_pred <= y_true, y_true > q90_const)
        mask = tf.logical_or(missed_cold, missed_warm)

        w = tf.where(mask, scale, 1.0)
        err = w * (y_pred - y_true)
        return tf.reduce_mean(tf.square(err))
    return exloss

def _build_cnn_trunk(zdim, filter1=64, filter2=128, drop_rate=0.2):
    inp = keras.Input(shape=(None, None, int(zdim)), name="cnn_input")

    x = Conv2D(filters=int(filter1), kernel_size=3, strides=(2, 2), padding="same",
               kernel_initializer="glorot_uniform", name="conv1")(inp)
    x = BatchNormalization(name="bn1")(x)
    x = Activation("tanh", name="tanh1")(x)

    x = Conv2D(filters=int(filter2), kernel_size=3, strides=(2, 2), padding="same",
               kernel_initializer="glorot_uniform", name="conv2")(x)
    x = BatchNormalization(name="bn2")(x)
    x = Activation("tanh", name="tanh2")(x)

    if float(drop_rate) > 0:
        x = Dropout(float(drop_rate), name="drop")(x)

    return keras.Model(inp, x, name="cnn_trunk")

class KTempCastModel(Model):
    def __init__(self,
                 shot, xdim, ydim, zdim,
                 filter1=64, filter2=128,
                 update=5,
                 band_kernel_size=None, branch_ratio=None,
                 drop_rate=0.2,
                 mlp_ratio=None,
                 region_idx=None,
                 # losses
                 q10=None, q90=None,
                 ex_scale=100/81,
                 alpha_ex=1.0,
                 # optim
                 inner_lr=0.001,
                 meta_lr=1e-4,
                 clip_norm=1.0,
                 gate_l2=1e-4,
                 use_tf_function=True,
                 freeze_trunk_in_inner=True):
        super().__init__()
        self.xdim, self.ydim, self.zdim = int(xdim), int(ydim), int(zdim)
        self.support_samples = int(shot)
        self.update = int(update)

        self.inner_lr = tf.constant(float(inner_lr), tf.float32)
        self.clip_norm = float(clip_norm)

        self.alpha_ex = float(alpha_ex)
        self.gate_l2 = float(gate_l2)

        self.loss_mse = keras.losses.MeanSquaredError()
        self.loss_ex = make_exloss_with_fixed_quantiles(q10, q90, scale=ex_scale) if (q10 is not None and q90 is not None) else None

        self.meta_opt = tf.keras.optimizers.Adam(learning_rate=tf.Variable(float(meta_lr), dtype=tf.float32))

        # region indices
        self.region_lon_idx = None
        self.region_lat_idx = None
        if region_idx is not None:
            self.region_lat_idx = tf.constant(np.asarray(region_idx["lat"], dtype=np.int32))
            self.region_lon_idx = tf.constant(np.asarray(region_idx["lon"], dtype=np.int32))

        # ✅ CNN trunk
        self.trunk = _build_cnn_trunk(
            zdim=self.zdim,
            filter1=filter1,
            filter2=filter2,
            drop_rate=drop_rate
        )
        self.gap = GlobalAveragePooling2D(name="gap")

        # heads
        self.head_g = Dense(1, kernel_initializer="glorot_uniform", name="head_global")
        self.head_r = Dense(1, kernel_initializer="glorot_uniform", name="head_regional")

        # gate net
        gate_hidden = max(8, int(filter2) // 2)
        self.gate_fc1 = Dense(gate_hidden, name="gate_fc1")
        self.gate_act = Activation(gelu, name="gate_gelu")
        self.gate_fc2 = Dense(1, name="gate_fc2")
        self.gate_sig = Activation("sigmoid", name="gate_sigmoid")

        # build once
        self._region_fracs = self._compute_region_fracs_from_idx()
        _ = self.forward(tf.zeros([1, self.xdim, self.ydim, self.zdim], dtype=tf.float32), training=False)

        self._use_tf_function = bool(use_tf_function)
        self.freeze_trunk_in_inner = bool(freeze_trunk_in_inner)

    def _inner_vars(self):
        if not self.freeze_trunk_in_inner:
            return self.trainable_variables

        vars_ = []
        vars_ += self.head_g.trainable_variables
        vars_ += self.head_r.trainable_variables
        vars_ += self.gate_fc1.trainable_variables
        vars_ += self.gate_fc2.trainable_variables
        return vars_

    def _compute_region_fracs_from_idx(self):
        if self.region_lon_idx is None or self.region_lat_idx is None:
            return None

        lon_idx = tf.cast(self.region_lon_idx, tf.float32)
        lat_idx = tf.cast(self.region_lat_idx, tf.float32)

        lon_min = tf.reduce_min(lon_idx) / tf.cast(self.xdim, tf.float32)
        lon_max = (tf.reduce_max(lon_idx) + 1.0) / tf.cast(self.xdim, tf.float32)

        lat_min = tf.reduce_min(lat_idx) / tf.cast(self.ydim, tf.float32)
        lat_max = (tf.reduce_max(lat_idx) + 1.0) / tf.cast(self.ydim, tf.float32)

        lon_min = tf.clip_by_value(lon_min, 0.0, 1.0)
        lon_max = tf.clip_by_value(lon_max, 0.0, 1.0)
        lat_min = tf.clip_by_value(lat_min, 0.0, 1.0)
        lat_max = tf.clip_by_value(lat_max, 0.0, 1.0)

        return (lon_min, lon_max, lat_min, lat_max)

    def _crop_feature_by_fracs(self, fmap, fracs):
        if fracs is None:
            return fmap

        lon_min, lon_max, lat_min, lat_max = fracs

        Hf = tf.shape(fmap)[1]
        Wf = tf.shape(fmap)[2]

        h0 = tf.cast(tf.math.floor(lon_min * tf.cast(Hf, tf.float32)), tf.int32)
        h1 = tf.cast(tf.math.ceil(lon_max * tf.cast(Hf, tf.float32)), tf.int32)
        w0 = tf.cast(tf.math.floor(lat_min * tf.cast(Wf, tf.float32)), tf.int32)
        w1 = tf.cast(tf.math.ceil(lat_max * tf.cast(Wf, tf.float32)), tf.int32)

        h0 = tf.clip_by_value(h0, 0, tf.maximum(Hf - 1, 0))
        w0 = tf.clip_by_value(w0, 0, tf.maximum(Wf - 1, 0))
        h1 = tf.clip_by_value(h1, h0 + 1, Hf)
        w1 = tf.clip_by_value(w1, w0 + 1, Wf)

        dh = tf.maximum(1, h1 - h0)
        dw = tf.maximum(1, w1 - w0)

        return tf.slice(fmap, [0, h0, w0, 0], [-1, dh, dw, -1])

    # ----------------------------
    # Public API
    # ----------------------------
    def inner_adapt(self, xs, ys, k_steps=None):
        return self._inner_adapt(xs, ys, k_steps=k_steps)

    def predict_point(self, x):
        return self._predict(x, training_flag=False)

    # ----------------------------
    # utilities
    # ----------------------------
    def set_meta_lr(self, lr_value):
        self.meta_opt.learning_rate.assign(float(lr_value))

    def _predict_all(self, x, training_flag=False):
        return self.forward(x, training=training_flag)

    def forward(self, x, training=False):
        fmap = self.trunk(x, training=training)   # (B,Hf,Wf,C)
        feat_g = self.gap(fmap)

        fmap_r = self._crop_feature_by_fracs(fmap, self._region_fracs)
        feat_r = self.gap(fmap_r)

        pg = self.head_g(feat_g)
        pr = self.head_r(feat_r)

        feat = tf.concat([feat_g, feat_r], axis=-1)
        w = self.gate_fc1(feat)
        w = self.gate_act(w)
        w = self.gate_fc2(w)
        w = self.gate_sig(w)

        p_final = pg + w * (pr - pg)

        return (tf.reshape(p_final, [-1]),
                tf.reshape(pg, [-1]),
                tf.reshape(pr, [-1]),
                tf.reshape(w, [-1]))

    def _loss_total(self, y_true, p_final, w_gate=None):
        y_true = tf.reshape(y_true, [-1])
        p_final = tf.reshape(p_final, [-1])

        loss = self.loss_mse(y_true, p_final)
        if self.loss_ex is not None and self.alpha_ex > 0:
            loss = loss + self.alpha_ex * self.loss_ex(y_true, p_final)

        if self.gate_l2 > 0 and w_gate is not None:
            w_gate = tf.reshape(w_gate, [-1])
            loss = loss + self.gate_l2 * tf.reduce_mean(tf.square(w_gate - 0.5))

        return loss

    def _clone_vars(self):
        return [tf.identity(v) for v in self.trainable_variables]

    def _assign_vars(self, values):
        for v, val in zip(self.trainable_variables, values):
            v.assign(val)

    def _predict(self, x, training_flag=False):
        p_final, _, _, _ = self._predict_all(x, training_flag=training_flag)
        return tf.reshape(p_final, [-1])

    def _clone_selected(self, var_list):
        return [tf.identity(v) for v in var_list]

    def _assign_selected(self, var_list, values):
        for v, val in zip(var_list, values):
            v.assign(val)

    def _inner_adapt(self, xs, ys, k_steps=None):
        if k_steps is None:
            k_steps = self.update
        k_steps = int(k_steps)

        ys = tf.reshape(ys, [-1])
        var_list = self._inner_vars()

        last_loss = None
        for _ in range(k_steps):
            with tf.GradientTape() as tape:
                p, _, _, w = self._predict_all(xs, training_flag=True)
                loss = self._loss_total(ys, p, w_gate=w)

            grads = tape.gradient(loss, var_list)
            grads, _ = tf.clip_by_global_norm(grads, self.clip_norm)

            for var, g in zip(var_list, grads):
                if g is not None:
                    var.assign_sub(self.inner_lr * g)

            last_loss = loss
        return last_loss

    def meta_train_step(self, xs, ys, xq, yq, inner_steps=None):
        if inner_steps is None:
            inner_steps = self.update
        inner_steps = int(inner_steps)

        inner_vars = self._inner_vars()
        theta0_inner = self._clone_selected(inner_vars)

        _ = self._inner_adapt(xs, ys, k_steps=inner_steps)

        with tf.GradientTape() as tape:
            pq, _, _, wq = self._predict_all(xq, training_flag=True)
            qloss = self._loss_total(tf.reshape(yq, [-1]), pq, w_gate=wq)

        grads = tape.gradient(qloss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.clip_norm)

        self._assign_selected(inner_vars, theta0_inner)
        self.meta_opt.apply_gradients(zip(grads, self.trainable_variables))
        return qloss

    # ----------------------------
    # IO
    # ----------------------------
    def save_encoder(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        _ = self.trunk(tf.zeros([1, self.xdim, self.ydim, self.zdim], dtype=tf.float32), training=False)
        self.trunk.save_weights(os.path.join(dir_path, "pretrained_encoder.weights.h5"))
        print("[OK] Saved trunk weights:", os.path.join(dir_path, "pretrained_encoder.weights.h5"))

        self.save_weights(os.path.join(dir_path, "full.weights.h5"))
        print("[OK] Saved FULL model weights:", os.path.join(dir_path, "full.weights.h5"))

    def load_encoder(self, dir_path):
        path = os.path.join(dir_path, "pretrained_encoder.weights.h5")
        if not os.path.exists(path):
            print(f"[WARN] Pretrained path does not exist: {path}")
            return

        _ = self.trunk(tf.zeros([1, self.xdim, self.ydim, self.zdim], dtype=tf.float32), training=False)

        try:
            self.trunk.load_weights(path)
            print("[OK] Loaded trunk weights:", path)
        except Exception as e:
            print("[WARN] trunk load failed; try by_name+skip_mismatch:", e)
            self.trunk.load_weights(path, by_name=True, skip_mismatch=True)
            print("[OK] Loaded trunk weights (by_name/skip_mismatch):", path)

