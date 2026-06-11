import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.ktemp_pretrain_autoencoder import PretrainAutoencoder
from matplotlib.gridspec import GridSpec
print("=" * 70)
print("Autoencoder Pretrain (CNN trunk aligned with Model")
print("=" * 70)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print("set_memory_growth failed:", e)

print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

FILTER1 = 64
FILTER2 = 128
DROP_RATE = 0.2

EPOCHS = 100
BATCH_SIZE = 32

SAVE_DIR = "./new_z500_cnn_std_2024_filter_64128_pretrained_weights"
TRAIN_NPY = "./summer_tz500uv_diff14d_train_daily.npy"
VAL_NPY   = "./summer_tz500uv_diff14d_val_daily.npy"
def to_nhwc_7215(x: np.ndarray) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError(f"Expected 4D array, got {x.ndim}D: shape={x.shape}")

    if x.shape[1] == 15 and x.shape[2] == 72 and x.shape[3] == 4:
        return x
    if x.shape[1] == 4 and x.shape[2] == 15 and x.shape[3] == 72:
        return np.transpose(x, (0, 2, 3, 1))

    if x.shape[1] == 4 and x.shape[2] == 72 and x.shape[3] == 15:
        return np.transpose(x, (0, 3, 2, 1))

    raise ValueError(f"Unexpected data shape {x.shape}. Expected lat=15, lon=72, lev=4")

CHANNEL_NAMES = ["T2M", "Z500", "U200", "V850"]

def visualize_reconstruction(ae, data, save_dir, batch_size=32, sample_idx=0, prefix="val"):
    os.makedirs(save_dir, exist_ok=True)

    x = data[sample_idx:sample_idx+1]
    recon = ae.autoencoder.predict(x, batch_size=1, verbose=0)[0]
    x0 = x[0]

    zdim = x0.shape[-1]
    chan_names = CHANNEL_NAMES[:zdim] if zdim <= len(CHANNEL_NAMES) else [f"Ch{c+1}" for c in range(zdim)]

    fig = plt.figure(figsize=(18, 3.2 * zdim))
    gs = GridSpec(zdim, 4, figure=fig, wspace=0.45, hspace=0.35)

    for ch in range(zdim):
        orig = x0[:, :, ch]
        rec  = recon[:, :, ch]
        diff = rec - orig
        abs_diff = np.abs(diff)

        vmin = min(orig.min(), rec.min())
        vmax = max(orig.max(), rec.max())

        ax = fig.add_subplot(gs[ch, 0])
        im = ax.imshow(orig, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        if ch == 0:
            ax.set_title("Original", fontsize=11, fontweight="bold")
        ax.set_ylabel(chan_names[ch], fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = fig.add_subplot(gs[ch, 1])
        im = ax.imshow(rec, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        if ch == 0:
            ax.set_title("Reconstructed", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = fig.add_subplot(gs[ch, 2])
        vmax_diff = np.max(np.abs(diff)) + 1e-8
        im = ax.imshow(diff, origin="lower", aspect="auto", cmap="RdBu_r",
                       vmin=-vmax_diff, vmax=vmax_diff)
        if ch == 0:
            ax.set_title("Residual (Recon - Orig)", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = fig.add_subplot(gs[ch, 3])
        ax.axis("off")
        mse = np.mean((orig - rec) ** 2)
        mae = np.mean(np.abs(orig - rec))
        corr = np.corrcoef(orig.ravel(), rec.ravel())[0, 1] if np.std(orig) > 0 and np.std(rec) > 0 else np.nan

        txt = (
            f"{chan_names[ch]}\n"
            f"{'-'*16}\n"
            f"MSE  : {mse:.5f}\n"
            f"MAE  : {mae:.5f}\n"
            f"Corr : {corr:.3f}\n"
            f"{'-'*16}\n"
            f"Orig range:\n[{orig.min():.2f}, {orig.max():.2f}]"
        )
        ax.text(
            0.05, 0.5, txt,
            fontsize=10, family="monospace",
            va="center",
            bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.9)
        )

    _, xdim, ydim, zdim = data.shape
    fig.suptitle(
        f"Autoencoder Reconstruction Example ({prefix} sample {sample_idx})\n"
        f"Input/Output shape: ({xdim}, {ydim}, {zdim})",
        fontsize=14, fontweight="bold", y=0.995
    )

    out_path = os.path.join(save_dir, "reconstruction_visualization.png")
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out_path}")

def save_reconstruction_summary(ae, train_data, val_data, save_dir, batch_size=32, n_eval=20):
    ntr = min(n_eval, len(train_data))
    nva = min(n_eval, len(val_data))

    tr_x = train_data[:ntr]
    va_x = val_data[:nva]

    tr_rec = ae.autoencoder.predict(tr_x, batch_size=batch_size, verbose=0)
    va_rec = ae.autoencoder.predict(va_x, batch_size=batch_size, verbose=0)

    tr_mse = np.mean((tr_x - tr_rec) ** 2)
    va_mse = np.mean((va_x - va_rec) ** 2)
    tr_mae = np.mean(np.abs(tr_x - tr_rec))
    va_mae = np.mean(np.abs(va_x - va_rec))

    summary_path = os.path.join(save_dir, "reconstruction_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Autoencoder Reconstruction Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Train sample count: {ntr}\n")
        f.write(f"Val sample count:   {nva}\n\n")
        f.write(f"Train MSE: {tr_mse:.6f}\n")
        f.write(f"Train MAE: {tr_mae:.6f}\n")
        f.write(f"Val MSE:   {va_mse:.6f}\n")
        f.write(f"Val MAE:   {va_mae:.6f}\n")

    print(f"[SAVE] {summary_path}")

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\n[1] Load data")
    train_data = np.load(TRAIN_NPY).astype(np.float32)
    val_data   = np.load(VAL_NPY).astype(np.float32)

    train_data = to_nhwc_7215(train_data)
    val_data   = to_nhwc_7215(val_data)

    print("★ 최종 정렬된 Train 데이터 Shape (N, H, W, C):", train_data.shape) 
    
    _, xdim, ydim, zdim = train_data.shape
    print(f"★ 모델 입력 차원 선언 -> xdim(lat): {xdim}, ydim(lon): {ydim}, zdim(lev): {zdim}")

    print("\n[2] Build AE (CNN trunk aligned)")
    ae = PretrainAutoencoder(
        xdim=xdim, ydim=ydim, zdim=zdim,
        filter1=FILTER1, filter2=FILTER2,
        drop_rate=DROP_RATE,
        training=True
    )

    ae.autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="mse",
        metrics=["mae"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]

    print("\n[3] Train")
    history = ae.autoencoder.fit(
        train_data, train_data,
        validation_data=(val_data, val_data),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    print("\n[4] Save trunk weights (Model/INCEPT.py load target)")
    ae.save_encoder_weights(SAVE_DIR)

    print("\n[5] Save loss history")
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], color="k", label="Train Loss", linewidth=2)
    plt.plot(history.history["val_loss"], color="tab:blue", label="Val Loss", linewidth=2)
    plt.title("Pretraining History (CNN trunk)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "history.png"), dpi=600)
    plt.close()

    print("\n[6] Save reconstruction visualization")
    visualize_reconstruction(
        ae=ae,
        data=val_data,
        save_dir=SAVE_DIR,
        batch_size=BATCH_SIZE,
        sample_idx=0,
        prefix="val"
    )

    save_reconstruction_summary(
        ae=ae,
        train_data=train_data,
        val_data=val_data,
        save_dir=SAVE_DIR,
        batch_size=BATCH_SIZE,
        n_eval=20
    )

    print(f"DONE: {SAVE_DIR}/pretrained_encoder.weights.h5")
    print(f"DONE: {SAVE_DIR}/history.png")
    print(f"DONE: {SAVE_DIR}/reconstruction_visualization.png")

if __name__ == "__main__":
    main()

