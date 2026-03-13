import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_anomaly_detrend_normalize(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    meta_train: np.ndarray,
    meta_val: np.ndarray,
    meta_test: np.ndarray,
):
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    meta_all = np.concatenate([meta_train, meta_val, meta_test], axis=0)

    years = meta_all[:, 0]
    months = meta_all[:, 1] 

    X_anom = np.empty_like(X_all)
    
    for m in range(5, 9):
        mask_m = months == m
        mask_clim = (years <= 2005) & (months == m)
        clim = X_all[mask_clim].mean(axis=0)
        X_anom[mask_m] = X_all[mask_m] - clim

    trend_mask = years <= 2005
    trend_years = years[trend_mask]
    
    X_detr = np.empty_like(X_anom)

    for c in range(X_anom.shape[1]):
        for i in range(X_anom.shape[2]):
            for j in range(X_anom.shape[3]):
                y = X_anom[:, c, i, j]
                
                y_tr = y[trend_mask]
                p2, p1, p0 = np.polyfit(trend_years, y_tr, 2) 
                trend = p2 * (years**2) + p1 * years + p0
                X_detr[:, c, i, j] = y - trend
    base = X_detr[trend_mask]
    mu  = base.mean(axis=0)
    std = base.std(axis=0) + 1e-8
    standardized = (X_detr - mu) / std
    X_norm = standardized
    n_tr, n_va = len(X_train), len(X_val)
    X_train_o = X_norm[:n_tr]
    X_val_o = X_norm[n_tr:n_tr + n_va]
    X_test_o = X_norm[n_tr + n_va:]

    return X_train_o.astype(np.float32), X_val_o.astype(np.float32), X_test_o.astype(np.float32)


def save_outputs(X_train, X_val, X_test):
    np.save("summer2024std_tzuv_preproc_train_2nd.npy", X_train)
    np.save("summer2024std_tzuv_preproc_val_2nd.npy", X_val)
    np.save("summer2024std_tzuv_preproc_test_2nd.npy", X_test)
    print("Saved preprocessed:")
    print("  preproc_train_2nd.npy", X_train.shape)
    print("  preproc_val_2nd.npy", X_val.shape)
    print("  preproc_test_2nd.npy", X_test.shape)


def visualize_samples(X: np.ndarray, title: str, idx: int = 0):
    fig, axes = plt.subplots(1, 4, figsize=(12, 6))
    chans = ["T2M", "Z250", "U200", "V850"] 
    for k, ax in enumerate(axes.ravel()):  
        im = ax.imshow(X[idx, k].T, origin="lower", aspect="auto")
        ax.set_title(chans[k])
        fig.colorbar(im, ax=ax, shrink=0.8)
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig("summer_preproc_sample_2nd.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    X_tr = np.load("summer2024_tzuv_raw_daily_train.npy")
    X_va = np.load("summer2024_tzuv_raw_daily_val.npy")
    X_te = np.load("summer2024_tzuv_raw_daily_test.npy")

    M_tr = np.load("summer2024_tzuv_raw_daily_meta_train.npy")
    M_va = np.load("summer2024_tzuv_raw_daily_meta_val.npy")
    M_te = np.load("summer2024_tzuv_raw_daily_meta_test.npy")

    X_tr_o, X_va_o, X_te_o = compute_anomaly_detrend_normalize(
        X_tr, X_va, X_te, M_tr, M_va, M_te
    )

    save_outputs(X_tr_o, X_va_o, X_te_o)
    
    visualize_samples(X_tr_o, "Preprocessed Train Sample", idx=0)



