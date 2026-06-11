import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict


def compute_daily_average(X_all, meta_all):
    years = meta_all[:, 0]
    months = meta_all[:, 1]
    days = meta_all[:, 2]
    
    # 일별로 그룹화
    daily_groups = defaultdict(list)  # (year, month, day) -> [indices]
    
    for idx in range(len(X_all)):
        key = (years[idx], months[idx], days[idx])
        daily_groups[key].append(idx)
    
    # 일평균 계산
    X_daily_list = []
    meta_daily_list = []
    
    for (year, month, day), indices in sorted(daily_groups.items()):
        # 이 날의 모든 시간대 평균
        X_day_mean = X_all[indices].mean(axis=0)  # (4, 15, 72)
        X_daily_list.append(X_day_mean)
        meta_daily_list.append([year, month, day])
    
    X_daily = np.array(X_daily_list, dtype=np.float32)
    meta_daily = np.array(meta_daily_list, dtype=np.int32)
    
    return X_daily, meta_daily


def compute_anomaly_detrend_normalize_by_date(
    X_daily,
    meta_daily,
    base_year_end=2004,
):
    
    years_all = meta_daily[:, 0]
    months_all = meta_daily[:, 1]
    days_all = meta_daily[:, 2]
    
    date_groups = defaultdict(list)  # (month, day) -> [indices]
    
    for idx in range(len(X_daily)):
        month = months_all[idx]
        day = days_all[idx]
        date_key = (month, day)
        date_groups[date_key].append(idx)
    
    X_std = np.zeros_like(X_daily, dtype=np.float32)
    
    for date_idx, (date_key, indices) in enumerate(sorted(date_groups.items())):
        month, day = date_key
        
        if (date_idx + 1) % 5 == 0 or date_idx == 0:
            print(f"   처리 중: {date_idx+1}/{len(date_groups)} ({month}월 {day}일)")
        
        indices = np.array(indices)
        X_date = X_daily[indices]  # (n_years, 4, 15, 72)
        years_date = years_all[indices]
        
        sort_idx = np.argsort(years_date)
        X_date = X_date[sort_idx]
        years_date = years_date[sort_idx]
        indices = indices[sort_idx]
        
        base_mask = years_date <= base_year_end
        
        if base_mask.sum() < 3:
            print(f"   [Warning]")
            for i, orig_idx in enumerate(indices):
                X_std[orig_idx] = X_date[i]
            continue
        
        clim = X_date[base_mask].mean(axis=0)  # (4, 15, 72)
        
        X_anom = X_date - clim
        
        X_detr = np.zeros_like(X_anom)
        
        trend_idx = np.where(years_date <= base_year_end)[0]
        trend_years = years_date[trend_idx]
        
        for c in range(X_anom.shape[1]):  # 4 channels
            for i in range(X_anom.shape[2]):  # 15 lat
                for j in range(X_anom.shape[3]):  # 72 lon
                    y = X_anom[:, c, i, j]
                    
                    if len(trend_idx) >= 3:
                        p2, p1, p0 = np.polyfit(trend_years, y[trend_idx], 2)
                        
                        trend = p2 * (years_date**2) + p1 * years_date + p0
                        X_detr[:, c, i, j] = y - trend
                    else:
                        X_detr[:, c, i, j] = y
        
        base_data = X_detr[base_mask]
        mu = base_data.mean(axis=0)  # (4, 15, 72)
        std = base_data.std(axis=0) + 1e-8  # (4, 15, 72)
        
        X_date_std = (X_detr - mu) / std

        for i, orig_idx in enumerate(indices):
            X_std[orig_idx] = X_date_std[i]

    
    return X_std


def visualize_samples(X, title, idx=0):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    chans = ["T2M", "Z500", "U200", "V850"]
    
    for k, ax in enumerate(axes.ravel()):
        # (4, 15, 72) -> (15, 72) for channel k
        im = ax.imshow(X[idx, k].T, origin="lower", aspect="auto", cmap="RdBu_r")
        ax.set_title(f"{chans[k]} (14-day diff)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Latitude index")
        ax.set_ylabel("Longitude index")
        fig.colorbar(im, ax=ax, shrink=0.8)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("summer_z500_diff14d_preproc_daily_avg.png", dpi=300)
    plt.close(fig)

def print_statistics(X, name):
    """데이터 통계 출력"""
    print(f"\n[{name} 통계]")
    print(f"  Shape: {X.shape}")
    print(f"  Mean:  {X.mean():.6f}")
    print(f"  Std:   {X.std():.6f}")
    print(f"  Min:   {X.min():.6f}")
    print(f"  Max:   {X.max():.6f}")
    print(f"  NaN:   {np.isnan(X).sum()}")


if __name__ == "__main__":
    
    X_train = np.load("summer_tz500uv_diff14d_train.npy")
    X_val = np.load("summer_tz500uv_diff14d_val.npy")
    X_test = np.load("summer_tz500uv_diff14d_test.npy")
    
    meta_train = np.load("summer_tz500uv_diff14d_meta_train.npy")
    meta_val = np.load("summer_tz500uv_diff14d_meta_val.npy")
    meta_test = np.load("summer_tz500uv_diff14d_meta_test.npy")

    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    meta_all = np.concatenate([meta_train, meta_val, meta_test], axis=0)

    X_daily, meta_daily = compute_daily_average(X_all, meta_all)

    X_std = compute_anomaly_detrend_normalize_by_date(
        X_daily,
        meta_daily,
        base_year_end=2004
    )

    years_daily = meta_daily[:, 0]
    
    train_mask = years_daily <= 2004
    val_mask = (years_daily > 2004) & (years_daily <= 2014)
    test_mask = (years_daily > 2014) & (years_daily <= 2024)
    
    X_train_std = X_std[train_mask].astype(np.float32)
    X_val_std = X_std[val_mask].astype(np.float32)
    X_test_std = X_std[test_mask].astype(np.float32)
    
    print_statistics(X_train_std, "Train")
    print_statistics(X_val_std, "Val")
    print_statistics(X_test_std, "Test")
    
    np.save("summer_tz500uv_diff14d_train_daily.npy", X_train_std)
    np.save("summer_tz500uv_diff14d_val_daily.npy", X_val_std)
    np.save("summer_tz500uv_diff14d_test_daily.npy", X_test_std)
    
    print("  -> summer_tz500uv_diff14d_train_daily.npy")
    print("  -> summer_tz500uv_diff14d_val_daily.npy")
    print("  -> summer_tz500uv_diff14d_test_daily.npy")
    
    if len(X_train_std) > 0:
        visualize_samples(
            X_train_std,
            "Preprocessed 14-day Difference",
            idx=0
        )
