"""
2주 차이(difference) 데이터에 대한 전처리 스크립트 (일평균 + 날짜별 그룹화)

처리 흐름:
1. 시간별 데이터(00, 06, 12, 18시) → 일평균
2. 각 날짜(4월 1일, 4월 2일, ...)마다 연도별 시계열 구성
3. 변수별, 격자별로 anomaly/detrend/standardize (기준: 1950-2004)

예시:
  원본: 1950/4/1/00시, 1950/4/1/06시, 1950/4/1/12시, 1950/4/1/18시
  → 평균: 1950/4/1 (일평균)
  
  4월 1일 그룹: [1950년 4/1, 1951년 4/1, ..., 2024년 4/1]
  → 격자별 전처리
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict


def compute_daily_average(X_all, meta_all):
    """
    시간별 데이터를 일평균으로 변환
    
    Parameters:
    -----------
    X_all : (N, 4, 15, 72) 시간별 차이 데이터
    meta_all : (N, 3) 년/월/일 메타데이터
    
    Returns:
    --------
    X_daily : (M, 4, 15, 72) 일평균 데이터
    meta_daily : (M, 3) 일별 메타데이터
    """
    print("\n>>> 일평균 계산 중...")
    
    years = meta_all[:, 0]
    months = meta_all[:, 1]
    days = meta_all[:, 2]
    
    # 일별로 그룹화
    daily_groups = defaultdict(list)  # (year, month, day) -> [indices]
    
    for idx in range(len(X_all)):
        key = (years[idx], months[idx], days[idx])
        daily_groups[key].append(idx)
    
    print(f"   시간별 샘플: {len(X_all)}개")
    print(f"   일별 그룹: {len(daily_groups)}개")
    
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
    
    print(f"   일평균 샘플: {len(X_daily)}개")
    
    return X_daily, meta_daily


def compute_anomaly_detrend_normalize_by_date(
    X_daily,
    meta_daily,
    base_year_end=2004,
):
    """
    날짜별로 그룹화하여 전처리 수행
    
    Parameters:
    -----------
    X_daily : (M, 4, 15, 72) 일평균 차이 데이터
    meta_daily : (M, 3) 년/월/일 메타데이터
    base_year_end : 기준 기간 마지막 연도 (기본값: 2004)
    
    Returns:
    --------
    X_std : (M, 4, 15, 72) 전처리된 데이터
    """
    
    print(f"\n[전처리 방식: 날짜별 그룹화]")
    print(f"[기준 기간: 1950-{base_year_end}]")
    
    years_all = meta_daily[:, 0]
    months_all = meta_daily[:, 1]
    days_all = meta_daily[:, 2]
    
    # [1] 날짜별로 그룹화
    print("\n>>> 2. 날짜별 그룹화 중...")
    date_groups = defaultdict(list)  # (month, day) -> [indices]
    
    for idx in range(len(X_daily)):
        month = months_all[idx]
        day = days_all[idx]
        date_key = (month, day)
        date_groups[date_key].append(idx)
    
    print(f"   총 {len(date_groups)}개 날짜 그룹")
    
    # [2] 각 날짜별로 전처리
    print("\n>>> 3. 날짜별 전처리 중...")
    X_std = np.zeros_like(X_daily, dtype=np.float32)
    
    for date_idx, (date_key, indices) in enumerate(sorted(date_groups.items())):
        month, day = date_key
        
        if (date_idx + 1) % 5 == 0 or date_idx == 0:
            print(f"   처리 중: {date_idx+1}/{len(date_groups)} ({month}월 {day}일)")
        
        # 이 날짜의 모든 연도 샘플
        indices = np.array(indices)
        X_date = X_daily[indices]  # (n_years, 4, 15, 72)
        years_date = years_all[indices]
        
        # 연도순 정렬
        sort_idx = np.argsort(years_date)
        X_date = X_date[sort_idx]
        years_date = years_date[sort_idx]
        indices = indices[sort_idx]
        
        # [2-1] Climatology (1950-base_year_end)
        base_mask = years_date <= base_year_end
        
        if base_mask.sum() < 3:
            # 기준 기간 샘플이 너무 적으면 원본 사용
            print(f"   [경고] {month}월 {day}일: 기준 샘플 {base_mask.sum()}개 (최소 3개 필요)")
            for i, orig_idx in enumerate(indices):
                X_std[orig_idx] = X_date[i]
            continue
        
        clim = X_date[base_mask].mean(axis=0)  # (4, 15, 72)
        
        # [2-2] Anomaly
        X_anom = X_date - clim
        
        # [2-3] Detrend (격자별, 2차 다항식)
        X_detr = np.zeros_like(X_anom)
        
        trend_idx = np.where(years_date <= base_year_end)[0]
        trend_years = years_date[trend_idx]
        
        for c in range(X_anom.shape[1]):  # 4 channels
            for i in range(X_anom.shape[2]):  # 15 lat
                for j in range(X_anom.shape[3]):  # 72 lon
                    y = X_anom[:, c, i, j]
                    
                    # 기준 기간으로 2차 다항식 fitting
                    if len(trend_idx) >= 3:
                        p2, p1, p0 = np.polyfit(trend_years, y[trend_idx], 2)
                        
                        # 모든 연도에 대해 추세 계산
                        trend = p2 * (years_date**2) + p1 * years_date + p0
                        
                        # 추세 제거
                        X_detr[:, c, i, j] = y - trend
                    else:
                        X_detr[:, c, i, j] = y
        
        # [2-4] Standardize (Z-score, 기준: 1950-base_year_end)
        base_data = X_detr[base_mask]
        mu = base_data.mean(axis=0)  # (4, 15, 72)
        std = base_data.std(axis=0) + 1e-8  # (4, 15, 72)
        
        X_date_std = (X_detr - mu) / std
        
        # [2-5] 원래 위치에 저장
        for i, orig_idx in enumerate(indices):
            X_std[orig_idx] = X_date_std[i]
    
    print("   완료!")
    
    return X_std


def visualize_samples(X, title, idx=0):
    """샘플 시각화"""
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
    print(f"[저장] summer_z500_diff14d_preproc_daily_avg.png")


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
    print("=" * 70)
    print(" Step 2: Preprocess (일평균 + 날짜별 그룹화)")
    print("=" * 70)
    
    # [1] Raw 차이 데이터 로드 (시간별)
    print("\n[1] Raw 차이 데이터 로드 중...")
    X_train = np.load("summer_tz500uv_diff14d_train.npy")
    X_val = np.load("summer_tz500uv_diff14d_val.npy")
    X_test = np.load("summer_tz500uv_diff14d_test.npy")
    
    meta_train = np.load("summer_tz500uv_diff14d_meta_train.npy")
    meta_val = np.load("summer_tz500uv_diff14d_meta_val.npy")
    meta_test = np.load("summer_tz500uv_diff14d_meta_test.npy")
    
    print(f"  Train (시간별): {X_train.shape}")
    print(f"  Val (시간별):   {X_val.shape}")
    print(f"  Test (시간별):  {X_test.shape}")
    
    # [2] 전체 데이터 합치기
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    meta_all = np.concatenate([meta_train, meta_val, meta_test], axis=0)
    
    # [3] 일평균 계산
    X_daily, meta_daily = compute_daily_average(X_all, meta_all)
    
    # [4] 전처리 수행 (날짜별 그룹화)
    print("\n[2] 전처리 수행 중...")
    X_std = compute_anomaly_detrend_normalize_by_date(
        X_daily,
        meta_daily,
        base_year_end=2004
    )
    
    # [5] Train/Val/Test 재분할
    print("\n>>> 4. Train/Val/Test 재분할...")
    years_daily = meta_daily[:, 0]
    
    train_mask = years_daily <= 2004
    val_mask = (years_daily > 2004) & (years_daily <= 2014)
    test_mask = (years_daily > 2014) & (years_daily <= 2024)
    
    X_train_std = X_std[train_mask].astype(np.float32)
    X_val_std = X_std[val_mask].astype(np.float32)
    X_test_std = X_std[test_mask].astype(np.float32)
    
    print(f"  Train (일평균): {X_train_std.shape}")
    print(f"  Val (일평균):   {X_val_std.shape}")
    print(f"  Test (일평균):  {X_test_std.shape}")
    
    # [6] 통계 확인
    print_statistics(X_train_std, "Train")
    print_statistics(X_val_std, "Val")
    print_statistics(X_test_std, "Test")
    
    # [7] 저장
    print("\n[3] 전처리된 데이터 저장 중...")
    np.save("summer_tz500uv_diff14d_train_daily.npy", X_train_std)
    np.save("summer_tz500uv_diff14d_val_daily.npy", X_val_std)
    np.save("summer_tz500uv_diff14d_test_daily.npy", X_test_std)
    
    print("\n[저장 완료]")
    print("  -> summer_tz500uv_diff14d_train_daily.npy")
    print("  -> summer_tz500uv_diff14d_val_daily.npy")
    print("  -> summer_tz500uv_diff14d_test_daily.npy")
    
    # [8] 시각화
    print("\n[4] 샘플 시각화 중...")
    if len(X_train_std) > 0:
        visualize_samples(
            X_train_std,
            "Preprocessed 14-day Difference (일평균)",
            idx=0
        )
    
    print("\n" + "=" * 70)
    print(" [완료] 다음 단계:")
    print("   1. ktemp_train_autoencoder_new.py 실행")
    print("\n[처리 요약]")
    print("  1. 시간별 데이터(00,06,12,18시) → 일평균")
    print("  2. 각 날짜(4월 1일, 4월 2일, ...)마다 연도별 시계열 구성")
    print("  3. 변수별, 격자별로 anomaly/detrend/standardize")
    print("  4. 기준: 1950-2004년")
