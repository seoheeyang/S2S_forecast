import os
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import xesmf as xe


def build_variables_info():
    return {
        "t2m": {
            "level": None,
            "template": "/path/ERA5/4xdaily/T2/T2.{year}{month:02d}.nc",
            "var": "T2",
        },
        "z500": {
            "level": 500,
            "template": "/path/ERA5/4xdaily/Z/Z.{year}{month:02d}.nc",
            "var": "Z",
        },
        "u200": {
            "level": 200,
            "template": "/path/ERA5/4xdaily/U/U.{year}{month:02d}.nc",
            "var": "U",
        },
        "v850": {
            "level": 850,
            "template": "/path/ERA5/4xdaily/V/V.{year}{month:02d}.nc",
            "var": "V",
        },
    }


def build_target_grid():
    return xr.Dataset(
        {
            "lat": (["lat"], np.arange(-90, 90.1, 5.0)),
            "lon": (["lon"], np.arange(0, 360, 5.0)),
        }
    )


def open_and_regrid_month(var_name, info, year, month, ds_out):
    file_path = info["template"].format(year=year, month=month)
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    ds = xr.open_dataset(file_path)
    da: xr.DataArray = ds[info["var"]]
    da = da.where(da != -999)
    
    # Level selection (if needed)
    level = info.get("level")
    if level is not None:
        if "lev" in da.dims: 
            da = da.sel(lev=level, method="nearest")
            if "lev" in da.dims: 
                da = da.squeeze("lev", drop=True) 
        if "lev" in da.coords:
            da = da.reset_coords("lev", drop=True)
    
    # 해당 월 데이터만 선택
    if "time" in da.dims:
        da = da.sel(time=da.time.dt.month == month)

    # Regridding
    ds_in = da.to_dataset(name=var_name)
    regridder = xe.Regridder(ds_in, ds_out, method="bilinear", periodic=True)
    
    da_rg = regridder(ds_in[var_name])
    # Latitude slice: -10 to 60
    da_rg = da_rg.sel(lat=slice(-10, 60)).sortby("lat")
    return da_rg


def extract_diff_samples(years, months, variables_info, time_lag=14):
    ds_out = build_target_grid()
    samples: List[np.ndarray] = []
    meta: List[Tuple[int, int, int]] = []
    
    for year in years:
        for month in months:
            print(f"[{year}-{month:02d}] ", end="")
            
            # 현재 월 데이터 로드
            per_var_current = {}
            try:
                for var_name, info in variables_info.items():
                    da = open_and_regrid_month(var_name, info, year, month, ds_out)
                    per_var_current[var_name] = da
            except FileNotFoundError as e:
                print(f"SKIP (missing current month: {e})")
                continue

            per_var_next = {}
            next_month = month + 1 if month < 12 else 1
            next_year = year if month < 12 else year + 1
            
            try:
                for var_name, info in variables_info.items():
                    da = open_and_regrid_month(var_name, info, next_year, next_month, ds_out)
                    per_var_next[var_name] = da
            except FileNotFoundError:
                per_var_next = {}
            
            per_var_all = {}
            for var_name in ["t2m", "z500", "u200", "v850"]:
                if var_name in per_var_next:
                    per_var_all[var_name] = xr.concat(
                        [per_var_current[var_name], per_var_next[var_name]], 
                        dim="time"
                    )
                else:
                    per_var_all[var_name] = per_var_current[var_name]
            
            times_all = per_var_all['t2m'].time.values
            
            n_matched = 0
            for i, t_curr in enumerate(times_all):

                month_i = t_curr.astype('datetime64[M]').astype(int) % 12 + 1
                if month_i != month:
                    continue

                t_target = t_curr + np.timedelta64(time_lag, 'D')
                
                matches = np.where(times_all == t_target)[0]
                
                if len(matches) > 0:
                    idx_future = matches[0]
                    
                    diffs = []
                    for var_name in ["t2m", "z500", "u200", "v850"]:
                        val_curr = per_var_all[var_name].isel(time=i).values
                        val_fut = per_var_all[var_name].isel(time=idx_future).values
                        
                        diff = val_fut - val_curr
                        diffs.append(diff)
                    
                    stacked = np.stack(diffs, axis=0).astype(np.float32)
                    samples.append(stacked)
                    
                    dt_curr = t_curr.astype('datetime64[D]').astype(object)
                    meta.append((year, month, dt_curr.day))
                    n_matched += 1
            
            print(f"OK ({n_matched} samples)")
    
    X = np.asarray(samples, dtype=np.float32)
    meta_arr = np.asarray(meta, dtype=np.int32)
    
    print(f"\n[Total] Extracted {len(X)} difference samples")
    return X, meta_arr


def split_and_save(X, meta):
    """데이터를 Train/Val/Test로 분할하여 저장"""
    years = meta[:, 0]

    train_mask = years <= 2004
    val_mask = (years > 2004) & (years <= 2014)
    test_mask = (years > 2014) & (years <= 2024)

    np.save("summer_tz500uv_diff14d_train.npy", X[train_mask])
    np.save("summer_tz500uv_diff14d_val.npy", X[val_mask])
    np.save("summer_tz500uv_diff14d_test.npy", X[test_mask])

    np.save("summer_tz500uv_diff14d_meta_train.npy", meta[train_mask])
    np.save("summer_tz500uv_diff14d_meta_val.npy", meta[val_mask])
    np.save("summer_tz500uv_diff14d_meta_test.npy", meta[test_mask])

    print("\n[Saved Files]:")
    print(f"  summer_tz500uv_diff14d_train.npy: {X[train_mask].shape}")
    print(f"  summer_tz500uv_diff14d_val.npy:   {X[val_mask].shape}")
    print(f"  summer_tz500uv_diff14d_test.npy:  {X[test_mask].shape}")
    print(f"\n[Info] Train period: 1950-2004 (for pretraining)")


if __name__ == "__main__":

    print("=" * 70)
    print(" Step 1: Extract 14-day Difference Samples (for AE Pretraining)")
    print("=" * 70)
    
    months = list(range(4, 9))
    years = list(range(1950, 2025))
    
    variables_info = build_variables_info()

    X, meta = extract_diff_samples(
        years=years,
        months=months,
        variables_info=variables_info,
        time_lag=14
    )
    
    print(f"\n[Collected] Shape: {X.shape}, Meta: {meta.shape}")
    
    split_and_save(X, meta)
    
    print("\n[Complete] Ready for step2_ae_diff_z500.py")
