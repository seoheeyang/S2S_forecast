"""
ERA5 기상 데이터에서 일일 샘플을 추출하는 스크립트

이 스크립트는 ERA5 재분석 데이터셋에서 필요한 기상 변수들을 추출하고,
일일 단위로 정리하여 numpy 배열로 저장합니다.

주요 기능:
1. 4가지 기상 변수 추출
2. 다양한 해상도의 데이터를 통일된 그리드로 변환 (regridding)
3. 일일 샘플로 변환하여 학습/검증/테스트 세트로 분할
"""

import os
from typing import Dict, Tuple, List

import numpy as np
import xarray as xr
import xesmf as xe


def build_variables_info() -> Dict[str, Dict[str, object]]:
    return {
        "t2m": {
            "level": None,
            "template": "/ERA5/4xdaily/T2/T2.{year}{month:02d}.nc",
            "var": "T2",
        },
        "z250": {
            "level": 250,
            "template": "/ERA5/4xdaily/Z/Z.{year}{month:02d}.nc",
            "var": "Z",
        },
        "u200": {
            "level": 200,
            "template": "/ERA5/4xdaily/U/U.{year}{month:02d}.nc",
            "var": "U",
        },
        "v850": {
            "level": 850,
            "template": "/ERA5/4xdaily/V/V.{year}{month:02d}.nc",
            "var": "V",
        },
    }


def build_target_grid() -> xr.Dataset:
    return xr.Dataset(
        {
            "lat": (["lat"], np.arange(-90, 90.1, 5.0)),
            "lon": (["lon"], np.arange(0, 360, 5.0)),
        }
    )


def open_and_regrid_month(
    var_name: str,
    info: Dict[str, object],
    year: int,
    month: int,
    ds_out: xr.Dataset,
) -> xr.DataArray:
    file_path = info["template"].format(year=year, month=month)
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    ds = xr.open_dataset(file_path)
    da: xr.DataArray = ds[info["var"]]
    da = da.where(da != -999)
    level = info.get("level")
    if level is not None:
        if "lev" in da.dims: 
            da = da.sel(lev=level, method="nearest")
            if "lev" in da.dims: 
                da = da.squeeze("lev", drop=True) 
        if "lev" in da.coords:
            da = da.reset_coords("lev", drop=True)
    if "time" in da.dims:
        da = da.sel(time=da.time.dt.month == month)

    ds_in = da.to_dataset(name=var_name)
    regridder = xe.Regridder(ds_in, ds_out, method="bilinear", periodic=True)
    
    da_rg = regridder(ds_in[var_name])
    da_rg = da_rg.sel(lat=slice(-10, 60)).sortby("lat")
    return da_rg


def extract_daily_samples(
    years: List[int],
    months: List[int],
    variables_info: Dict[str, Dict[str, object]],
) -> Tuple[np.ndarray, np.ndarray]:
    ds_out = build_target_grid()

    samples: List[np.ndarray] = []
    meta: List[Tuple[int, int, int]] = [] 
    for year in years:
        for month in months:
            print(f"[{year}-{month:02d}] ...", end=" ")
            per_var_daily: Dict[str, xr.DataArray] = {}
            
            try:
                for var_name, info in variables_info.items():
                    per_var_daily[var_name] = open_and_regrid_month(
                        var_name, info, year, month, ds_out
                    )
            except FileNotFoundError as e:
                print(f"SKIP (missing {e})")
                continue

            times = None
            for da in per_var_daily.values():
                if times is None:
                    times = da.time.values if "time" in da.dims else None
                else:
                    if ("time" in da.dims) and (len(da.time) != len(times)):
                        print("SKIP (mismatched time length)")
                        times = None
                        break
            if times is None:
                continue

            n_days = len(times)
            
            stacked = xr.concat(
                [per_var_daily[k] for k in ["t2m", "z250", "u200", "v850"]], 
                dim="variable"
            )
            
            if stacked.dims[0] != "variable":
                stacked = stacked.transpose("variable", ...)

            for d in range(n_days):
                arr = stacked.isel(time=d).values.astype(np.float32)
                samples.append(arr)
                
                day = int(np.datetime64(stacked.time.values[d], "D").astype("datetime64[D]").astype(object).day)
                meta.append((year, month, day))

            print(f"OK ({n_days/4.} days)")

    X = np.asarray(samples, dtype=np.float32) 
    meta_arr = np.asarray(meta, dtype=np.int32)
    return X, meta_arr


def split_and_save(X: np.ndarray, meta: np.ndarray) -> None:
    years = meta[:, 0]

    train_mask = years <= 2004
    val_mask = (years > 2004) & (years <= 2014)  # 2005~2014년
    test_mask = (years > 2014) & (years <= 2024)  # 2015~2024년

    np.save("summer2024_tzuv_raw_daily_train.npy", X[train_mask])
    np.save("summer2024_tzuv_raw_daily_val.npy", X[val_mask])
    np.save("summer2024_tzuv_raw_daily_test.npy", X[test_mask])

    np.save("summer2024_tzuv_raw_daily_meta_train.npy", meta[train_mask])
    np.save("summer2024_tzuv_raw_daily_meta_val.npy", meta[val_mask])
    np.save("summer2024_tzuv_raw_daily_meta_test.npy", meta[test_mask])

    print("Saved:")
    print("  summer_daily_train.npy", X[train_mask].shape)
    print("  summer_daily_val.npy", X[val_mask].shape)
    print("  summer_daily_test.npy", X[test_mask].shape)


if __name__ == "__main__":
    """
    메인 실행 부분: 전체 데이터 추출 파이프라인 실행
    
    실행 순서:
    1. 처리할 연도와 월 리스트 생성
    2. 변수 메타데이터 생성
    3. 일일 샘플 추출
    4. 데이터 분할 및 저장
    """
    months = list(range(1, 13))
    years = list(range(1950, 2025))
    
    variables_info = build_variables_info()

    X, meta = extract_daily_samples(years, months, variables_info)
    print("Collected:", X.shape, meta.shape)
    
    split_and_save(X, meta)



