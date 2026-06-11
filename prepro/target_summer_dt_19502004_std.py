from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
from scipy import stats

# 설정
data_dir = Path("/path/ERA5/daily/T2")

lat_range = (35, 40)  # Korea (내림차순 slice용)
lon_range = (125, 130)
months = [6, 7, 8]  # June, July, August

# 리그리드 대상 격자 (5도 간격)
target_grid = xr.Dataset({
    "lat": (["lat"], np.arange(-90, 90.1, 5.0)),
    "lon": (["lon"], np.arange(0, 360, 5.0))
})

for month in months:
    temp_maps = []
    dss_years = []
    regridder = None
    years = range(1950, 2025)

    for year in years:
        filename = data_dir / f"T2.{year}{month:02d}.nc"
        try:
            ds = xr.open_dataset(filename)
            da = ds["T2"].where(ds["T2"] != -999)
            ds_in = da.to_dataset(name="T2")

            if regridder is None:
                regridder = xe.Regridder(ds_in, target_grid, method="bilinear", periodic=True)
            da_regrid = regridder(ds_in["T2"])

            da_region = da_regrid.sel(
                lat=slice(lat_range[0], lat_range[1]),
                lon=slice(lon_range[0], lon_range[1])
            )

            weights = np.cos(np.deg2rad(da_region.lat))
            weights.name = "weights"
            regional_mean = da_region.weighted(weights).mean(dim=("lat", "lon"))
            mean_map = regional_mean.mean(dim="time")
            temp_maps.append(mean_map)
            dss_years.append(year)

        except Exception as e:
            print(f"{year} {month} 처리 중 오류 발생: {e}")
            continue

    temp_all = xr.concat(temp_maps, dim="time")
    temp_all["time"]=pd.date_range("1950-01-01", periods=len(dss_years), freq="YS")
    clim = temp_all.sel(time=slice("1950", "2004")).mean("time")
    print("check temp_all.time",temp_all.sel(time=slice("1950", "2004")))
    anomaly = temp_all - clim

    trend_period = anomaly.sel(time=slice("1950", "2004"))
    trend_years = trend_period.time.dt.year.values
    full_years = anomaly.time.dt.year.values

    p2, p1, p0 = np.polyfit(trend_years, trend_period.values, 2)
    trend_full = p2 * (full_years**2) + p1 * full_years + p0
    detrended = anomaly - trend_full
    np.save(f"trendfull_{month}_19502004_std_2nd.npy",trend_full)

    clim_period = detrended.sel(time=slice("1950", "2004"))

    mu  = clim_period.mean().item()
    std = clim_period.std().item() + 1e-8

    standardized = (detrended - mu) / std

    np.save(f"mu_{month}_19502004.npy", mu)
    np.save(f"std_{month}_19502004.npy", std)

    ds_out = standardized.to_dataset(name="target_anomaly")

    print("month:", month,'climatology',clim,mu)

    # 결과 저장
    ds_out["target_anomaly"].attrs = {
        "long_name": "Korea T2M anomaly",
        "units": "1",  # standardized
        "_FillValue": -9.99e8,
        "missing_value": -9.99e8
    }
    ds_out.attrs["title"] = "Korea standardized 2m temperature anomaly"
    ds_out.attrs["reference_period"] = "1950-2004"

    # 저장 파일 이름
    month_name = {6: "june", 7: "july", 8: "augu"}[month]
    out_path = f"ysh_lab_{month_name}_dt_19502004_std_2nd.nc"
    ds_out.to_netcdf(out_path)

    # 최종 확인
    print(f"{month_name.upper()} saved:")

