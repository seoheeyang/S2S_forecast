import numpy as np
import xarray as xr
import xesmf as xe
import os
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
months_needed = [4, 5] 
output_month_name = "may"

variables_info = {
    "t2m": {
        "level": None,
        "template": "/path/ERA5/daily/T2/T2.{year}{month:02d}.nc",
        "var": "T2"
    },
    "z500": {
        "level": 500,
        "template": "/path/ERA5/daily/Z/Z.{year}{month:02d}.nc",
        "var": "Z"
    },
    "u200": {
        "level": 200,
        "template": "/path/ERA5/daily/U/U.{year}{month:02d}.nc",
        "var": "U"
    },
    "v850": {
        "level": 850,
        "template": "/path/ERA5/daily/V/V.{year}{month:02d}.nc",
        "var": "V"
    },
}

years = range(1950, 2025)

# Output grid definition (5.0 degree)
ds_out = xr.Dataset({
    'lat': (['lat'], np.arange(-90, 90.1, 5.0)),
    'lon': (['lon'], np.arange(0, 360, 5.0))
})

input_data_list = []

print("Processing years...")
for year in years:
    year_vars_diff = []
    
    for var_name, info in variables_info.items():
        file_path_apr = info["template"].format(year=year, month=4)
        if not os.path.exists(file_path_apr):
            print(f"File missing: {file_path_apr}")
            break
            
        ds_apr = xr.open_dataset(file_path_apr)
        da_apr = ds_apr[info["var"]].where(ds_apr[info["var"]] != -999)
        
        if info["level"] is not None:
            if "lev" in da_apr.dims:
                # .sel()이 차원을 자동으로 축소하므로 .squeeze() 제거
                da_apr = da_apr.sel(lev=info["level"], method="nearest")
            if "lev" in da_apr.coords:
                da_apr = da_apr.drop_vars("lev") # reset_coords 대신 확실하게 변수 제거

        da_apr_period = da_apr.sel(time=(da_apr.time.dt.day >= 24) & (da_apr.time.dt.day <= 30))
        
        # --- 2. Load May Data (Month 5) ---
        file_path_may = info["template"].format(year=year, month=5)
        if not os.path.exists(file_path_may):
            print(f"File missing: {file_path_may}")
            break
            
        ds_may = xr.open_dataset(file_path_may)
        da_may = ds_may[info["var"]].where(ds_may[info["var"]] != -999)
        
        # [수정 2] May Level Selection
        if info["level"] is not None:
            if "lev" in da_may.dims:
                # .sel()이 차원을 자동으로 축소하므로 .squeeze() 제거
                da_may = da_may.sel(lev=info["level"], method="nearest")
            if "lev" in da_may.coords:
                da_may = da_may.drop_vars("lev")

        da_may_period = da_may.sel(time=(da_may.time.dt.day >= 8) & (da_may.time.dt.day <= 14))
        # --- 3. Regrid both periods ---
        ds_in = da_apr_period.isel(time=0).to_dataset(name=var_name)
        regridder = xe.Regridder(ds_in, ds_out, method="bilinear", periodic=True)
        
        # Regrid and Calculate Means
        da_apr_regrid = regridder(da_apr_period)
        mean_apr = da_apr_regrid.mean(dim="time")
        
        da_may_regrid = regridder(da_may_period)
        mean_may = da_may_regrid.mean(dim="time")
        
        # --- 4. Calculate Difference ---
        diff_val = mean_may - mean_apr
        
        # Slice Latitudes (-10 to 60) and Sort
        diff_val = diff_val.sel(lat=slice(-10, 60)).sortby('lat')
        
        year_vars_diff.append(diff_val)

    # Stack variables for this year
    if len(year_vars_diff) == 4:
        stacked = xr.concat(year_vars_diff, dim="variable")
        input_data_list.append(stacked)

# Concatenate all years along time dimension
input_data = xr.concat(input_data_list, dim="time")
input_data = input_data.assign_coords(
    variable=list(variables_info.keys()),
    time=np.arange(len(input_data.time))
)

# -----------------------------
# Anomaly Calculation
# -----------------------------
# Climatology based on 1950-2004 (indices 0 to 54)
clim = input_data.sel(time=slice(0, 54)).mean(dim="time")
anomaly = input_data - clim

# -----------------------------
# Detrending (Grid-wise, based on 1950-2004)
# -----------------------------
years_all = np.arange(1950, 2025)
trend_idx = np.where((years_all >= 1950) & (years_all <= 2004))[0]
trend_years = years_all[trend_idx]

# Copy anomaly data to preserve structure
detrended_vals = np.empty_like(anomaly.values)

print("Detrending...")
for lev in range(anomaly.shape[1]):
    for i in range(anomaly.shape[2]):
        for j in range(anomaly.shape[3]):
            y = anomaly[:, lev, i, j].values
            # Fit polynomial (degree 2) on reference period
            p2, p1, p0 = np.polyfit(trend_years, y[trend_idx], 2)
            # Calculate trend for all years
            trend = p2 * (years_all**2) + p1 * years_all + p0
            # Remove trend
            detrended_vals[:, lev, i, j] = y - trend

# Create DataArray for detrended data
input_detrended = xr.DataArray(
    detrended_vals,
    dims=anomaly.dims,
    coords=anomaly.coords,
    name="input_detrended"
)

# -----------------------------
# Standardization (Z-score)
# -----------------------------
# Calculate Mean/Std based on 1950-2004 (indices 0 to 54)
base = input_detrended.sel(time=slice(0, 54))
mu  = base.mean(dim="time")
std = base.std(dim="time") + 1e-8
standardized = (input_detrended - mu) / std

# -----------------------------
# Save to NetCDF
# -----------------------------
# Rename dims to match your requested output format
standardized = standardized.rename({"variable": "lev"}).assign_coords(
    time=("time", years_all.astype(np.float64)),
    lev=("lev", np.arange(1, 5, dtype=np.float64)),
    lat=("lat", standardized.lat.values.astype(np.float64)),
    lon=("lon", standardized.lon.values.astype(np.float64))
)

output_ds = xr.Dataset(
    {
        "anomaly": xr.DataArray(
            standardized.data.astype(np.float32),
            dims=["time", "lev", "lat", "lon"],
            coords={
                "time": ("time", years_all.astype(np.float64), {
                    "standard_name": "time",
                    "units": "hours since 1-1-1 00:00:00",
                    "calendar": "standard",
                    "axis": "T"
                }),
                "lev": ("lev", np.arange(1, 5, dtype=np.float64), {"axis": "Z"}),
                "lat": ("lat", standardized.lat.values.astype(np.float64), {
                    "standard_name": "latitude", "units": "degrees_north", "axis": "Y"
                }),
                "lon": ("lon", standardized.lon.values.astype(np.float64), {
                    "standard_name": "longitude", "units": "degrees_east", "axis": "X"
                }),
            },
            attrs={"long_name": "variables", "missing_value": -9.99e+08}
        )
    }
)

encoding = {
    "time": {"dtype": "float64"},
    "lev": {"dtype": "float64"},
    "lat": {"dtype": "float64"},
    "lon": {"dtype": "float64"},
    "anomaly": {"dtype": "float32", "_FillValue": -9.99e+08}
}

fname = f"ysh_z500_diff_tzuv_inp_{output_month_name}_dt_19502004_std_2nd.nc"
output_ds.to_netcdf(fname, format="NETCDF4", encoding=encoding)
print(f"Complete: {fname}")
