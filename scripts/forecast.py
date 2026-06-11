import os
import sys
import pathlib
import numpy as np
import tensorflow as tf
from netCDF4 import Dataset

CONFIG = {
    "MAIN_DIR": pathlib.Path("/path/MAML_for_climate/"),
    "INPUT_DIR": "dataset/",
    
    "TRAINED_MODEL_PATH": "/path/MAML_for_climate/output/new_z500_na/train_itr300_update3/ensemble_num0ENSEMBLE/",
    
    "START_YEAR": 1950,
    "TRAIN_END_YEAR": 2004,
    "PREDICT_YEARS": [2025, 2026], 
    
    "FILTER1": 64,
    "FILTER2": 128,
    "DROP_RATE": 0.2,
    
    "REGION_LAT_MIN": 0.0,
    "REGION_LAT_MAX": 60.0,
    "REGION_LON_MIN": 260.0,
    "REGION_LON_MAX": 360.0,
    
    "SHOT": 5,
    "TRAIN_UPDATES": 3,
    "INNER_LR": 0.002,
    "CLIP_NORM": 0.5,
    
    "EX_SCALE": 100/81,
    "ALPHA_EX": 1.0,
    "GATE_L2": 0.5,
    
    "SUPPORT_WINDOW": 20,
    "SEED": 42,
}

sys.path.append(str(CONFIG["MAIN_DIR"]))
from Model.KTEMPCAST import KTempCastModel

def setup_gpu():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU Setup Complete.")
    else:
        print("No GPU detected!")

def to_nhwc_lonlat_from_netcdf(inp: np.ndarray) -> np.ndarray:
    x = np.asarray(inp, dtype=np.float32)
    return np.transpose(x, (0, 3, 2, 1))

def _get_lat_lon_vars(nc):
    lat_names = ["lat", "latitude", "nav_lat", "LAT", "Latitude"]
    lon_names = ["lon", "longitude", "nav_lon", "LON", "Longitude"]
    lat = lon = None
    for n in lat_names:
        if n in nc.variables:
            lat = np.array(nc.variables[n][:]).astype(np.float32)
            break
    for n in lon_names:
        if n in nc.variables:
            lon = np.array(nc.variables[n][:]).astype(np.float32)
            break
    if lat is not None and lat.ndim == 2:
        lat = np.unique(lat)
    if lon is not None and lon.ndim == 2:
        lon = np.unique(lon)
    return lat, lon

def compute_region_idx(lat_1d, lon_1d, lat_min, lat_max, lon_min, lon_max):
    lat_1d = np.asarray(lat_1d).reshape(-1)
    lon_1d = np.asarray(lon_1d).reshape(-1)
    lat_mask = (lat_1d >= lat_min) & (lat_1d <= lat_max)
    lon_mask = (lon_1d >= lon_min) & (lon_1d <= lon_max)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError("[ERROR] region idx empty.")
    return {"lat": lat_idx.astype(np.int32), "lon": lon_idx.astype(np.int32)}

def load_climate_data(input_path):
    inp_file = input_path / "ysh_z500_diff_tzuv_inp_may_dt_19502004_std_2nd_19502026.nc"
    
    with Dataset(inp_file, "r") as f:
        inp_raw = f.variables["anomaly"][:]
        lat_1d, lon_1d = _get_lat_lon_vars(f)
        region_idx = compute_region_idx(
            lat_1d, lon_1d,
            CONFIG["REGION_LAT_MIN"], CONFIG["REGION_LAT_MAX"],
            CONFIG["REGION_LON_MIN"], CONFIG["REGION_LON_MAX"]
        )
    
    inp = to_nhwc_lonlat_from_netcdf(inp_raw)
    print(f"[CHECK] inp shape: {inp.shape}")  # (77, lon, lat, 4) - 1950~2026
    
    lab_file = input_path / "ysh_lab_june_dt_19502004_std_2nd_19502025.nc"
    with Dataset(lab_file, "r") as f:
        lab = f.variables["target_anomaly"][:]
    lab = np.array(lab, dtype=np.float32).reshape(-1)
    
    train_len = 55  # 1950-2004
    
    # Quantiles (훈련 기간 기준)
    train_lab = lab[:train_len]
    q10 = np.percentile(train_lab, 10).astype("float32")
    q90 = np.percentile(train_lab, 90).astype("float32")
    q33 = np.percentile(train_lab, 33.3333).astype("float32")
    q66 = np.percentile(train_lab, 66.6667).astype("float32")
    
    return inp, lab, train_len, region_idx, (q10, q90, q33, q66)

def stratified_support_from_pool(y_all, pool_idx, shot, q33, q66):
    y = np.asarray(y_all).reshape(-1)
    pool_idx = np.asarray(pool_idx, dtype=np.int32).reshape(-1)
    
    cold = pool_idx[y[pool_idx] < q33]
    norm = pool_idx[(y[pool_idx] >= q33) & (y[pool_idx] <= q66)]
    warm = pool_idx[y[pool_idx] > q66]
    
    n = shot // 3
    r = shot % 3
    
    def pick(arr, k):
        if arr.size == 0:
            return np.random.choice(pool_idx, k, replace=True)
        return np.random.choice(arr, k, replace=(arr.size < k))
    
    sel = np.concatenate([pick(cold, n), pick(norm, n), pick(warm, n)])
    if r > 0:
        remain = np.array(list(set(pool_idx.tolist()) - set(sel.tolist())), dtype=np.int32)
        if remain.size < r:
            remain = pool_idx
        sel = np.concatenate([sel, np.random.choice(remain, r, replace=False)])
    
    np.random.shuffle(sel)
    return sel.astype(np.int32)

def save_result_to_binary(data, output_path, filename, start_year):
    data = np.asarray(data, dtype=np.float32).reshape(-1)
    nc_path = output_path / f"{filename}.nc"
    
    # Simple NetCDF save
    from netCDF4 import Dataset as NC4Dataset
    with NC4Dataset(nc_path, "w") as nc:
        nc.createDimension("time", len(data))
        var = nc.createVariable("p", "f4", ("time",))
        var[:] = data
    
    print(f"[SAVED] {nc_path}")

def main():
    ENSEMBLE = int(os.environ.get("ENSEMBLE", 1))
    
    np.random.seed(CONFIG["SEED"] * ENSEMBLE)
    tf.random.set_seed(CONFIG["SEED"] * ENSEMBLE)
    
    setup_gpu()

    inp_all, lab_all, train_len, region_idx, qs = load_climate_data(
        CONFIG["MAIN_DIR"] / CONFIG["INPUT_DIR"]
    )
    q10, q90, q33, q66 = qs
    
    _, xdim, ydim, zdim = inp_all.shape
    print(f"[DIMS] xdim(lon)={xdim}, ydim(lat)={ydim}, zdim(C)={zdim}")
    print(f"[QS] q10={q10:.3f}, q33={q33:.3f}, q66={q66:.3f}, q90={q90:.3f}")
    
    model = KTempCastModel(
        shot=CONFIG["SHOT"],
        xdim=xdim, ydim=ydim, zdim=zdim,
        filter1=CONFIG["FILTER1"], filter2=CONFIG["FILTER2"],
        update=CONFIG["TRAIN_UPDATES"],
        drop_rate=CONFIG["DROP_RATE"],
        region_idx=region_idx,
        q10=q10, q90=q90,
        ex_scale=CONFIG["EX_SCALE"],
        alpha_ex=CONFIG["ALPHA_EX"],
        inner_lr=CONFIG["INNER_LR"],
        meta_lr=0.001,
        clip_norm=CONFIG["CLIP_NORM"],
        gate_l2=CONFIG["GATE_L2"],
        use_tf_function=False,
    )
    
    trained_path = CONFIG["TRAINED_MODEL_PATH"].replace("0ENSEMBLE", f"0{ENSEMBLE}")
    weights_file = pathlib.Path(trained_path) / "full.weights.h5"
    
    if not weights_file.exists():
        raise FileNotFoundError(f"[ERROR] Trained weights not found: {weights_file}")
    
    print(f"[LOAD] Loading trained weights from: {weights_file}")
    model.load_weights(str(weights_file))
    print("[OK] Weights loaded!")

    predict_indices = []
    predict_years = []
    for year in CONFIG["PREDICT_YEARS"]:
        idx = year - CONFIG["START_YEAR"]
        if idx < len(inp_all):
            predict_indices.append(idx)
            predict_years.append(year)
        else:
            print(f"[WARN] Year {year} out of range, skipping")
    
    print(f"[PREDICT] Years: {predict_years}")
    
    preds = []
    for idx, year in zip(predict_indices, predict_years):
        # Support pool: 과거 20년
        w = CONFIG["SUPPORT_WINDOW"]
        start = max(0, idx - w)
        pool = np.arange(start, idx, dtype=np.int32)
        
        if pool.size < CONFIG["SHOT"]:
            print(f"[WARN] Not enough support for year {year}, using all available")
            pool = np.arange(0, idx, dtype=np.int32)
        
        # Stratified sampling
        s_idx = stratified_support_from_pool(lab_all, pool, CONFIG["SHOT"], q33, q66)
        
        # Inner adaptation
        theta0 = model._clone_vars()
        _ = model.inner_adapt(
            xs=tf.convert_to_tensor(inp_all[s_idx], tf.float32),
            ys=tf.convert_to_tensor(lab_all[s_idx], tf.float32),
            k_steps=CONFIG["TRAIN_UPDATES"],
        )
        
        p = model.predict_point(
            tf.convert_to_tensor(inp_all[idx:idx+1], tf.float32)
        ).numpy()[0]
        
        model._assign_vars(theta0)
        preds.append(p)
        
        print(f"  Year {year}: pred={p:.4f}, support_pool=[{pool[0]+CONFIG['START_YEAR']}..{pool[-1]+CONFIG['START_YEAR']}] (n={len(pool)})")
    
    preds = np.asarray(preds, dtype=np.float32)

    output_dir = pathlib.Path(trained_path)
    save_result_to_binary(preds, output_dir, f"forecast_2025_2026_ens{ENSEMBLE}", start_year=predict_years[0])
    
    print(f"[DONE] Predictions for {predict_years}: {preds}")

    if 2025 in predict_years:
        idx_2025 = predict_years.index(2025)
        lab_2025 = lab_all[2025 - CONFIG["START_YEAR"]]
        print(f"[2025] Pred={preds[idx_2025]:.4f}, True={lab_2025:.4f}, Error={preds[idx_2025]-lab_2025:.4f}")

if __name__ == "__main__":
    main()
