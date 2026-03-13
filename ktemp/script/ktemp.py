# ==========================================================
import os
import sys
import pathlib
import numpy as np
import tensorflow as tf
from netCDF4 import Dataset

CONFIG = {
    "MAIN_DIR": pathlib.Path("/dir/"),
    "INPUT_DIR": "dataset/",

    "PRETRAIN_PATH": "./cnn_std_2024_filter_64128_pretrained_weights/",

    "START_YEAR": 1950,
    "TRAIN_END_YEAR": 2004,

    "FILTER1": 64,
    "FILTER2": 128,

    "DROP_RATE": 0.2,

    "REGION_LAT_MIN": 0.0,
    "REGION_LAT_MAX": 60.0,
    "REGION_LON_MIN": 260.0,
    "REGION_LON_MAX": 360.0,

    "SHOT": 5,
    "NUM_QUERY": 15,

    "TRAIN_ITER": int("TRAIN_ITERS"),
    "TRAIN_UPDATES": int("TRAIN_UPDATE"),
    "INNER_LR": 0.002,
    "META_LR_START": 0.002,
    "META_LR_END": 0.001,
    "CLIP_NORM": 0.5,

    "EX_SCALE": 100/81,
    "ALPHA_EX": 1.0,
    "GATE_L2": 0.0,

    "SUPPORT_WINDOW": 20,
    "SUPPORT_ONLY_TRAIN": True,

    "USE_CALIBRATION": True,
    "USE_TF_FUNCTION": True,
    "SEED": 42,

    "EXP_NAME": "na_cnn_v1_rolling_support_exloss",
}

CONFIG["OUTPUT_DIR"] = (
    CONFIG["MAIN_DIR"]
    / "output"
    / CONFIG["EXP_NAME"]
    / f"train_itr{CONFIG['TRAIN_ITER']}_update{CONFIG['TRAIN_UPDATES']}"
    / "ensemble_num0ENSEMBLE"
)

sys.path.append(str(CONFIG["MAIN_DIR"]))
from Model.KTEMPCAST import KTempCastModel
from grad.gradcam_analyzer import run_gradcam_analysis

def setup_gpu():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU Setup Complete. Devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    else:
        print("No GPU detected!")

def to_nhwc_lonlat_from_netcdf(inp: np.ndarray) -> np.ndarray:
    x = np.asarray(inp, dtype=np.float32)
    return np.transpose(x, (0, 3, 2, 1))  # (T, lon, lat, C)

def _get_lat_lon_vars(nc):
    lat_names = ["lat", "latitude", "nav_lat", "LAT", "Latitude"]
    lon_names = ["lon", "longitude", "nav_lon", "LON", "Longitude"]
    lat = None
    lon = None
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
    inp_file = input_path / "ysh_diff_tzuv_inp_may_dt_19502004_std_2nd.nc"

    with Dataset(inp_file, "r") as f:
        inp_raw = f.variables["anomaly"][:]
        lat_1d, lon_1d = _get_lat_lon_vars(f)
        region_idx = compute_region_idx(
            lat_1d, lon_1d,
            CONFIG["REGION_LAT_MIN"], CONFIG["REGION_LAT_MAX"],
            CONFIG["REGION_LON_MIN"], CONFIG["REGION_LON_MAX"]
        )
        print(f"[REGION] lat_idx=[{region_idx['lat'][0]}..{region_idx['lat'][-1]}] (n={len(region_idx['lat'])}), "
              f"lon_idx=[{region_idx['lon'][0]}..{region_idx['lon'][-1]}] (n={len(region_idx['lon'])})")

    inp = to_nhwc_lonlat_from_netcdf(inp_raw)
    print(f"[CHECK] inp shape (no lon/lat): {inp.shape}")

    lab_file = input_path / "ysh_lab_june_dt_19502004_std_2nd.nc"
    with Dataset(lab_file, "r") as f:
        lab = f.variables["target_anomaly"][:]
    lab = np.array(lab, dtype=np.float32).reshape(-1)

    train_len = 55
    train_inp, test_inp = inp[:train_len], inp[train_len:]
    train_lab, test_lab = lab[:train_len], lab[train_len:]

    q10 = np.percentile(train_lab, 10).astype("float32")
    q90 = np.percentile(train_lab, 90).astype("float32")
    q33 = np.percentile(train_lab, 33.3333).astype("float32")
    q66 = np.percentile(train_lab, 66.6667).astype("float32")

    data_tuple = (inp, lab, train_inp, train_lab, test_inp, test_lab, train_len)
    qs_tuple = (q10, q90, q33, q66)
    return data_tuple, qs_tuple, region_idx

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

def random_query_from_pool(pool_idx, k, exclude_idx):
    pool_idx = np.asarray(pool_idx, dtype=np.int32).reshape(-1)
    ex = set(np.asarray(exclude_idx, dtype=np.int32).reshape(-1).tolist())
    cand = np.array([i for i in pool_idx.tolist() if i not in ex], dtype=np.int32)
    if cand.size < k:
        return np.random.choice(pool_idx, k, replace=True).astype(np.int32)
    return np.random.choice(cand, k, replace=False).astype(np.int32)

def build_support_pool_for_year_index(target_global_idx, train_len):
    if CONFIG["SUPPORT_ONLY_TRAIN"]:
        pool = np.arange(0, train_len, dtype=np.int32)
    else:
        pool = np.arange(0, target_global_idx, dtype=np.int32)

    w = int(CONFIG["SUPPORT_WINDOW"])
    if w > 0 and pool.size > w:
        pool = pool[-w:]
    return pool

def build_train_rolling_pool(target_train_idx):
    end = int(target_train_idx)
    if end <= 0:
        return None
    w = int(CONFIG["SUPPORT_WINDOW"])
    start = max(0, end - w)
    pool = np.arange(start, end, dtype=np.int32)
    if pool.size == 0:
        return None
    return pool

def save_result_to_binary(data, output_dir: pathlib.Path, filename, start_year=2005):
    data = np.asarray(data, dtype=np.float32).reshape(-1)
    tdim = len(data)
    gdat_path = output_dir / f"{filename}.gdat"
    ctl_path = output_dir / f"{filename}.ctl"
    nc_path = output_dir / f"{filename}.nc"

    data.tofile(str(gdat_path))

    ctl_content = f"""dset ^{filename}.gdat
undef -9.99e+08
xdef   1  linear   0.  1
ydef   1  linear   0.  1
zdef   1  linear 1 1
tdef {tdim}  linear jan{start_year} 1yr
vars   1
p      1   1  variable
ENDVARS
"""
    with open(ctl_path, "w") as f:
        f.write(ctl_content)

    os.system(f"cdo -f nc import_binary {ctl_path} {nc_path}")
    os.system(f"rm -f {ctl_path} {gdat_path}")

def main():
    SEEDD = (CONFIG["SEED"]) * ENSEMBLE
    np.random.seed(SEEDD)
    tf.random.set_seed(SEEDD)

    setup_gpu()
    CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)

    (inp_all, lab_all, tr_inp, tr_lab, ts_inp, ts_lab, train_len), qs, region_idx = load_climate_data(
        CONFIG["MAIN_DIR"] / CONFIG["INPUT_DIR"]
    )
    q10, q90, q33, q66 = qs

    _, xdim, ydim, zdim = tr_inp.shape
    print(f"[DIMS] xdim(lon)={xdim}, ydim(lat)={ydim}, zdim(C)={zdim}")
    print(f"[QS] q10={q10:.3f}, q33={q33:.3f}, q66={q66:.3f}, q90={q90:.3f}")
    print(f"[SPLIT] train_len={train_len}, test_len={len(ts_inp)}")
    print(f"[ROLL] WINDOW={CONFIG['SUPPORT_WINDOW']}  SHOT={CONFIG['SHOT']}  QUERY={CONFIG['NUM_QUERY']}")

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
        meta_lr=CONFIG["META_LR_START"],
        clip_norm=CONFIG["CLIP_NORM"],
        gate_l2=CONFIG["GATE_L2"],
        use_tf_function=CONFIG["USE_TF_FUNCTION"],
    )

    if CONFIG["PRETRAIN_PATH"] and os.path.exists(CONFIG["PRETRAIN_PATH"]):
        print(f"[LOAD] pretrained: {CONFIG['PRETRAIN_PATH']}")
        model.load_encoder(CONFIG["PRETRAIN_PATH"])
    else:
        print("[LOAD] skip pretrained (CNN trunk)")

    print("[TRAIN] Start meta-training + meta_lr decay (rolling support + query from full train)")
    decay_steps = float(CONFIG["TRAIN_ITER"])
    lr0 = float(CONFIG["META_LR_START"])
    lr1 = float(CONFIG["META_LR_END"])
    min_t = max(1, int(CONFIG["SUPPORT_WINDOW"]))

    for itr in range(int(CONFIG["TRAIN_ITER"])):
        frac = float(itr) / max(decay_steps - 1.0, 1.0)
        meta_lr = lr0 * (1.0 - frac) + lr1 * frac
        model.set_meta_lr(meta_lr)

        t = np.random.randint(min_t, train_len)

        pool_s = build_train_rolling_pool(t)
        if pool_s is None or pool_s.size < 2:
            continue

        s_idx = stratified_support_from_pool(tr_lab, pool_s, CONFIG["SHOT"], q33, q66)

        pool_q = np.arange(0, train_len, dtype=np.int32)
        q_idx = random_query_from_pool(pool_q, CONFIG["NUM_QUERY"], exclude_idx=s_idx)

        xs = tf.convert_to_tensor(tr_inp[s_idx], dtype=tf.float32)
        ys = tf.convert_to_tensor(tr_lab[s_idx], dtype=tf.float32)
        xq = tf.convert_to_tensor(tr_inp[q_idx], dtype=tf.float32)
        yq = tf.convert_to_tensor(tr_lab[q_idx], dtype=tf.float32)

        qloss = model.meta_train_step(xs, ys, xq, yq, inner_steps=CONFIG["TRAIN_UPDATES"])

        if (itr + 1) % 100 == 0:
            print(f"itr={itr+1}, qloss={float(qloss.numpy()):.6f}, meta_lr={meta_lr:.3e}")

    model.save_encoder(str(CONFIG["OUTPUT_DIR"]))

    print("[TEST] Predict with rolling support")
    preds = []
    test_start_year = CONFIG["TRAIN_END_YEAR"] + 1

    for i in range(len(ts_inp)):
        year = test_start_year + i
        target_global_idx = train_len + i

        pool = build_support_pool_for_year_index(target_global_idx, train_len=train_len)
        s_idx = stratified_support_from_pool(lab_all, pool, CONFIG["SHOT"], q33, q66)

        theta0 = model._clone_vars()
        _ = model.inner_adapt(
            xs=tf.convert_to_tensor(inp_all[s_idx], tf.float32),
            ys=tf.convert_to_tensor(lab_all[s_idx], tf.float32),
            k_steps=CONFIG["TRAIN_UPDATES"],
        )

        p = model.predict_point(
            tf.convert_to_tensor(inp_all[target_global_idx:target_global_idx+1], tf.float32)
        ).numpy()[0]
        model._assign_vars(theta0)
        preds.append(p)

        if (i + 1) % 5 == 0:
            print(f"  year={year}, support_pool=[{pool[0]}..{pool[-1]}] (n={len(pool)})")

    preds = np.asarray(preds, dtype=np.float32)

    print("std(y_test)   =", float(np.std(ts_lab.astype(np.float32))))
    print("std(pred_test)=", float(np.std(preds)))
    corr = np.corrcoef(preds, ts_lab.astype(np.float32))[0, 1]
    print("corr(test)    =", float(corr))

    save_result_to_binary(preds, CONFIG["OUTPUT_DIR"], "forecast_0ENSEMBLE", start_year=test_start_year)
    print("[DONE] Saved to:", CONFIG["OUTPUT_DIR"])

    q_thr = float(q66)

    run_gradcam_analysis(
        model=model,
        inp_all=inp_all,
        lab_all=lab_all,
        train_len=train_len,
        config=CONFIG,
        q_thr=q_thr,
        save_dir=str(CONFIG["OUTPUT_DIR"]),
        select_by="pred",     # "pred" or "obs"
        target="final",       # "final"/"global"/"regional"/"delta"
        max_cases=None
    )

if __name__ == "__main__":
    main()

