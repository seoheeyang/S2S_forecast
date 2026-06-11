# drawgl_2025_2026.py
import os, sys
import numpy as np
from netCDF4 import Dataset
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ============================================================
# Paths / imports
# ============================================================
main_dir = "/path/MAML_for_climate/"
sys.path.append(os.path.join(main_dir, "src"))
from utils import correlation, rmse

# ============================================================
# User settings
# ============================================================
enss = 30
train_iter   = 300
train_update = 3
meta_exp = f"train_itr{train_iter}_update{train_update}"

month = 6
month_name = {6: "june", 7: "july", 8: "augu"}[month]

# Experiments
#EXP_NR = "n60cnn_v1_rolling_support_exloss"
EXP_NR = "new_z500_na"
# ============================================================
# Restore params (z-score inverse)
# ============================================================
mu_path    = os.path.join(main_dir, "dataset", f"mu_{month}_19502004.npy")
std_path   = os.path.join(main_dir, "dataset", f"std_{month}_19502004.npy")
trend_path = os.path.join(main_dir, "dataset", f"trendfull_{month}_19502004_std_2nd.npy")

# ============================================================
# Time axis (2026까지 확장)
# ============================================================
START_YEAR = 1950
END_YEAR   = 2026
years_all  = np.arange(START_YEAR, END_YEAR + 1)
TALL       = len(years_all)  # 77

# plot period
Y1 = 2005
Y2 = 2026
i1 = Y1 - START_YEAR
i2 = Y2 - START_YEAR
N_KEEP = (i2 - i1 + 1)  # 22

# metric period (2005-2024)
Y_MET1 = 2005
Y_MET2 = 2024
im1 = Y_MET1 - START_YEAR
im2 = Y_MET2 - START_YEAR
N_MET = im2 - im1 + 1  # 20

# Tercile baseline (1991-2020) - WMO standard #GloSea6 1993-~
TERCILE_Y1 = 1993
TERCILE_Y2 = 2020
it1 = TERCILE_Y1 - START_YEAR
it2 = TERCILE_Y2 - START_YEAR

# GloSea6 split
GLOSEA_SPLIT_YEAR = 2017
txt_path = "jun.ano.2005-2024.1-9.txt"

# ============================================================
# Base climatology (Kelvin)
# ============================================================
CLIM_BASE_K = 291.80363808  # June

# ============================================================
# Helpers
# ============================================================
def load_nc_1d(path, varname):
    with Dataset(path, "r") as f:
        x = np.asarray(f.variables[varname][:]).reshape(-1)
    return x

def load_1d_np(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] missing {name}: {path}")
    arr = np.load(path)
    arr = np.asarray(arr).reshape(-1).astype(np.float64)
    return arr

def load_scalar_np(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] missing {name}: {path}")
    a = np.load(path)
    a = np.asarray(a).reshape(-1)
    if a.size < 1:
        raise ValueError(f"[ERROR] empty {name}: {path}")
    return float(a[0])

def load_model_ens_stats(exp_name, n_keep):
    """Returns ensemble mean and std (z-score space)"""
    em = np.zeros((enss, n_keep), dtype=np.float64)
    for e, ens in enumerate(range(1, enss + 1)):
        ip = os.path.join(
            main_dir, "output", exp_name, meta_exp,
            f"ensemble_num0{ens}", f"forecast_0{ens}.nc"
        )
        if not os.path.exists(ip):
            print(f"[WARN] missing forecast: {ip}")
            continue
        fflat = load_nc_1d(ip, "p")
        if fflat.size < n_keep:
            print(f"[WARN] forecast too short: {fflat.size} < {n_keep} in {ip}")
            continue
        em[e] = fflat[:n_keep]

    ens_mean = np.mean(em, axis=0)
    ens_std  = np.std(em, axis=0, ddof=1)
    return ens_mean, ens_std

def load_2025_2026_pred(exp_name):
    """Load 2025-2026 predictions (z-score)"""
    pred_file = os.path.join(
        main_dir, "output", exp_name, meta_exp,
        "ensemble_num01", "forecast_2025_2026_ens1.nc"
    )
    if not os.path.exists(pred_file):
        print(f"[WARN] Missing 2025-2026 pred: {pred_file}")
        return None
    
    pred_z = load_nc_1d(pred_file, "p")
    if pred_z.size < 2:
        print(f"[WARN] 2025-2026 pred too short: {pred_z.size}")
        return None
    return pred_z[:2]

# ============================================================
# Font (Times)
# ============================================================
font_path = "/Data/home/seoheey/.fonts/times.ttf"
font_prop = None
if not os.path.exists(font_path):
    print(f"[WARN] missing font: {font_path}, using default")
    mpl.rcParams["font.family"] = "serif"
else:
    font_prop = fm.FontProperties(fname=font_path, size=15)
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
    mpl.rcParams["mathtext.fontset"] = "dejavuserif"

# ============================================================
# Load restore parameters
# ============================================================
trend_path_extended = os.path.join(main_dir, "dataset", f"trendfull_{month}_19502004_std_2nd_19502026.npy")
if os.path.exists(trend_path_extended):
    trend_full = load_1d_np(trend_path_extended, "trend_full")
else:
    trend_orig = load_1d_np(trend_path, "trend_full")
    years_orig = np.arange(1950, 2025)
    p = np.polyfit(years_orig[-10:], trend_orig[-10:], 1)
    trend_2025 = np.polyval(p, 2025)
    trend_2026 = np.polyval(p, 2026)
    trend_full = np.concatenate([trend_orig, [trend_2025, trend_2026]])
    print(f"[INFO] Extended trend: 2025={trend_2025:.4f}, 2026={trend_2026:.4f}")

mu_scalar  = load_scalar_np(mu_path,  "mu")
std_scalar = load_scalar_np(std_path, "std")

if trend_full.size < TALL:
    raise ValueError(f"[ERROR] trend_full too short")
trend_full = trend_full[:TALL]

# ============================================================
# Load ERA5 label
# ============================================================
lab_nc = os.path.join(main_dir, "dataset", f"ysh_lab_{month_name}_dt_19502004_std_2nd_19502025.nc")
if not os.path.exists(lab_nc):
    lab_nc = os.path.join(main_dir, "dataset", f"ysh_lab_{month_name}_dt_19502004_std_2nd.nc")

lab_z = load_nc_1d(lab_nc, "target_anomaly")

# 2026년 NaN 추가
if lab_z.size == 76:
    lab_z_all = np.concatenate([lab_z, [np.nan]])
elif lab_z.size == 75:
    lab_z_all = np.concatenate([lab_z, [np.nan, np.nan]])
else:
    lab_z_all = lab_z[:TALL]

# z-score -> raw(K) -> anomaly wrt 1993-2020
era5_raw_all = (lab_z_all * std_scalar + mu_scalar) + trend_full + CLIM_BASE_K

# Climatology: 1993-2020 평균
clim_raw_mean_9320 = float(np.nanmean(era5_raw_all[it1:it2 + 1]))

era5_anom_9320_all = era5_raw_all - clim_raw_mean_9320
era5_anom = era5_anom_9320_all[i1:i2 + 1]
years = np.arange(Y1, Y2 + 1)
x = np.arange(len(years))

# ============================================================
# Tercile thresholds
# ============================================================
tercile_period_anom = era5_anom_9320_all[it1:it2 + 1]  # 1993-2020
T_L = np.percentile(tercile_period_anom, 33.33)  # Lower tercile
T_U = np.percentile(tercile_period_anom, 66.67)  # Upper tercile

print(f"[TERCILE THRESHOLDS (1993-2020)]")
print(f"  Cold (< T_L): < {T_L:.3f}K")
print(f"  Normal (T_L ~ T_U): {T_L:.3f}K ~ {T_U:.3f}K")
print(f"  Warm (> T_U): > {T_U:.3f}K")

def get_tercile_category(anom_value, t_lower, t_upper):
    """
    Tercile categorization based on 1993-2020 climatology
    """
    if np.isnan(anom_value):
        return "No Data", "black"
    elif anom_value > t_upper:
        return "Warm", "red"
    elif anom_value < t_lower:
        return "Cold", "blue"
    else:
        return "Normal", "gray"

# ============================================================
# Load Model predictions
# ============================================================
trend_keep_20 = trend_full[im1:im2 + 1]

def restore_mean_std(mz, sz, trend_keep):
    model_raw = (mz * std_scalar + mu_scalar) + trend_keep + CLIM_BASE_K
    anom_mean = model_raw - clim_raw_mean_9320
    anom_std  = sz * std_scalar
    return anom_mean, anom_std

# 2005-2024
mz_nr, sz_nr = load_model_ens_stats(EXP_NR, n_keep=N_MET)
k_nr_mean_20, k_nr_std_20 = restore_mean_std(
    mz_nr.astype(np.float64), 
    sz_nr.astype(np.float64), 
    trend_keep_20
)

# 2025-2026
pred_2526_z = load_2025_2026_pred(EXP_NR)
if pred_2526_z is not None and pred_2526_z.size == 2:
    trend_2526 = trend_full[75:77]
    pred_2526_raw = (pred_2526_z * std_scalar + mu_scalar) + trend_2526 + CLIM_BASE_K
    pred_2526_anom = pred_2526_raw - clim_raw_mean_9320
else:
    pred_2526_anom = np.array([np.nan, np.nan])
    print("[WARN] No 2025-2026 predictions")

# Combine
k_nr_mean_full = np.concatenate([k_nr_mean_20, pred_2526_anom])

# ============================================================
# Tercile categorization (2025, 2026)
# ============================================================
pred_2025 = pred_2526_anom[0]
pred_2026 = pred_2526_anom[1]

status_2025, color_2025 = get_tercile_category(pred_2025, T_L, T_U)
status_2026, color_2026 = get_tercile_category(pred_2026, T_L, T_U)

print(f"[2025] Pred={pred_2025:.3f}K → {status_2025}")
print(f"[2026] Pred={pred_2026:.3f}K → {status_2026}")

# ============================================================
# Load GloSea6
# ============================================================
txt_vals_aligned = None
if os.path.exists(txt_path):
    txt_vals = np.loadtxt(txt_path, dtype=float).squeeze()
    txt_vals_aligned = np.asarray(txt_vals).reshape(-1).astype(np.float64)

# ============================================================
# Metrics (2005-2024)
# ============================================================
era5_anom_met = era5_anom_9320_all[im1:im2 + 1]

def metric_line(name, series):
    acc = correlation(series[:N_MET], era5_anom_met)
    r   = rmse(series[:N_MET], era5_anom_met)[0]
    print(f"[{name}] ACC={acc:.3f} RMSE={r:.3f}")
    return acc, r

acc_nr, rmse_nr = metric_line("K-TempCast(NA)", k_nr_mean_full)

acc_g, rmse_g = None, None
if txt_vals_aligned is not None and txt_vals_aligned.size == N_MET:
    acc_g = correlation(txt_vals_aligned, era5_anom_met)
    rmse_g = rmse(txt_vals_aligned, era5_anom_met)[0]
    print(f"[GloSea6] ACC={acc_g:.3f} RMSE={rmse_g:.3f}")

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(11, 4.5))

# ERA5 (2005-2025)
era5_valid_idx = ~np.isnan(era5_anom)
x_era5 = x[era5_valid_idx]
era5_valid = era5_anom[era5_valid_idx]
ax.plot(x_era5, era5_valid, color="black", marker="o",
        linewidth=2.2, markersize=4, label="ERA5", zorder=10)

# K-TempCast(NA): 2005-2024 (실선)
split_idx = N_MET
ax.plot(x[:split_idx], k_nr_mean_full[:split_idx], 
        color="red", marker="o", linewidth=2.2, markersize=4, 
        linestyle="-", label="K-TempCast(NA)", zorder=5)

# 2025-2026 예측 (점선)
if not np.all(np.isnan(pred_2526_anom)):
    ax.plot(x[split_idx-1:], k_nr_mean_full[split_idx-1:], 
            color="red", marker="^", linewidth=2.2, markersize=5, 
            linestyle="--", label="K-TempCast Forecast", zorder=5)

# 2025년 상태 마커 (삼각형)
if not np.isnan(pred_2025):
    ax.scatter(x[-2], k_nr_mean_full[-2], 
               s=150, marker="*", color=color_2025, 
               edgecolors="black", linewidths=1.5, zorder=14)

# 2026년 상태 마커 (별)
if not np.isnan(pred_2026):
    ax.scatter(x[-1], k_nr_mean_full[-1], 
               s=200, marker="*", color=color_2026, 
               edgecolors="black", linewidths=1.5, zorder=15)

# GloSea6
if txt_vals_aligned is not None and txt_vals_aligned.size == N_MET:
    txt_years = years[:N_MET]
    split_idx_g = int(np.where(txt_years == GLOSEA_SPLIT_YEAR)[0][0]) if GLOSEA_SPLIT_YEAR in txt_years else None

    if split_idx_g is None:
        ax.plot(x[:N_MET], txt_vals_aligned, color="blue", marker="o",
                linewidth=2.0, markersize=4, linestyle="--", label="GloSea6")
    else:
        ax.plot(x[:split_idx_g], txt_vals_aligned[:split_idx_g], 
                color="blue", marker="o", linewidth=2.0, markersize=4, 
                linestyle="-", label="GloSea6 Hindcast")
        ax.plot(x[split_idx_g-1:N_MET], txt_vals_aligned[split_idx_g-1:N_MET], 
                color="mediumblue", marker="o", linewidth=2.0, markersize=4, 
                linestyle="--", label="GloSea6 Forecast")

# Tercile threshold lines
ax.axhline(T_L, color="blue", linestyle=":", linewidth=1.0, alpha=0.6, label=f"T_L={T_L:.2f}K")
ax.axhline(T_U, color="red", linestyle=":", linewidth=1.0, alpha=0.6, label=f"T_U={T_U:.2f}K")
ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.2)

# Ticks / labels
ax.set_xticks(x)
xt = [""] * len(years)
for k in range(0, len(years), 2):
    xt[k] = str(years[k])
ax.set_xticklabels(xt, fontsize=15, fontproperties=font_prop)

ax.set_xlabel("Year", fontsize=15, fontproperties=font_prop)
ax.set_ylabel("Anomaly (K)", fontsize=15, fontproperties=font_prop)

ax.tick_params(axis="y", labelsize=15)
for lab in ax.get_yticklabels():
    lab.set_fontproperties(font_prop)

ax.grid(True, which="major", axis="x", linestyle=":", linewidth=0.7)
ax.set_ylim(-3.0, 3.0)
ax.set_yticks(np.arange(-3.0, 3.5, 1.0))

# ============================================================
# Metrics text
# ============================================================
lines = [
    f"K-TempCast(NA): ACC={acc_nr:.2f}, RMSE={rmse_nr:.2f} (2005-2024)",
]
if acc_g is not None:
    lines.append(f"GloSea6: ACC={acc_g:.2f}, RMSE={rmse_g:.2f} (2005-2024)")

lines.append("Forecast (Tercile: 1993-2020)")
lines.append(f"2025: {pred_2025:+.2f}K → {status_2025}")
lines.append(f"2026: {pred_2026:+.2f}K → {status_2026}")

ax.text(
    0.01, 0.98,
    "\n".join(lines),
    transform=ax.transAxes, fontsize=13, va="top", ha="left",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    fontproperties=font_prop
)

# ============================================================
# Legend
# ============================================================
legend_elements = [
    Line2D([0], [0], color="black", marker="o", linewidth=2.2, 
           markersize=4, label="ERA5"),
    Line2D([0], [0], color="red", marker="o", linewidth=2.2, 
           markersize=4, linestyle="-", label="K-TempCast(NA)"),
    Line2D([0], [0], color="red", marker="^", linewidth=2.2, 
           markersize=5, linestyle="--", label="Forecast"),
]

if txt_vals_aligned is not None:
    legend_elements.append(
        Line2D([0], [0], color="blue", marker="o", linewidth=2.0, 
               markersize=4, label="GloSea6")
    )

# Tercile 설명
#legend_elements.extend([
#    Line2D([0], [0], marker="none", linestyle="none", label=""),
#    Patch(facecolor="red", edgecolor="black", label=f"Warm (>{T_U:.2f}K)"),
#    Patch(facecolor="gray", edgecolor="black", label=f"Normal ({T_L:.2f}~{T_U:.2f}K)"),
#    Patch(facecolor="blue", edgecolor="black", label=f"Cold (<{T_L:.2f}K)"),
#])

ax.legend(
    handles=legend_elements,
    loc="lower right",
    prop=font_prop,
    frameon=True,
    ncol=2,
    columnspacing=1.0,
    handletextpad=0.5,
    framealpha=0.9,
    fontsize=13
)

plt.tight_layout()

out_path = "KTempCast_NA_2025_2026_Forecast_Tercile.tiff"
plt.savefig(
    out_path,
    format="tiff",
    dpi=600,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)
print(f"[SAVED] {out_path}")

plt.show()
plt.close()
