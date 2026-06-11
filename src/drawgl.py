import os, sys
import numpy as np
from netCDF4 import Dataset
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ============================================================
# Paths / imports
# ============================================================
main_dir = "/path/MAML_for_climate/"
sys.path.append(os.path.join(main_dir, "src"))
from utils import correlation, rmse

# ============================================================
# User settings
# ============================================================
enss = 15
train_iter   = 300
train_update = 3
meta_exp = f"train_itr{train_iter}_update{train_update}"

month = 6
month_name = {6: "june", 7: "july", 8: "augu"}[month]

# Experiments (name)
EXP_NA = "new_z500_na" # NA
EXP_EA = "new_z500_ea" # EA
EXP_GL = "new_z500_gl" # GL

labpath = f"./ysh_lab_{month_name}_dt_19502004_std_2nd.nc"
labvar  = "target_anomaly"

# GloSea6 txt: anomaly wrt 1993-2020
txt_path = "jun.ano.2005-2024.1-9.txt"

# ============================================================
# Restore params (z-score inverse)
# ============================================================
mu_path    = os.path.join(main_dir, "dataset", f"mu_{month}_19502004.npy")
std_path   = os.path.join(main_dir, "dataset", f"std_{month}_19502004.npy")
trend_path = os.path.join(main_dir, "dataset", f"trendfull_{month}_19502004_std_2nd.npy")

# ============================================================
# Time axis
# ============================================================
START_YEAR = 1950
END_YEAR   = 2024
years_all  = np.arange(START_YEAR, END_YEAR + 1)  # 1950..2024
TALL       = len(years_all)  # 75

# plot period
Y1 = 2005
Y2 = 2024
i1 = Y1 - START_YEAR
i2 = Y2 - START_YEAR
N_KEEP = (i2 - i1 + 1)  # 20

# metric period (fair comparison with GloSea6 txt, 2005-2024)
Y_MET1 = 2005
Y_MET2 = 2024
im1 = Y_MET1 - START_YEAR
im2 = Y_MET2 - START_YEAR
N_MET = im2 - im1 + 1  # 20

# anomaly baseline (1993-2020)
CLIM1 = 1993
CLIM2 = 2020
ic1 = CLIM1 - START_YEAR
ic2 = CLIM2 - START_YEAR

# GloSea6 style split: Hindcast <= 2016 (solid), Forecast >= 2017 (dashed)
GLOSEA_SPLIT_YEAR = 2017  # first forecast year

# ============================================================
# Base climatology (1950-2004 mean, Kelvin) - month specific
# ============================================================
CLIM_BASE_K = 291.80363808  # June (target 파일 만들 때 나오는 climatology value (월마다 다름))

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
    """
    Returns both MEAN and STD of the ensemble (in z-score space)
    forecast_0{ens}.nc (variable 'p') from each ensemble folder
    return (mean, std) for first n_keep years (starting at Y1=2005)
    """
    em = np.zeros((enss, n_keep), dtype=np.float64)
    for e, ens in enumerate(range(1, enss + 1)):
        ip = os.path.join(
            main_dir, "output", exp_name, meta_exp,
            f"ensemble_num0{ens}", f"forecast_0{ens}.nc"
        )
        if not os.path.exists(ip):
            raise FileNotFoundError(f"[ERROR] missing forecast: {ip}")
        fflat = load_nc_1d(ip, "p")
        if fflat.size < n_keep:
            raise ValueError(f"[ERROR] forecast too short: {fflat.size} < {n_keep} in {ip}")
        em[e] = fflat[:n_keep]

    ens_mean = np.mean(em, axis=0)
    ens_std  = np.std(em, axis=0, ddof=1)  # sample std
    return ens_mean, ens_std

# ============================================================
# Font (Times) - set globally
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
trend_full = load_1d_np(trend_path, "trend_full")
mu_scalar  = load_scalar_np(mu_path,  "mu")
std_scalar = load_scalar_np(std_path, "std")

if trend_full.size < TALL:
    raise ValueError(f"[ERROR] trend_full too short: len={trend_full.size}, need >= {TALL}")
trend_full = trend_full[:TALL]

if i2 >= TALL or ic2 >= TALL:
    raise ValueError(f"[ERROR] time axis too short: TALL={TALL}, need index i2={i2}, ic2={ic2}")

# ============================================================
# Load ERA5 label (z-score) -> raw(K) -> anomaly wrt 1993-2020
# ============================================================
lab_nc = os.path.join(main_dir, "dataset", labpath)
if not os.path.exists(lab_nc):
    raise FileNotFoundError(f"[ERROR] missing ERA5 label nc: {lab_nc}")

lab_z_all = load_nc_1d(lab_nc, labvar)[:TALL].astype(np.float64)

era5_raw_all = (lab_z_all * std_scalar + mu_scalar) + trend_full + CLIM_BASE_K
clim_raw_mean_9320 = float(np.mean(era5_raw_all[ic1:ic2 + 1]))

era5_anom_9320_all = era5_raw_all - clim_raw_mean_9320
era5_anom = era5_anom_9320_all[i1:i2 + 1]
years = np.arange(Y1, Y2 + 1)
x = np.arange(len(years))

# ============================================================
# Load Models (z-score) -> raw(K) -> anomaly
# ============================================================
trend_keep = trend_full[i1:i2 + 1]

def restore_mean_std(mz, sz, trend_keep):
    """
    mz, sz: mean/std in z-score space (length N_KEEP)
    returns: (anom_mean, anom_std) in Kelvin anomaly wrt 1993-2020
    """
    model_raw = (mz * std_scalar + mu_scalar) + trend_keep + CLIM_BASE_K
    anom_mean = model_raw - clim_raw_mean_9320
    anom_std  = sz * std_scalar  # std only scales
    return anom_mean, anom_std

# NA
mz_na, sz_na = load_model_ens_stats(EXP_NA, n_keep=N_KEEP)
k_na_mean, k_na_std = restore_mean_std(mz_na.astype(np.float64), sz_na.astype(np.float64), trend_keep)

# EA
mz_ea, sz_ea = load_model_ens_stats(EXP_EA, n_keep=N_KEEP)
k_ea_mean, k_ea_std = restore_mean_std(mz_ea.astype(np.float64), sz_ea.astype(np.float64), trend_keep)

# GL
mz_gl, sz_gl = load_model_ens_stats(EXP_GL, n_keep=N_KEEP)
k_gl_mean, k_gl_std = restore_mean_std(mz_gl.astype(np.float64), sz_gl.astype(np.float64), trend_keep)

# ============================================================
# Load GloSea6 txt (already anomaly wrt 1993-2020)
# ============================================================
txt_vals_aligned = None
if os.path.exists(txt_path):
    txt_vals = np.loadtxt(txt_path, dtype=float).squeeze()
    txt_vals_aligned = np.asarray(txt_vals).reshape(-1).astype(np.float64)
else:
    print(f"[WARN] TXT not found: {txt_path}")

# ============================================================
# Metrics (2005-2024 common period)
# ============================================================
era5_anom_met = era5_anom_9320_all[im1:im2 + 1]  # 2005..2024

def metric_line(name, series):
    acc = correlation(series[:N_MET], era5_anom_met)
    r   = rmse(series[:N_MET], era5_anom_met)[0]
    print(f"[{name} vs ERA5] 2005-2024 ACC={acc:.3f} RMSE={r:.3f}")
    return acc, r

acc_ea, rmse_ea = metric_line("K-TempCast(EA)", k_ea_mean)
acc_na, rmse_na = metric_line("K-TempCast(NA)", k_na_mean)
acc_gl, rmse_gl = metric_line("K-TempCast(GL)", k_gl_mean)

acc_g, rmse_g = None, None
if txt_vals_aligned is not None and txt_vals_aligned.size == N_MET:
    acc_g = correlation(txt_vals_aligned, era5_anom_met)
    rmse_g = rmse(txt_vals_aligned, era5_anom_met)[0]
    print(f"[GloSea6 vs ERA5] 2005-2024 ACC={acc_g:.3f} RMSE={rmse_g:.3f}")

# ============================================================
# Year-by-year analysis (Best/Worst 3 years)
# ============================================================
print("\n" + "="*70)
print("Year-by-Year Analysis: K-TempCast(GL) vs ERA5")
print("="*70)

# Calculate absolute error for each year
year_errors = np.abs(k_gl_mean[:N_MET] - era5_anom_met)
years_met = np.arange(Y_MET1, Y_MET2 + 1)

# Create year-error pairs
year_error_pairs = list(zip(years_met, year_errors))

# Sort by error (ascending = best predictions first)
sorted_by_error = sorted(year_error_pairs, key=lambda x: x[1])

# Best 3 years (smallest error)
print("\n[BEST 3 Years - Smallest Absolute Error]")
for rank, (year, error) in enumerate(sorted_by_error[:3], 1):
    idx = year - Y_MET1
    era5_val = era5_anom_met[idx]
    pred_val = k_gl_mean[idx]
    print(f"  {rank}. {int(year)}: Error={error:.3f}K | ERA5={era5_val:.3f}K, Pred={pred_val:.3f}K")

# Worst 3 years (largest error)
print("\n[WORST 3 Years - Largest Absolute Error]")
for rank, (year, error) in enumerate(reversed(sorted_by_error[-3:]), 1):
    idx = year - Y_MET1
    era5_val = era5_anom_met[idx]
    pred_val = k_gl_mean[idx]
    print(f"  {rank}. {int(year)}: Error={error:.3f}K | ERA5={era5_val:.3f}K, Pred={pred_val:.3f}K")

print("="*70 + "\n")

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 4))

# ERA5
ax.plot(x, era5_anom, color="black", marker="o",
        linewidth=2.2, markersize=4, label="ERA5", zorder=10)

# K-TempCast(GL) 
ax.plot(x, k_gl_mean, color="red", marker="o",
        linewidth=2.2, markersize=4, label="K-TempCast(GL)", zorder=5)

# GloSea6 (Blue) - Hindcast/Forecast split
if txt_vals_aligned is not None and txt_vals_aligned.size == N_MET:
    txt_years = years[:N_MET]
    split_idx = int(np.where(txt_years == GLOSEA_SPLIT_YEAR)[0][0]) if GLOSEA_SPLIT_YEAR in txt_years else None

    if split_idx is None:
        ax.plot(x[:N_MET], txt_vals_aligned, color="blue", marker="o",
                linewidth=2.0, markersize=4, linestyle="--", label="GloSea6")
    else:
        ax.plot(x[:split_idx], txt_vals_aligned[:split_idx], color="blue", marker="o",
                linewidth=2.0, markersize=4, linestyle="-", label="GloSea6 Hindcast")
        ax.plot(x[split_idx-1:N_MET], txt_vals_aligned[split_idx-1:N_MET], color="mediumblue", marker="o",
                linewidth=2.0, markersize=4, linestyle="--", label="GloSea6 Forecast")

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

ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
ax.grid(True, which="major", axis="x", linestyle=":", linewidth=0.7)

# Y-axis range
ax.set_ylim(-3.0, 3.0)
ax.set_yticks(np.arange(-3.0, 3.0 + 0.5, 1.0))

# Metrics text
lines = [
    f"K-TempCast(GL): ACC={acc_gl:.2f}, RMSE={rmse_gl:.2f}",
]
if acc_g is not None and rmse_g is not None:
    lines.append(f"GloSea6 : ACC={acc_g:.2f}, RMSE={rmse_g:.2f}")

ax.text(
    0.01, 0.98,
    "\n".join(lines),
    transform=ax.transAxes, fontsize=13, va="top", ha="left",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, edgecolor="none"),
    fontproperties=font_prop
)

# Legend
ax.legend(
    loc="lower right",
    prop=font_prop,
    frameon=False,
    ncol=2,
    columnspacing=1.2,
    handletextpad=0.6
)

plt.tight_layout()

# Save
out_path = "KTempCast_GL_ERA5_GloSea6_June_Shaded.tiff"
plt.savefig(
    out_path,
    format="tiff",
    dpi=600,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"}
)

plt.show()
plt.close()

