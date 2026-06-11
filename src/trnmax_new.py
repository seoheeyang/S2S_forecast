import os, sys
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use("TkAgg")  
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

# ============================================================
# Paths / imports
# ============================================================
main_dir = "/dir/MAML_for_climate/"
sys.path.append(os.path.join(main_dir, "src"))

# ============================================================
# Font (Times)
# ============================================================
font_path = "/Data/home/seoheey/.fonts/times.ttf"
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path, size=16)
else:
    font_prop = fm.FontProperties(size=16)

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14
})

# ============================================================
# User settings
# ============================================================
train_iter   = 300
train_update = 3
enss = 30
meta_exp = f"train_itr{train_iter}_update{train_update}"

# Experiments
EXP_NA = "new_z500_na" #NA
EXP_EA = "new_z500_ea" #EA
EXP_GL = "new_z500_gl" #GL


# Colors (match your figure vibe)
COLOR_NA = "red"
COLOR_EA = "green"
COLOR_GL = "blue"
COLOR_GLO = "black"

month = 6
month_name = {6: "june", 7: "july", 8: "augu"}[month]

lab_nc_path = f"../dataset/ysh_lab_{month_name}_dt_19502004_std_2nd.nc"
labvar = "target_anomaly"

mu_path    = os.path.join(main_dir, "dataset", f"mu_{month}_19502004.npy")
std_path   = os.path.join(main_dir, "dataset", f"std_{month}_19502004.npy")
trend_path = os.path.join(main_dir, "dataset", f"trendfull_{month}_19502004_std_2nd.npy")

# Monthly absolute climatology (1950-2004 mean, K)
climat = 291.80363808  # June
# climat = 295.77047906  # July
# climat = 297.38706479  # Aug

# GloSea6 txt (assumed anomaly wrt 1993-2020 already)
glosea_txt = "jun.ano.2005-2024.1-9.txt"
# glosea_txt = "jul.ano.2005-2024.1-9.txt"
# glosea_txt = "aug.ano.2005-2024.1-9.txt"

# Eval period
eval_start_year = 2005
eval_end_year   = 2024

# ============================================================
# Helpers
# ============================================================
def load_scalar_np(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] missing {name}: {path}")
    a = np.load(path)
    a = np.asarray(a).reshape(-1)
    if a.size < 1:
        raise ValueError(f"[ERROR] empty {name}: {path}")
    return float(a[0])

def load_1d_np(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] missing {name}: {path}")
    arr = np.load(path)
    return np.asarray(arr).reshape(-1).astype(np.float64)

def inv_zscore(x, mu, std):
    return np.asarray(x, dtype=np.float64) * float(std) + float(mu)

def load_meta_anom_for_exp(exp_name, tdim, mu, std, climat, trend_full, clim_obs):
    """
    ensemble mean(z) -> restored raw(K) -> anomaly wrt 1993-2020 (using clim_obs)
    """
    em = np.zeros((enss, tdim), dtype=np.float64)
    for e, ens in enumerate(range(1, enss + 1)):
        ip = os.path.join(
            main_dir, "output", exp_name, meta_exp,
            f"ensemble_num0{ens}", f"forecast_0{ens}.nc"
        )
        if not os.path.exists(ip):
            raise FileNotFoundError(f"[ERROR] missing forecast: {ip}")
        with Dataset(ip, "r") as g:
            fvar = np.asarray(g.variables["p"][:]).reshape(-1)
        if fvar.size != tdim:
            raise ValueError(f"[ERROR] len mismatch in {ip}: {fvar.size} != {tdim}")
        em[e] = fvar

    meta_mean_z = np.mean(em, axis=0)  # (tdim,)

    trend_seg = trend_full[55:55+tdim]
    meta_restored = inv_zscore(meta_mean_z, mu, std) + float(climat) + trend_seg
    meta_anom = meta_restored - float(clim_obs)
    return meta_anom

def to_cat(x, q33, q67):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.int32)
    out[x < q33] = -1   # Below-normal
    out[x > q67] =  1   # Above-normal
    return out          # Near-normal: 0

def one_vs_rest(series, target_code):
    return (np.asarray(series) == target_code).astype(int)

def binary_counts(pred01, obs01):
    pred01 = np.asarray(pred01).astype(int)
    obs01  = np.asarray(obs01).astype(int)
    H = int(np.sum((pred01 == 1) & (obs01 == 1)))
    M = int(np.sum((pred01 == 0) & (obs01 == 1)))
    F = int(np.sum((pred01 == 1) & (obs01 == 0)))
    return H, M, F

def pod_sr_for_cat(pred_cat, obs_cat, code):
    H, M, F = binary_counts(one_vs_rest(pred_cat, code),
                            one_vs_rest(obs_cat,  code))
    POD = H / (H + M) if (H + M) > 0 else np.nan
    FAR = F / (H + F) if (H + F) > 0 else np.nan
    SR  = 1.0 - FAR if np.isfinite(FAR) else np.nan
    return POD, SR

def f1_for_cat(pred_cat, obs_cat, code):
    H, M, F = binary_counts(one_vs_rest(pred_cat, code),
                            one_vs_rest(obs_cat,  code))
    denom = (2*H + F + M)
    return (2*H / denom) if denom > 0 else np.nan

# -------------------------
# Performance background
# -------------------------
def draw_performance_background(ax):
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Success Ratio (1 - FAR)", fontproperties=font_prop)
    ax.set_ylabel("Probability of Detection (POD)", fontproperties=font_prop)
    ax.grid(True, ls=":", alpha=0.5)

    sr = np.linspace(0.001, 0.999, 600)

    # CSI curves
    for csi in [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pod = (csi * sr) / (sr + csi*sr - csi)
        pod[pod > 1] = np.nan
        ax.plot(sr, pod, color="black", lw=1.0)
        idx = np.nanargmin(np.abs(sr - 0.85))
        if np.isfinite(pod[idx]):
            ax.text(sr[idx], pod[idx] + 0.02, f"{csi:.1f}",
                    fontsize=14, ha="center", fontproperties=font_prop)

    # Bias curves
    bias_vals = [0.25, 0.5, 0.8, 1.0, 1.3, 1.5, 2, 3, 5, 10]
    for b in bias_vals:
        pod = b * sr
        pod[pod > 1] = np.nan
        ax.plot(sr, pod, color="black", lw=0.8, ls="--")

    for b in bias_vals:
        if b < 1:
            x, y = 1.02, b
            if y <= 1.0:
                ax.text(x, y, f"{b:g}", ha="left", va="center",
                        fontsize=14, fontproperties=font_prop)
        elif b > 1:
            x, y = 1.0/b, 1.02
            if x >= 0:
                ax.text(x, y, f"{b:g}", ha="center", va="bottom",
                        fontsize=14, fontproperties=font_prop)
        else:
            ax.text(1.02, 1.02, "1", ha="left", va="bottom",
                    fontsize=14, fontproperties=font_prop)

    ax.text(1.05, 0.5, "Bias", ha="left", va="center",
            fontsize=14, fontproperties=font_prop,
            rotation=90, transform=ax.transAxes)

# ============================================================
# Load data
# ============================================================
mu_scalar  = load_scalar_np(mu_path, "mu")
std_scalar = load_scalar_np(std_path, "std")
trend_full = load_1d_np(trend_path, "trend_full")

if not os.path.exists(lab_nc_path):
    raise FileNotFoundError(f"[ERROR] missing label nc: {lab_nc_path}")

with Dataset(lab_nc_path, "r") as f:
    # 2005..2024 = [55:]
    lab_z_test = np.asarray(f.variables[labvar][55:]).reshape(-1).astype(np.float64)
    # 1993..2020 = [43:71] baseline
    lab_z_9320 = np.asarray(f.variables[labvar][43:71]).reshape(-1).astype(np.float64)

tdim = lab_z_test.size
years = np.arange(2005, 2005 + tdim)

# baseline mean (1993-2020) in absolute K
lab_raw_9320 = inv_zscore(lab_z_9320, mu_scalar, std_scalar) + float(climat) + trend_full[43:71]
clim_obs = float(np.mean(lab_raw_9320))

# observed anomaly for 2005-2024
lab_raw_test = inv_zscore(lab_z_test, mu_scalar, std_scalar) + float(climat) + trend_full[55:55+tdim]
lab_anom = lab_raw_test - clim_obs

# GloSea6 anomaly (already wrt 1993-2020 assumed)
if os.path.exists(glosea_txt):
    glo_anom_all = np.loadtxt(glosea_txt, dtype=float).squeeze()
    glo_anom_all = np.atleast_1d(glo_anom_all).ravel().astype(np.float64)
else:
    glo_anom_all = None
    print(f"[WARN] TXT not found: {glosea_txt}")

# tercile thresholds from ERA5 baseline (1993-2020)
obs_hist_anom = lab_raw_9320 - clim_obs
q33, q67 = np.quantile(obs_hist_anom, [1/3, 2/3])
print("[TERCILES] q33, q67 =", float(q33), float(q67))

# categorize obs
obs_cat_all = to_cat(lab_anom, q33, q67)

# load models -> categories
na_anom = load_meta_anom_for_exp(EXP_NA, tdim, mu_scalar, std_scalar, climat, trend_full, clim_obs)
ea_anom = load_meta_anom_for_exp(EXP_EA, tdim, mu_scalar, std_scalar, climat, trend_full, clim_obs)
gl_anom = load_meta_anom_for_exp(EXP_GL, tdim, mu_scalar, std_scalar, climat, trend_full, clim_obs)

na_cat_all = to_cat(na_anom, q33, q67)
ea_cat_all = to_cat(ea_anom, q33, q67)
gl_cat_all = to_cat(gl_anom, q33, q67)

if glo_anom_all is not None:
    glo_cat_all = to_cat(glo_anom_all[:tdim], q33, q67)
else:
    glo_cat_all = None

# evaluation slice (2005-2024)
idx1 = np.where(years == eval_start_year)[0][0]
idx2 = np.where(years == eval_end_year)[0][0] + 1

obs_cat = obs_cat_all[idx1:idx2]
na_cat  = na_cat_all[idx1:idx2]
ea_cat  = ea_cat_all[idx1:idx2]
gl_cat  = gl_cat_all[idx1:idx2]

if glo_cat_all is not None:
    glo_cat = glo_cat_all[idx1:min(idx2, len(glo_cat_all))]
else:
    glo_cat = None

# ============================================================
# Plot: (a) Performance diagram, (b) F1-score bars
# ============================================================
marker_map = {-1: "v", 0: "s", 1: "^"}  # Below, Near, Above
msize = 170

# 각 모델별 약간의 오프셋 적용 (4개 모델)
offsets = {
    COLOR_NA:  (0.00,  0.00),    # NA - 기준점
    COLOR_EA:  (0.01,  0.01),    # EA - 약간 우상향
    COLOR_GL:  (-0.01, 0.01),    # GL - 약간 좌상향
    COLOR_GLO: (0.00,  -0.01),   # GloSea6 - 약간 하향
}

def scatter_model(ax, pred_cat, obs_cat, color):
    dx, dy = offsets[color]
    for code in [-1, 0, 1]:
        pod, sr = pod_sr_for_cat(pred_cat, obs_cat, code)
        # NaN 체크 추가
        if np.isfinite(pod) and np.isfinite(sr):
            ax.scatter(
                sr + dx, pod + dy,  # 오프셋 적용!
                s=msize,
                marker=marker_map[code],
                facecolors=color,
                edgecolors="black",
                linewidths=1.0
            )

def annotate_bars(ax):
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h):
            ax.text(p.get_x() + p.get_width()/2, h + 0.02, f"{h:.2f}",
                    ha="center", va="bottom",
                    fontsize=15, fontproperties=font_prop)

fig = plt.figure(figsize=(14.5, 6.5))

# (a) performance diagram
ax1 = fig.add_subplot(1, 2, 1)
draw_performance_background(ax1)

scatter_model(ax1, na_cat, obs_cat, color=COLOR_NA)
scatter_model(ax1, ea_cat, obs_cat, color=COLOR_EA)
scatter_model(ax1, gl_cat, obs_cat, color=COLOR_GL)

if glo_cat is not None and len(glo_cat) > 0:
    scatter_model(ax1, glo_cat, obs_cat[:len(glo_cat)], color=COLOR_GLO)

# Legend 1: Model colors (lower-left)
model_handles = [
    Line2D([0],[0], marker='o', linestyle='None',
           markerfacecolor=COLOR_NA, markeredgecolor='black',
           markersize=13, label="K-TempCast(NA)"),
    Line2D([0],[0], marker='o', linestyle='None',
           markerfacecolor=COLOR_EA, markeredgecolor='black',
           markersize=13, label="K-TempCast(EA)"),
    Line2D([0],[0], marker='o', linestyle='None',
           markerfacecolor=COLOR_GL, markeredgecolor='black',
           markersize=13, label="K-TempCast(GL)"),
    Line2D([0],[0], marker='o', linestyle='None',
           markerfacecolor=COLOR_GLO, markeredgecolor='black',
           markersize=13, label="GloSea6"),
]
leg1 = ax1.legend(handles=model_handles,
                  loc="upper left",
                  frameon=True, framealpha=1.0, edgecolor="black",
                  prop=fm.FontProperties(fname=font_path, size=15))

# Legend 2: Category markers (lower-right)
cat_handles = [
    Line2D([0],[0], marker='v', linestyle='None',
           markerfacecolor='white', markeredgecolor='black',
           markersize=13, label="Below-normal"),
    Line2D([0],[0], marker='s', linestyle='None',
           markerfacecolor='white', markeredgecolor='black',
           markersize=13, label="Near-normal"),
    Line2D([0],[0], marker='^', linestyle='None',
           markerfacecolor='white', markeredgecolor='black',
           markersize=13, label="Above-normal"),
]
leg2 = ax1.legend(handles=cat_handles,
                  loc="lower right",
                  frameon=True, framealpha=1.0, edgecolor="black",
                  prop=fm.FontProperties(fname=font_path, size=15))
ax1.add_artist(leg1)

for lbl in ax1.get_xticklabels() + ax1.get_yticklabels():
    lbl.set_fontproperties(font_prop)

# (b) F1-score bars (NA/EA/GL/GloSea6)
ax2 = fig.add_subplot(1, 2, 2)

cat_names = ["Below-normal","Near-normal", "Above-normal"]
codes = [-1, 0, 1]

f1_na = [f1_for_cat(na_cat, obs_cat, c) for c in codes]
f1_ea = [f1_for_cat(ea_cat, obs_cat, c) for c in codes]
f1_gl = [f1_for_cat(gl_cat, obs_cat, c) for c in codes]

if glo_cat is not None and len(glo_cat) > 0:
    f1_glo = [f1_for_cat(glo_cat, obs_cat[:len(glo_cat)], c) for c in codes]
else:
    f1_glo = [np.nan, np.nan, np.nan]

x = np.arange(len(codes))
w = 0.20

# bars
ax2.bar(x - 1.5*w, f1_na,  width=w, label="K-TempCast(NA)", color=COLOR_NA, edgecolor="black", linewidth=1.2)
ax2.bar(x - 0.5*w, f1_ea,  width=w, label="K-TempCast(EA)", color=COLOR_EA, edgecolor="black", linewidth=1.2)
ax2.bar(x + 0.5*w, f1_gl,  width=w, label="K-TempCast(GL)", color=COLOR_GL, edgecolor="black", linewidth=1.2)
ax2.bar(x + 1.5*w, f1_glo, width=w, label="GloSea6",        color=COLOR_GLO, edgecolor="black", linewidth=1.2)

ax2.set_xticks(x)
ax2.set_xticklabels(cat_names, fontproperties=font_prop)
ax2.set_ylim(0, 1.02)
for label in ax2.get_yticklabels():
    label.set_fontproperties(font_prop)
ax2.set_ylabel("F1-Score", fontproperties=font_prop)
ax2.grid(True, axis="y", ls=":", alpha=0.5)
ax2.legend(loc="lower right", frameon=True, framealpha=1.0, edgecolor="black",
           prop=fm.FontProperties(fname=font_path, size=14))

annotate_bars(ax2)

plt.tight_layout()
plt.show()
