import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _stratified_support_from_pool(y_all, pool_idx, shot, q33, q66, rng):
    y = np.asarray(y_all).reshape(-1)
    pool_idx = np.asarray(pool_idx, dtype=np.int32).reshape(-1)

    cold = pool_idx[y[pool_idx] < q33]
    norm = pool_idx[(y[pool_idx] >= q33) & (y[pool_idx] <= q66)]
    warm = pool_idx[y[pool_idx] > q66]

    n = shot // 3
    r = shot % 3

    def pick(arr, k):
        if arr.size == 0:
            return rng.choice(pool_idx, k, replace=True)
        return rng.choice(arr, k, replace=(arr.size < k))

    sel = np.concatenate([pick(cold, n), pick(norm, n), pick(warm, n)])
    if r > 0:
        remain = np.array(list(set(pool_idx.tolist()) - set(sel.tolist())), dtype=np.int32)
        if remain.size < r:
            remain = pool_idx
        sel = np.concatenate([sel, rng.choice(remain, r, replace=False)])

    rng.shuffle(sel)
    return sel.astype(np.int32)

def compute_gradcam(model, x_input, target="final"):
    x_input = tf.convert_to_tensor(x_input, dtype=tf.float32)

    with tf.GradientTape() as tape:
        fmap = model.trunk(x_input, training=False)   # (1, Hf, Wf, C)
        tape.watch(fmap)

        feat_g = model.gap(fmap)
        fmap_r = model._crop_feature_by_fracs(fmap, model._region_fracs)
        feat_r = model.gap(fmap_r)

        pg = model.head_g(feat_g)   # (1,1)
        pr = model.head_r(feat_r)   # (1,1)

        feat = tf.concat([feat_g, feat_r], axis=-1)
        w = model.gate_fc1(feat)
        w = model.gate_act(w)
        w = model.gate_fc2(w)
        w = model.gate_sig(w)

        p_final = pg + w * (pr - pg)  # (1,1)

        if target == "final":
            out = tf.reshape(p_final, [-1])[0]
        elif target == "global":
            out = tf.reshape(pg, [-1])[0]
        elif target == "regional":
            out = tf.reshape(pr, [-1])[0]
        elif target == "delta":
            out = tf.reshape(pr - pg, [-1])[0]
        else:
            raise ValueError(f"unknown target={target}")

    grads = tape.gradient(out, fmap)  # (1,Hf,Wf,C)

    # channel weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    # weighted sum
    fmap0 = fmap[0]  # (Hf,Wf,C)
    heatmap = tf.reduce_sum(fmap0 * pooled_grads, axis=-1)  # (Hf,Wf)

    heatmap = tf.maximum(heatmap, 0.0)
    maxv = tf.reduce_max(heatmap)
    heatmap = tf.where(maxv > 0, heatmap / maxv, heatmap)

    extras = {
        "pg": float(tf.reshape(pg, [-1])[0].numpy()),
        "pr": float(tf.reshape(pr, [-1])[0].numpy()),
        "w":  float(tf.reshape(w,  [-1])[0].numpy()),
        "p_final": float(tf.reshape(p_final, [-1])[0].numpy()),
    }
    return heatmap.numpy(), extras

def run_gradcam_analysis(
    model,
    inp_all, lab_all, train_len,
    config,
    q_thr,                  # 예: q90
    save_dir,
    select_by="pred",       # "pred" or "obs"
    target="final",         # "final"/"global"/"regional"/"delta"
    max_cases=None,         # None이면 전부, 아니면 상한
    seed=123
):

    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    inp_all = np.asarray(inp_all)
    lab_all = np.asarray(lab_all).reshape(-1)

    ts_inp = inp_all[train_len:]
    ts_lab = lab_all[train_len:]

    tr_lab = lab_all[:train_len]
    q33 = float(np.percentile(tr_lab, 33.33))
    q66 = float(np.percentile(tr_lab, 66.67))

    test_start_year = int(config["TRAIN_END_YEAR"]) + 1

    accum = None
    n_hit = 0
    years_hit = []

    # support pool base
    for i in range(len(ts_inp)):
        year = test_start_year + i
        target_global_idx = train_len + i

        # support candidate pool
        if config["SUPPORT_ONLY_TRAIN"]:
            pool = np.arange(0, train_len, dtype=np.int32)
        else:
            pool = np.arange(0, target_global_idx, dtype=np.int32)

        w = int(config["SUPPORT_WINDOW"])
        if w > 0 and pool.size > w:
            pool = pool[-w:]

        s_idx = _stratified_support_from_pool(
            y_all=lab_all,
            pool_idx=pool,
            shot=int(config["SHOT"]),
            q33=q33, q66=q66,
            rng=rng
        )

        theta0 = model._clone_vars()
        _ = model.inner_adapt(
            xs=tf.convert_to_tensor(inp_all[s_idx], tf.float32),
            ys=tf.convert_to_tensor(lab_all[s_idx], tf.float32),
            k_steps = int(config.get("TRAIN_UPDATES", config.get("TRAIN_UPDATE", 3)))
        )

        x_target = inp_all[target_global_idx:target_global_idx+1]
        heatmap, extras = compute_gradcam(model, x_target, target=target)

        pred_val = extras["p_final"]
        obs_val = float(ts_lab[i])

        ok = (pred_val <= q_thr) if (select_by == "pred") else (obs_val <= q_thr)

        if ok:
            if accum is None:
                accum = np.zeros_like(heatmap, dtype=np.float64)
            accum += heatmap.astype(np.float64)
            n_hit += 1
            years_hit.append(year)

            if max_cases is not None and n_hit >= int(max_cases):
                model._assign_vars(theta0)
                break

        model._assign_vars(theta0)

    if n_hit <= 0:
        print(f"[WARN] No cases matched: select_by={select_by}, thr={q_thr:.4f}")
        return

    avg = (accum / n_hit).astype(np.float32)

    # save numpy
    np.save(os.path.join(save_dir, f"gradcam_avg_{target}_{select_by}_thr.npy"), avg)
    with open(os.path.join(save_dir, f"gradcam_years_{target}_{select_by}_thr.txt"), "w") as f:
        f.write(",".join([str(y) for y in years_hit]) + "\n")

    # plot
    plt.figure(figsize=(10, 5))
    plt.imshow(avg, aspect="auto", interpolation="bilinear", cmap="jet", origin="lower")
    plt.colorbar(label="Grad-CAM (normalized)")
    plt.title(f"Grad-CAM avg | target={target} | select_by={select_by} | thr={q_thr:.2f} | N={n_hit}")
    plt.xlabel("lat_feature_index")
    plt.ylabel("lon_feature_index")
    out_png = os.path.join(save_dir, f"gradcam_avg_{target}_{select_by}_thr.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[DONE] Grad-CAM saved: {out_png}")
    print(f"[DONE] N_hit={n_hit}, years={years_hit}")

