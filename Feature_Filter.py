import numpy as np
from sklearn.neighbors import KernelDensity
from joblib import Parallel, delayed, cpu_count
import pandas as pd
from skrebate import ReliefF
from dataclasses import dataclass
from typing import Optional
from sklearn.utils import check_random_state
from sklearn.metrics import check_scoring
from sklearn.utils import Bunch
from sklearn.ensemble._bagging import _generate_indices
import math
from threadpoolctl import threadpool_limits
import gc
from sklearn.ensemble import RandomForestClassifier
import shap
import os
from sklearn.model_selection import StratifiedKFold
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, log_loss, balanced_accuracy_score, recall_score, confusion_matrix, classification_report
from sklearn.base import clone

def Approximate_density_1d(data, n):
    
    h1d = 0.9 * n**(-1/5)  
    x = data.to_numpy() if hasattr(data, "to_numpy") else np.asarray(data)
    X = x.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=h1d,
                        metric='euclidean').fit(X)
    return kde.score_samples(X)

def Approximate_density_2d(data_1, data_2, n):
    h2d = n**(-1/6)

    x1 = data_1.to_numpy() if hasattr(data_1, "to_numpy") else np.asarray(data_1)
    x2 = data_2.to_numpy() if hasattr(data_2, "to_numpy") else np.asarray(data_2)
    XY = np.column_stack([x1, x2])
    kde = KernelDensity(kernel='gaussian', bandwidth=h2d, metric='euclidean').fit(XY)
    return kde.score_samples(XY)

def effective_n_jobs(n_jobs):
    if n_jobs is None:
        return 1
    if n_jobs < 0:
        return cpu_count()
    return max(1, min(n_jobs, cpu_count()))

def choose_block_size_for_hxx(p, n_jobs, target_tasks_per_worker = 6, min_bs = 16, max_bs = 64):
    n_workers = effective_n_jobs(n_jobs)

    if n_workers <= 1:
        return max_bs

    bs = int(np.sqrt(p * p / (2 * n_workers * target_tasks_per_worker)))
    bs = max(min_bs, min(max_bs, bs))
    return bs

def _hxx_block(Xv, n_param, i0, i1, j0, j1):
    bi, bj = i1 - i0, j1 - j0
    out = np.empty((bi, bj), dtype=float)

    for a, i in enumerate(range(i0, i1)):
        xi = Xv[:, i]
        for b, j in enumerate(range(j0, j1)):
            if j < i:
                out[a, b] = np.nan
                continue
            xj = Xv[:, j]
            ld2 = Approximate_density_2d(xi, xj, n_param)
            out[a, b] = -np.mean(ld2)

    return i0, i1, j0, j1, out

def Compute_Information(X, y, SU = False, CMI = False, glob = True, adjust = True, eps = 1e-12, block_size = None, n_jobs = -1):
    feats = X.columns
    Xv = X.to_numpy(copy=False)
    n, p = Xv.shape
    
    log_density = np.zeros((n, p))
    if block_size is None:
        block_size = choose_block_size_for_hxx(p, n_jobs)
    if n_jobs is None:
        for idx, i in enumerate(feats):
            log_density[:, idx] = Approximate_density_1d(Xv[:, idx], n)
    else:
        ld_list = Parallel(n_jobs=n_jobs, backend="loky", batch_size=64)(
            delayed(Approximate_density_1d)(Xv[:, idx], n) for idx in range(p)
        )
        log_density = np.column_stack(ld_list)

    Hx = -np.mean(log_density, axis=0)
    Hxx = np.zeros((p, p), dtype = float)

    blocks = []
    for i0 in range(0, p, block_size):
        i1 = min(p, i0 + block_size)
        for j0 in range(i0, p, block_size):   # 只做上三角 blocks
            j1 = min(p, j0 + block_size)
            blocks.append((i0, i1, j0, j1))

    if n_jobs is None:
        res = ( _hxx_block(Xv, n, i0, i1, j0, j1) for (i0,i1,j0,j1) in blocks )
    else:
        res = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1)(
            delayed(_hxx_block)(Xv, n, i0, i1, j0, j1) for (i0, i1, j0, j1) in blocks
            )
    for i0, i1, j0, j1, blk in res:
        for a, i in enumerate(range(i0, i1)):
            for b, j in enumerate(range(j0, j1)):
                if j < i:
                    continue
                hij = blk[a, b]
                Hxx[i, j] = hij
                Hxx[j, i] = hij

    Hxx = 0.5 * (Hxx + Hxx.T)
    MI = Hx[:, None] + Hx[None, :] - Hxx

    classes, counts = np.unique(y, return_counts=True)
    priors = counts / counts.sum()

    Hx_y = np.zeros(p, dtype=float)
    Hy = 0.0
    use_parallel_1d_per_class = (n_jobs is not None) and (len(classes) <= 5) and (p >= 200)
    for cls, p_cls in zip(classes, priors):
        mask = (y == cls)
        Xk_v = Xv[mask]
        n_cls = Xk_v.shape[0]

        cls_log_density = np.zeros((n_cls, p))

        if glob:
            n_param_1d = n
        else:
            n_param_1d = n_cls

        if use_parallel_1d_per_class:
            ld_list = Parallel(n_jobs=n_jobs, backend="loky", batch_size=64)(
                delayed(Approximate_density_1d)(Xk_v[:, idx], n_param_1d) for idx in range(p)
            )
            cls_log_density = np.column_stack(ld_list)
        else:
            for idx in range(p):
                cls_log_density[:, idx] = Approximate_density_1d(Xk_v[:, idx], n_param_1d)

        Hx_cls = -np.mean(cls_log_density, axis=0)
        Hx_y += p_cls * Hx_cls
        Hy += -p_cls*np.log(p_cls + eps)

    target_MI = Hx - Hx_y
    target_MI = np.maximum(target_MI, 0.0)
        
    MI = np.maximum(MI, 0.0)
    if adjust:
        np.fill_diagonal(MI, 0.0)
    else:
        np.fill_diagonal(MI, Hx)
    MI_df = pd.DataFrame(MI, index=feats, columns=feats)
    target_MI_series = pd.Series(target_MI, index=feats)
    
    if SU:
        su_xy = 2*target_MI/(Hx + Hy + eps)
        su_xx = (2.0 * MI) / (Hx[:, None] + Hx[None, :] + 1e-12)
        su_xx = np.clip(su_xx, 0.0, 1.0)
        np.fill_diagonal(su_xx, 1.0)

        su_df = pd.DataFrame(su_xx, index=feats, columns=feats)
        target_su_series = pd.Series(su_xy, index=feats)

    if CMI:
        Hxx_y = np.zeros((p, p), dtype=float)
        for cls, p_cls in zip(classes, priors):
            mask = (y == cls)
            Xk_v = Xv[mask] 
            n_k = Xk_v.shape[0]
            n_param = n if glob else n_k
            if n_jobs is None:
                res = ( _hxx_block(Xk_v, n_param, i0, i1, j0, j1) for (i0,i1,j0,j1) in blocks )
            else:
                res = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1)(
                                delayed(_hxx_block)(Xk_v, n_param, i0, i1, j0, j1) for (i0, i1, j0, j1) in blocks
                                )   

            for i0, i1, j0, j1, blk in res:
                for a, i in enumerate(range(i0, i1)):
                    for b, j in enumerate(range(j0, j1)):
                        if j < i:
                            continue
                        hij = blk[a, b]
                        Hxx_y[i, j] += p_cls * hij
                        Hxx_y[j, i] += p_cls * hij

        Hxx_y = 0.5 * (Hxx_y + Hxx_y.T)
        CMI_mat = Hxx + Hx_y[None, :] - Hx[None, :] - Hxx_y
        np.fill_diagonal(CMI_mat, 0)
        CMI_mat = np.maximum(CMI_mat, 0.0)
        CMI_df = pd.DataFrame(CMI_mat, index=feats, columns=feats)

    if SU and CMI:
        return MI_df, target_MI_series, su_df, target_su_series, CMI_df
    elif SU:
        return MI_df, target_MI_series, su_df, target_su_series, None
    elif CMI:
        return MI_df, target_MI_series, None, None, CMI_df
    else:
        return MI_df, target_MI_series, None, None, None
    
def mRMR(MI_df, target_MI_series, k):
    feats = list(target_MI_series.index)
    MI_df = MI_df.loc[feats, feats]
    MI_mat = MI_df.to_numpy().copy()
    np.fill_diagonal(MI_mat, 0)
    rel = target_MI_series.to_numpy().copy()

    p = len(feats)
    K = min(k, p)

    selected_idx = []
    selected_names = []
    records = []

    first = int(np.argmax(rel))
    selected_idx.append(first)
    selected_names.append(feats[first])

    records.append({
            "step": 1,
            "feature": feats[first],
            "score": np.nan,
            "relevance_I_f_C": rel[first],
            "redundancy_mean_I_f_S": np.nan
        })

    red_sum = MI_mat[:, first].copy()

    for step in range(2, K + 1):
        S_size = len(selected_idx)

        red_mean = red_sum / S_size
        denom = 1.0 + red_mean                      
        score = rel / denom

        score[selected_idx] = -np.inf

        best = int(np.argmax(score))
        selected_idx.append(best)
        selected_names.append(feats[best])

        records.append({
            "step": step,
            "feature": feats[best],
            "score": float(score[best]),
            "relevance_I_f_C": float(rel[best]),
            "redundancy_mean_I_f_S": float(red_mean[best])
        })

        red_sum += MI_mat[:, best]

    history = pd.DataFrame(records)
    return selected_names, history

def fcbf(su_df, target_su_series, k_target = None):
    feats = list(target_su_series.index)
    su_xx = np.asarray(su_df, dtype=float)
    su_xy = np.asarray(target_su_series, dtype=float)
    p = su_xy.shape[0]
    if su_xx.shape != (p, p):
        raise ValueError(f"su_xx must be (p,p). Got {su_xx.shape}, p={p}")
    order = np.argsort(-su_xy) 

    alive = np.ones(p, dtype=bool)
    selected = []

    for i_pos, p_idx in enumerate(order):
        if not alive[p_idx]:
            continue

        selected.append(feats[p_idx])

        for q_idx in order[i_pos + 1:]:
            if not alive[q_idx]:
                continue
            if su_xx[p_idx, q_idx] >= su_xy[q_idx]:
                alive[q_idx] = False

    if k_target is None:
        final = selected
    else:
        k = min(int(k_target), len(selected))
        final = selected[:k]

    return final, None

def cmim(CMI_df, target_MI_series, K, eps = 0.0):

    feats = list(target_MI_series.index)

    CMI = CMI_df.loc[feats, feats]
    rel = target_MI_series.loc[feats]

    score = rel.copy()

    selected = []
    conditioning = []  

    history = []

    for step in range(1, min(K, len(feats)) + 1):
        best = score.drop(index=selected).idxmax()
        best_score = float(score.loc[best])

        selected.append(best)

        col = CMI[best]
        col_max = float(col.max())

        used_as_condition = (col_max > eps)
        if used_as_condition:
            conditioning.append(best)
            score = np.minimum(score, col)

        score.loc[best] = -np.inf

        history.append({
            "step": step,
            "selected": best,
            "score": best_score,
            "relevance": float(rel.loc[best]),
            "used_as_condition": used_as_condition,
            "cmi_col_max": col_max,
            "conditioning_size": len(conditioning),
        })

        remaining = score.drop(index=selected)
        if remaining.empty:
            break
        if float(remaining.max()) <= eps:
            break

    return selected, pd.DataFrame(history)

def jmi(CMI_df, target_MI_series, K, eps = 0.0):
    feats = list(target_MI_series.index)
    CMI = CMI_df.loc[feats, feats]
    rel = target_MI_series.loc[feats]

    p = len(feats)
    K = min(int(K), p)

    selected = []
    records = []

    first = rel.idxmax()
    selected.append(first)

    score = CMI[first].astype(float).copy()
    score.loc[first] = -np.inf

    records.append({
        "step": 1,
        "selected": first,
        "criterion": "max_relevance",
        "relevance_I_Y_X": float(rel.loc[first]),
        "score_selected": float(rel.loc[first]),
        "added_condition": None,
        "added_cmi": np.nan,
        "current_score_max": float(score.replace(-np.inf, np.nan).max()),
    })
    for step in range(2, K + 1):
        best = score.idxmax()
        best_score = float(score.loc[best])

        selected.append(best)

        records.append({
            "step": step,
            "selected": best,
            "criterion": "max_sum_CMI",
            "relevance_I_Y_X": float(rel.loc[best]),
            "score_selected": best_score,
            "added_condition": selected[-2],    
            "added_cmi": float(CMI.loc[best, selected[-2]]),
            "current_score_max": best_score,
        })

        score = score + CMI[best]

        for s in selected:
            score.loc[s] = -np.inf

        remaining = score.drop(index=selected)
        if remaining.empty:
            break
        if float(remaining.max()) <= eps:
            break
    history = pd.DataFrame(records)

    return selected, history

def reliefF(X, y, K):

    rf = ReliefF(
        n_neighbors=10,     
        n_features_to_select=None 
    )

    rf.fit(X.values, y.values)

    scores = rf.feature_importances_
    relieff_series = pd.Series(scores, index=X.columns).sort_values(ascending=False)

    return list(relieff_series.index[:K]), None

def run_selectors_with_matrices(*, X, y, mats, K, methods):
    ranked_lists = {}
    histories = {}
    def _need_mats_for(method_name: str):
        if mats is None:
            raise ValueError(f"{method_name} requires MI/SU/CMI matrices, but mats is None.")
    for m in methods:
        if m == "mRMR":
            _need_mats_for("mRMR")
            feats, hist = mRMR(mats.MI_df, mats.target_MI_series, K)
            ranked_lists["mRMR"] = list(feats)
            histories["mRMR"] = hist

        elif m == "FCBF":
            _need_mats_for("FCBF")
            if mats.SU_df is None or mats.target_SU_series is None:
                raise ValueError("FCBF requires SU_df and target_SU_series.")
            feats, hist = fcbf(mats.SU_df, mats.target_SU_series, K)
            ranked_lists["FCBF"] = list(feats)

        elif m == "CMIM":
            _need_mats_for("CMIM")
            if mats.CMI_df is None:
                raise ValueError("CMIM/JMI requires CMI_df.")
            feats, hist = cmim(mats.CMI_df, mats.target_MI_series, K)
            ranked_lists["CMIM"] = list(feats)
            histories["CMIM"] = hist
        elif m == "JMI":
            _need_mats_for("JMI")
            if mats.CMI_df is None:
                raise ValueError("CMIM/JMI requires CMI_df.")
            feats, hist = jmi(mats.CMI_df, mats.target_MI_series, K)
            ranked_lists["JMI"] = list(feats)
            histories["JMI"] = hist
        elif m == "reliefF":
            feats, hist = reliefF(X, y, K)
            ranked_lists["reliefF"] = list(feats)

        else:
            raise ValueError(f"Unknown method: {m}")

    return ranked_lists, histories

def run_selector(X, y, K, methods=("mRMR", "FCBF", "CMIM", "JMI", "reliefF")):

    @dataclass
    class Matrices:
        MI_df: Optional[pd.DataFrame]
        target_MI_series: Optional[pd.Series]
        SU_df: Optional[pd.DataFrame]
        target_SU_series: Optional[pd.Series]
        CMI_df: Optional[pd.DataFrame]

    methods = list(methods)

    info_methods = {"mRMR", "FCBF", "CMIM", "JMI"}
    need_info = any(m in info_methods for m in methods)

    if need_info:
        MI_df, target_MI_series, su_df, target_su_series, CMI_df = Compute_Information(X, y, SU=("FCBF" in methods), CMI=any(m in methods for m in ["CMIM", "JMI"]))

        mats = Matrices(
            MI_df=MI_df,
            target_MI_series=target_MI_series,
            SU_df=su_df,
            target_SU_series=target_su_series,
            CMI_df=CMI_df
        )

    ranked_lists, histories = run_selectors_with_matrices(X = X, y = y, mats = mats, methods = methods, K = K)

    return ranked_lists, histories

def capped_union_rank(ranked_feature_lists, K_target, missing_rank = "len_plus_one"):

    methods = list(ranked_feature_lists.keys())
    union_features = sorted(set().union(*ranked_feature_lists.values()))
    if not union_features:
        return pd.DataFrame(), pd.DataFrame()

    rank_df = pd.DataFrame(index=union_features)

    for m, feats in ranked_feature_lists.items():
        rmap = {f: i + 1 for i, f in enumerate(feats)}
        if missing_rank == "len_plus_one":
            mr = len(feats) + 1
        else:
            mr = int(missing_rank)

        rank_df[m] = [rmap.get(f, mr) for f in union_features]

    rank_df["rank_sum"] = rank_df[methods].sum(axis=1)
    rank_df["best_rank"] = rank_df[methods].min(axis=1)
    rank_df["worst_rank"] = rank_df[methods].max(axis=1)

    rank_df = rank_df.sort_values(
        by=["rank_sum", "best_rank", "worst_rank"],
        ascending=[True, True, True]
    )

    K = min(K_target, len(rank_df))
    topK = rank_df.iloc[:K]
    if K == len(rank_df):
        return topK.index.tolist(), rank_df

    kth = rank_df.iloc[K - 1][["rank_sum", "best_rank", "worst_rank"]]
    tie_mask = (
        (rank_df["rank_sum"] == kth["rank_sum"]) &
        (rank_df["best_rank"] == kth["best_rank"]) &
        (rank_df["worst_rank"] == kth["worst_rank"])
    )

    selected_df = pd.concat([topK, rank_df[tie_mask]]).drop_duplicates()
    selected_features = selected_df.index.tolist()

    return selected_features, rank_df

def final_selector(X, y, K, methods=("mRMR", "FCBF", "CMIM", "JMI", "reliefF"), K_final = None, missing_rank="len_plus_one"):
    if K_final is None:
        K_final = K

    # mats = None

    ranked_lists, histories = run_selector(X = X, y = y, K = K, methods = methods)
    final_features, rank_df = capped_union_rank(
        ranked_feature_lists=ranked_lists,
        K_target = K_final,
        missing_rank=missing_rank
    )

    return final_features, rank_df, histories, ranked_lists

def get_per_tree_oob_indices(rf, n_samples, n_samples_bootstrap = None):
    if not hasattr(rf, "estimators_"):
        raise ValueError("RandomForest must be fitted before computing OOB indices.")

    if not rf.bootstrap:
        raise ValueError("bootstrap=False: OOB samples do not exist.")

    if n_samples_bootstrap is None:
        n_samples_bootstrap = n_samples

    oob_indices_list = []

    for tree in rf.estimators_:
        sample_indices = _generate_indices(
            random_state=check_random_state(tree.random_state) ,
            bootstrap=True,
            n_population=n_samples,
            n_samples=n_samples_bootstrap,
        )

        inbag = np.zeros(n_samples, dtype=bool)
        inbag[sample_indices] = True

        oob_indices = np.flatnonzero(~inbag)
        oob_indices_list.append(oob_indices)

    return oob_indices_list

def z_only_cell_ids_for_oob(tree, X_oob_np, z_feature_idx_set):
    tr = tree.tree_
    feature = tr.feature
    threshold = tr.threshold
    children_left = tr.children_left
    children_right = tr.children_right

    cell_ids = []
    for x in X_oob_np:
        node = 0
        sig = []
        while feature[node] != -2:  
            f = feature[node]
            thr = threshold[node]

            go_left = x[f] <= thr
            if f in z_feature_idx_set:
                sig.append((node, 0 if go_left else 1))  

            node = children_left[node] if go_left else children_right[node]

        cell_ids.append(tuple(sig))
    return cell_ids

def precompute_oob_paths_for_tree(tree, X_oob_np):

    tr = tree.tree_
    feat = tr.feature
    thr = tr.threshold
    left = tr.children_left
    right = tr.children_right

    paths_nodes, paths_dirs, paths_feats = [], [], []
    for x in X_oob_np:
        node = 0
        nodes_i, dirs_i, feats_i = [], [], []

        while feat[node] != -2: 
            f = feat[node]
            go_left = x[f] <= thr[node]
            d = 0 if go_left else 1

            nodes_i.append(node)
            dirs_i.append(d)
            feats_i.append(f)

            node = left[node] if go_left else right[node]

        paths_nodes.append(np.asarray(nodes_i, dtype=np.int32))
        paths_dirs.append(np.asarray(dirs_i, dtype=np.int8))
        paths_feats.append(np.asarray(feats_i, dtype=np.int32))

    return paths_nodes, paths_dirs, paths_feats

def z_only_cell_ids_from_cached_paths(paths_nodes, paths_dirs, paths_feats, z_set):
    z_list = list(z_set)
    cell_ids = []
    for nodes_i, dirs_i, feats_i in zip(paths_nodes, paths_dirs, paths_feats):
        if feats_i.size == 0:
            cell_ids.append(tuple())
            continue
        mask = np.isin(feats_i, z_list, assume_unique=False)
        sig = tuple(zip(nodes_i[mask].tolist(), dirs_i[mask].tolist()))
        cell_ids.append(sig)
    return cell_ids

def conditional_shuffle_one_column_in_cells(X_oob, col_idx, cell_ids, random_state=None):
    rs = check_random_state(random_state)
    X_perm = X_oob.copy()
    groups = {}

    for i, cid in enumerate(cell_ids):
        groups.setdefault(cid, []).append(i)

    for idxs in groups.values():
        if len(idxs) <= 1:
            continue
        idxs = np.asarray(idxs)
        perm = idxs.copy()
        rs.shuffle(perm)
        X_perm[idxs, col_idx] = X_perm[perm, col_idx]

    return X_perm

def make_blocks(n_items, block_size):
    return [list(range(i, min(i + block_size, n_items)))for i in range(0, n_items, block_size)]

def choose_block_size(p, n_jobs, target_tasks_per_worker=6, min_bs=5, max_bs=50):
    n_workers = effective_n_jobs(n_jobs)
    if n_workers <= 1:
        return max_bs
    bs = math.ceil(p / (n_workers * target_tasks_per_worker))
    return max(min_bs, min(max_bs, bs))

def conditional_permutation_importance_rf_feature(rf, X, y, Z_idx_map, scoring="roc_auc", random_seed = None, n_jobs = -1, feature_block_size=None):
    
    X_all = X.to_numpy()
    y_all = np.asarray(y)
    scorer = check_scoring(rf.estimators_[0], scoring=scoring)

    oob_indices_list = get_per_tree_oob_indices(
        rf,
        n_samples=X_all.shape[0]
    )

      
    n_features = X.shape[1]
    n_trees = len(rf.estimators_)

    base_scores = np.full(n_trees, np.nan)

    for t, tree in enumerate(rf.estimators_):
        oob_idx = oob_indices_list[t]
        if len(oob_idx) <= 1:
            continue
        base_scores[t] = scorer(tree, X_all[oob_idx], y_all[oob_idx])

    drop_matrix = np.full((n_features, n_trees), np.nan)

    if feature_block_size is None:
        feature_block_size = choose_block_size(n_features, n_jobs)
    blocks = make_blocks(n_features, feature_block_size)
    def _compute_feature_block(js):
        out = []
        for j in js:
            row = np.full(n_trees, np.nan)
            z_set = set(Z_idx_map[j])

            for t, tree in enumerate(rf.estimators_):
                oob_idx = oob_indices_list[t]
                if len(oob_idx) <= 1 or np.isnan(base_scores[t]):
                    continue

                X_oob = X_all[oob_idx]
                y_oob = y_all[oob_idx]

                cell_ids = z_only_cell_ids_for_oob(tree, X_oob, z_set)

                if random_seed is None:
                    seed = None
                else:
                    seed = random_seed + j * n_trees + t
                X_perm = conditional_shuffle_one_column_in_cells(X_oob, j, cell_ids, random_state = seed)
                perm = scorer(tree, X_perm, y_oob)

                row[t] = base_scores[t] - perm
            out.append((j, row))
        return out
    
    if effective_n_jobs(n_jobs) == 1:
        for js in blocks:
            for j, row in _compute_feature_block(js):
                drop_matrix[j, :] = row
    else:
        results = Parallel(n_jobs=n_jobs, backend='loky', max_nbytes="50M", mmap_mode="r")(
            delayed(_compute_feature_block)(js) for js in blocks
        )
        for blk in results:
            for j, row in blk:
                drop_matrix[j, :] = row

    result = Bunch(
        CPI = drop_matrix,
        CPI_mean = np.nanmean(drop_matrix, axis = 1),
        CPI_std = np.nanstd(drop_matrix, axis = 1)
    )

    return result

def conditional_permutation_importance_rf_tree(rf, X, y, Z_idx_map, scoring="roc_auc", random_seed = None, n_jobs = -1, tree_block_size=None):
    
    X_all = X.to_numpy()
    y_all = np.asarray(y)
    scorer = check_scoring(rf.estimators_[0], scoring=scoring)

    oob_indices_list = get_per_tree_oob_indices(
        rf,
        n_samples=X_all.shape[0]
    )

    n_features = X.shape[1]
    n_trees = len(rf.estimators_)

    drop_matrix = np.full((n_features, n_trees), np.nan)

    if tree_block_size is None:
        tree_block_size = choose_block_size(n_trees, n_jobs)
    tree_blocks = make_blocks(n_trees, tree_block_size)
    def _compute_tree_block(ts):
        out = []
        for t in ts:
            tree = rf.estimators_[t]
            oob_idx = oob_indices_list[t]
            if len(oob_idx) <= 1:
                out.append((t, np.full(n_features, np.nan)))
                continue
            X_oob = X_all[oob_idx]
            y_oob = y_all[oob_idx]
            base = scorer(tree, X_oob, y_oob)
            paths_nodes, paths_dirs, paths_feats = precompute_oob_paths_for_tree(tree, X_oob)
            col = np.full(n_features, np.nan)
            for j in range(n_features):
                z_set = set(Z_idx_map[j])
                cell_ids = z_only_cell_ids_from_cached_paths(paths_nodes, paths_dirs, paths_feats, z_set)
                if random_seed is None:
                    seed = None
                else:
                    seed = random_seed + j * n_trees + t
                X_perm = conditional_shuffle_one_column_in_cells(X_oob, j, cell_ids, random_state=seed)
                perm = scorer(tree, X_perm, y_oob)
                col[j] = base - perm
            out.append((t, col))
        return out
    
    if effective_n_jobs(n_jobs) == 1:
        for ts in tree_blocks:
            for t, col in _compute_tree_block(ts):
                drop_matrix[:, t] = col
    else:
        results = Parallel(n_jobs=n_jobs, backend='loky', max_nbytes="50M", mmap_mode="r")(
            delayed(_compute_tree_block)(ts) for ts in tree_blocks
        )
        for blk in results:
            for t, col in blk:
                drop_matrix[:, t] = col

    result = Bunch(
        CPI = drop_matrix,
        CPI_mean = np.nanmean(drop_matrix, axis = 1),
        CPI_std = np.nanstd(drop_matrix, axis = 1)
    )
    
    return result

def cpi_auto(rf, X, y, Z_idx_map, scoring="roc_auc", random_seed=None, n_jobs=-1, cutoff=200, feature_block_size=None, tree_block_size=None):
    p = X.shape[1]
    if p < cutoff:
        return conditional_permutation_importance_rf_feature(
            rf, X, y, Z_idx_map,
            scoring=scoring,
            random_seed=random_seed,
            n_jobs=n_jobs,
            feature_block_size=feature_block_size
        )
    else:
        return conditional_permutation_importance_rf_tree(
            rf, X, y, Z_idx_map,
            scoring=scoring,
            random_seed=random_seed,
            n_jobs=n_jobs,
            tree_block_size=tree_block_size
        )
    
def create_corr_var_idx_from_abs_corr(abs_corr_df, threshold = 0.7):

    thr = threshold
    cols = abs_corr_df.columns.tolist()
    name_to_idx = {f: i for i, f in enumerate(cols)}

    Z_idx_map = {}
    for f in cols:
        Z_names = abs_corr_df.index[(abs_corr_df[f] >= thr) & (abs_corr_df.index != f)].tolist()
        j = name_to_idx[f]
        Z_idx_map[j] = {name_to_idx[z] for z in Z_names if z in name_to_idx}

    return Z_idx_map

def select_features_cpi_iterative_threshold(X_df, y, feature, threshold = 0.7, *, rf_params = None, scoring="roc_auc", cpi_threshold=0.0,  min_features=10, max_iter=20, min_drop_abs=2, 
                                            min_drop_ratio=0.01, n_jobs = -1, random_state =  None, cpi_kwargs=None):
    if cpi_kwargs is None:
        cpi_kwargs = {}
    if n_jobs is None:
        n_jobs = 1
    abs_corr_full = X_df.corr(method="pearson").abs()

    features = feature.copy()
    if random_state is None:
        base_seed = 42
    else:
        base_seed = random_state
    if rf_params is None:
        rf_params = dict(
                n_estimators=1000,
                max_features="sqrt",
                max_depth=10,
                n_jobs=n_jobs,
                class_weight="balanced",
                random_state = base_seed
            )

    for it in range(max_iter):
        X_sub = X_df[features]

        abs_corr_sub = abs_corr_full.loc[features, features]
        corr_idx_map = create_corr_var_idx_from_abs_corr(abs_corr_sub, threshold=threshold)
        
        rf = RandomForestClassifier(**rf_params)
        with threadpool_limits(limits=1):
            rf.fit(X_sub, y)
        gc.collect()
        with threadpool_limits(limits=1):
            cpi_res = cpi_auto(rf, X_sub, y, scoring=scoring, random_seed = random_state, Z_idx_map = corr_idx_map, n_jobs=n_jobs, **cpi_kwargs)
        cpi_mean = np.asarray(cpi_res.CPI_mean)

        drop_mask = cpi_mean <= cpi_threshold
        drop_list = [f for f, m in zip(features, drop_mask) if m]

        if len(drop_list) == 0: 
            break

        if len(features) - len(drop_list) < min_features:
            break

        drop_ratio = len(drop_list) / len(features)
        if (len(drop_list) <= min_drop_abs) or (drop_ratio <= min_drop_ratio):
            features = [f for f in features if f not in drop_list]
            break

        new_features = [f for f in features if f not in drop_list]

        features = new_features

        if len(features) <= min_features:
            break

    return features

def _compute_k_drop(n_feats: int, drop_fraction):
    if isinstance(drop_fraction, bool):
        raise ValueError("drop_fraction must be int or float, not bool.")

    if isinstance(drop_fraction, int):
        k = drop_fraction
    elif isinstance(drop_fraction, float):
        if drop_fraction <= 0:
            raise ValueError("drop_fraction must be > 0.")
        if drop_fraction < 1:
            k = max(1, int(n_feats * drop_fraction))
        elif drop_fraction.is_integer():
            k = int(drop_fraction)
        else:
            raise ValueError("If drop_fraction >= 1, it must be an integer value (e.g., 2.0).")
    else:
        raise ValueError("drop_fraction must be int or float.")

    return k

def _compute_k_add(n_feats: int, floating_k):
    if isinstance(floating_k, bool):
        raise ValueError("floating_k must be int or float, not bool.")

    if isinstance(floating_k, int):
        k = floating_k
    elif isinstance(floating_k, float):
        if floating_k <= 0:
            raise ValueError("floating_k must be > 0.")
        if floating_k < 1:
            k = max(1, int(n_feats * floating_k))
        elif floating_k.is_integer():
            k = int(floating_k)
        else:
            raise ValueError("If floating_k >= 1, it must be an integer value (e.g., 2.0).")
    else:
        raise ValueError("floating_k must be int or float.")

    return k

def select_features_cpi_rfe_oob(
    X_df, y, feature, threshold = 0.7, *, rank_df=None, rf_params = None, scoring="roc_auc", drop_fraction=0.10, switch_ratio_samples=0.20, min_switch_abs=20, min_features=10, max_iter=50, oob_delta=0.003, patience=1, 
    floating_every=0, floating_k=0, decay_factor=0.75, protect_rounds=0, n_jobs = -1, random_state=None, cpi_kwargs=None):
    if cpi_kwargs is None:
        cpi_kwargs = {}
    abs_corr_full = X_df.corr(method="pearson").abs()
    features = feature.copy()
    switch_at = max(min_switch_abs, int(X_df.shape[0] * switch_ratio_samples))
    if rank_df is not None:
        base_score = 1.0 / (rank_df["rank_sum"] + 1)
        base_prob = base_score / base_score.sum()

        base_prob = base_prob.to_dict()

        floating_weight = {f: 1.0 for f in rank_df.index}
    else:
        base_prob = None
        floating_weight = None

    protected = {}
    best_oob = -np.inf
    bad_count = 0
    mode = "coarse" 
    if random_state is None:
        base_seed = 42
    else:
        base_seed = random_state
    
    if rf_params is None:
        rf_params = dict(
                n_estimators=1000,
                max_features="sqrt",
                max_depth=10,
                oob_score = True,
                n_jobs=n_jobs,
                class_weight="balanced",
                random_state=base_seed
            )
    prev_features = features.copy()
    floating_added = set()
    rng = np.random.default_rng(random_state)
    for it in range(max_iter):
        add_list = []
        prev_features = features.copy()

        X_sub = X_df[features]
        abs_corr_sub = abs_corr_full.loc[features, features]
        corr_idx_map = create_corr_var_idx_from_abs_corr(abs_corr_sub, threshold=threshold)
        rf = RandomForestClassifier(**rf_params)
        with threadpool_limits(limits=1):
            rf.fit(X_sub, y)
        gc.collect()
        oob = getattr(rf, "oob_score_", np.nan)
        with threadpool_limits(limits=1):
            cpi_res = cpi_auto(rf, X_sub, y, scoring=scoring, random_seed = random_state, Z_idx_map = corr_idx_map, n_jobs=n_jobs, **cpi_kwargs)
        cpi_mean = np.asarray(cpi_res.CPI_mean)

        if oob > best_oob:
            best_oob = oob
            bad_count = 0
        elif oob < best_oob - oob_delta:
            bad_count += 1

        if bad_count >= patience:
            if mode == "fine":
                
                break
            else:
                features = prev_features
                mode = "fine"
                bad_count = 0 
                
                continue

        if mode == "coarse" and len(features) <= switch_at:
            mode = "fine"

        if len(features) <= min_features:
            break

        candidate_idx = [
            i for i, f in enumerate(features) if protected.get(f, 0) <= 0
        ]

        if len(candidate_idx) == 0:
            break

        if mode == "coarse":
            k_drop = _compute_k_drop(len(candidate_idx), drop_fraction)
            if k_drop <= 1:
                k_drop = 1
                mode = "fine"
        else:
            k_drop = 1

        order = np.argsort(cpi_mean[candidate_idx]) 
        drop_pos = order[:k_drop]
        drop_idx = [candidate_idx[i] for i in drop_pos]
        drop_list = [features[i] for i in drop_idx]
        if len(features) - len(drop_list) < min_features:
            drop_list = drop_list[:max(0, len(features) - min_features)]
            if len(drop_list) == 0:
                break

        if floating_weight is not None:
            for f in drop_list:
                if f in floating_added:
                    floating_weight[f] *= decay_factor
                    
        new_features = [f for f in features if f not in drop_list]

        features = new_features

        if (
            rank_df is not None
            and floating_every > 0
            and mode == "coarse"
            and it > 0 and it % floating_every == 0
        ):
            pool = [f for f in rank_df.index if f not in features]

            if pool:
                k_add = _compute_k_add(len(features), floating_k)
                k_add = min(k_add, k_drop, len(pool))

                if k_add > 0:
                    probs = np.array([
                        base_prob.get(f, 0.0) * floating_weight.get(f, 0.0) for f in pool
                    ], dtype = float)
                    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                    s = probs.sum()
                    if s <= 0:
                        probs = None
                    else:
                        probs = probs / s

                    add_list = rng.choice(pool, size=k_add, replace=False, p=probs).tolist()
                    floating_added.update(add_list)

                    features.extend(add_list)

                    for f in add_list:
                        if protect_rounds > 0:
                            protected[f] = protect_rounds


        for f in list(protected.keys()):
            protected[f] -= 1
            if protected[f] <= 0:
                del protected[f]


    return features

def select_feature_through_wrapper(X, y, initial_K, filter = ("mRMR", "FCBF", "CMIM", "JMI", "reliefF"), threshold = 0.7, scoring = 'roc_auc',
                                    wrapper = 'threshold', n_jobs = -1, random_state = None, rf_params = None, cpi_kwargs = None, filter_kwargs = None, iterative_kwargs = None, rfe_kwargs = None):
    if filter_kwargs is None:
            filter_kwargs = {}
    top_feature, top_rank, _, _ = final_selector(X, y, initial_K, methods = filter, **filter_kwargs)    
    if wrapper == 'threshold':
        if iterative_kwargs is None:
            iterative_kwargs = {}
        features = select_features_cpi_iterative_threshold(X, y, top_feature, threshold = threshold, rf_params = rf_params, scoring = scoring,
                                                            n_jobs = n_jobs, random_state = random_state, cpi_kwargs = cpi_kwargs, **iterative_kwargs)
    elif wrapper == 'rfe':
        if rfe_kwargs is None:
            rfe_kwargs = {}
        features = select_features_cpi_rfe_oob(X, y, top_feature, threshold = threshold, rank_df = top_rank, rf_params = rf_params, scoring = scoring, 
                                               random_state = random_state, n_jobs = n_jobs, cpi_kwargs = cpi_kwargs, **rfe_kwargs)
    elif wrapper is None:
        features = top_feature
    return features

def _one_fold(X, y_series, rf_params, fold_id, tr_idx, va_idx):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr = y_series.iloc[tr_idx]
    y_va = y_series.iloc[va_idx].values

    model = RandomForestClassifier(**rf_params)
    model.fit(X_tr, y_tr)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_va)

    y_va = y_series.iloc[va_idx].values
    class_labels = model.classes_
    class_to_idx = {c: i for i, c in enumerate(class_labels)}

    if isinstance(shap_values, list):
        S_va = np.zeros_like(shap_values[0])
        for i, y_i in enumerate(y_va):
            k = class_to_idx[y_i]
            S_va[i, :] = shap_values[k][i, :]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            idx = np.array([class_to_idx[y_i] for y_i in y_va])
            S_va = arr[np.arange(len(y_va)), :, idx]
        else:
            S_va = arr

    return va_idx, S_va

def oof_shap_matrix_rf(
    X: pd.DataFrame, y, *, rf_params= None, n_splits= 5, random_state = None, n_jobs = -1
):
    if rf_params is None:
        rf_params = dict(
            n_estimators=1000,
            max_features="sqrt",
            max_depth=10,
            n_jobs = 1,
            class_weight="balanced",
            random_state=random_state,
        )

    y_series = y if hasattr(y, "iloc") else pd.Series(y, index=X.index)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    n, p = X.shape
    shap_oof = np.full((n, p), np.nan, dtype=float)
    if n_jobs is None:
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_series), start=1):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr = y_series.iloc[tr_idx]

            model = RandomForestClassifier(**rf_params)
            model.fit(X_tr, y_tr)

            explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(X_va)

            y_va = y_series.iloc[va_idx].values
            class_labels = model.classes_
            class_to_idx = {c: i for i, c in enumerate(class_labels)}

            if isinstance(shap_values, list):
                S_va = np.zeros_like(shap_values[0])
                for i, y_i in enumerate(y_va):
                    k = class_to_idx[y_i]
                    S_va[i, :] = shap_values[k][i, :]
            else:
                arr = np.asarray(shap_values)
                if arr.ndim == 3:
                    idx = np.array([class_to_idx[y_i] for y_i in y_va])
                    S_va = arr[np.arange(len(y_va)), :, idx]
                else:
                    S_va = arr

            shap_oof[va_idx, :] = S_va
    else:
        if n_jobs == -1:
            workers = os.cpu_count()
        else:
            workers = n_jobs
        folds = list(skf.split(X, y_series))
        results = Parallel(n_jobs=min(n_splits, workers), backend="loky")(
            delayed(_one_fold)(X, y_series, rf_params, fold_id, tr_idx, va_idx)
            for fold_id, (tr_idx, va_idx) in enumerate(folds)
        )
        for va_idx, S_va in results:
            shap_oof[va_idx, :] = S_va
    shap_oof_df = pd.DataFrame(shap_oof, index=X.index, columns=X.columns)

    return shap_oof_df

def shap_summary(shap_oof_df,  rho0 = 0.65, cluster = True, *, method_corr = "pearson", linkage_method = "average", eps = 1e-12 ):

    if not isinstance(shap_oof_df, pd.DataFrame):
        raise TypeError("shap_oof_df must be a pandas DataFrame with columns as feature names.")

    A = shap_oof_df.abs()
    mean_abs = A.mean(axis=0)
    std_abs = A.std(axis=0, ddof=0)
    cv_abs = std_abs / (mean_abs + eps)

    summary_df = pd.DataFrame({
        "shap_abs_mean": mean_abs,
        "shap_abs_std": std_abs,
        "shap_abs_cv": cv_abs,
    })

    if cluster:
        R = A.corr(method=method_corr)
        D = 1.0 - R.values
        np.fill_diagonal(D, 0.0)
        D_cond = squareform(D, checks=False)

        Z = linkage(D_cond, method=linkage_method)

        t = 1.0 - rho0
        labels = fcluster(Z, t=t, criterion="distance")

        feat_names = R.index.tolist()
        cluster_id = pd.Series(labels, index=feat_names, name="cluster")
        summary_df["cluster"] = cluster_id

    return summary_df

def _compute_score(model, X_va, y_va, scoring):
    if scoring == "roc_auc":
        proba = model.predict_proba(X_va)[:, 1]
        return roc_auc_score(y_va, proba)

    elif scoring == "average_precision":
        proba = model.predict_proba(X_va)[:, 1]
        return average_precision_score(y_va, proba)

    elif scoring == "accuracy":
        pred = model.predict(X_va)
        return accuracy_score(y_va, pred)

    elif scoring == "balanced_accuracy":
        pred = model.predict(X_va)
        return balanced_accuracy_score(y_va, pred)

    elif scoring == "f1":
        pred = model.predict(X_va)
        return f1_score(y_va, pred)

    elif scoring == "log_loss":
        proba = model.predict_proba(X_va)
        return -log_loss(y_va, proba)

    elif scoring == "recall":
        pred = model.predict(X_va)
        return recall_score(y_va, pred)

    else:
        raise ValueError(f"Unsupported scoring: {scoring}")
    
def _evaluate_fixed_cv(X, y, features, estimator, splits, scoring = "roc_auc", base_seed = None):

    y_series = y if hasattr(y, "iloc") else pd.Series(y, index=X.index)

    scores = []
    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        X_tr = X.iloc[tr_idx][features]
        X_va = X.iloc[va_idx][features]
        y_tr = y_series.iloc[tr_idx]
        y_va = y_series.iloc[va_idx]
        model = clone(estimator)
        if base_seed is not None:
            if hasattr(model, "random_state"):
                model.set_params(random_state=base_seed + fold_id)
        with threadpool_limits(limits=1):
            model.fit(X_tr, y_tr)

        score = _compute_score(model, X_va, y_va, scoring)
        scores.append(score)

    return float(np.mean(scores))

def fixed_splits(X, y, n_splits, random_state = None):
    if hasattr(y, "index"):
        assert X.index.equals(y.index)
    y_series = y if hasattr(y, "iloc") else pd.Series(y, index=X.index)
    skf = StratifiedKFold(
        n_splits = n_splits,
        shuffle = True,
        random_state = random_state,
    )
    splits = list(skf.split(X, y_series))
    return splits

def _loo_trial_one_feature(f_drop, X, y, current_features, estimator, splits, scoring, current_score, summary_df=None, extra_cols=(), base_seed = None):

    trial_features = [f for f in current_features if f != f_drop]

    with threadpool_limits(limits=1):
        trial_score = _evaluate_fixed_cv(X, y, trial_features, estimator, splits, scoring, base_seed)

    delta = trial_score - current_score

    row = {
        "feature": f_drop,
        "trial_score": float(trial_score),
        "delta": float(delta),
    }

    if summary_df is not None:
        for c in extra_cols:
            row[c] = float(summary_df.loc[f_drop, c])

    return row


def loo_sweep_parallel(candidates, X, y, current_features, estimator, splits, scoring, current_score, summary_df=None, extra_cols=(), n_jobs=2, base_seed = None, backend="loky", batch_size="auto"):

    rows = Parallel(n_jobs=n_jobs, backend=backend, batch_size=batch_size)(
        delayed(_loo_trial_one_feature)(
            f_drop,
            X, y, current_features, estimator, splits, scoring,
            current_score,
            summary_df, extra_cols, base_seed = base_seed
        )
        for f_drop in candidates
    )

    return pd.DataFrame(rows)

def phase1_cluster_pruning(X, y, summary_df, splits, rf_params = None, random_state = None, scoring = "roc_auc", tolerance = 0.001, use_anchor = True, verbose = False, n_jobs = -1):
    if "cluster" not in summary_df.columns:
        raise ValueError("summary_df must contain column 'cluster'")
    if n_jobs is not None:
        if n_jobs == -1:
            max_workers = os.cpu_count()
        else:
            max_workers = max(n_jobs, os.cpu_count())
    if use_anchor and "shap_abs_mean" not in summary_df.columns:
        raise ValueError("use_anchor=True requires 'shap_abs_mean'")
    if random_state is None:
        base_seed = 42
    else:
        base_seed = random_state
    if rf_params is None:
        rf_params = dict(
            n_estimators=1000,
            max_features="sqrt",
            max_depth=10,
            n_jobs = 1,
            class_weight="balanced",
            random_state=base_seed,
        )
    estimator = RandomForestClassifier(**rf_params)
    current_features = [
        f for f in summary_df.index.tolist() if f in X.columns
    ]

    current_score = _evaluate_fixed_cv(
        X, y, current_features, estimator, splits, scoring, base_seed = base_seed
    )

    if verbose:
        print(
            f"[Phase1] Baseline score = {current_score:.6f} "
            f"with {len(current_features)} features"
        )
    history = []
    for cid in sorted(summary_df["cluster"].unique()):
        cluster_feats = [
            f for f in summary_df.index[summary_df["cluster"] == cid]
            if f in current_features
        ]
        if len(cluster_feats) <= 1:
            continue
        if use_anchor:
            anchor = (
                summary_df.loc[cluster_feats, "shap_abs_mean"]
                .idxmax()
            )
        else:
            anchor = None

        if verbose:
            msg = f"\n[Cluster {cid}] start with {len(cluster_feats)} features"
            if anchor:
                msg += f", anchor = {anchor}"
            print(msg)
        while True:
            cluster_now = [
                f for f in cluster_feats if f in current_features
            ]

            if len(cluster_now) <= 1:
                if verbose:
                    print(f"[Cluster {cid}] stop (only 1 feature left)")
                break

            candidates = [
                f for f in cluster_now if f != anchor
            ]

            if not candidates:
                if verbose:
                    print(f"[Cluster {cid}] stop (only anchor remains)")
                break
            if n_jobs is None:
                trial_results = []
                for f_drop in candidates: 
                    trial_features = [ f for f in current_features if f != f_drop ] 
                    trial_score = _evaluate_fixed_cv(X, y, trial_features, estimator, splits, scoring, base_seed = base_seed)
                    delta = trial_score - current_score
                    trial_results.append({ "feature": f_drop, 
                                          "trial_score": trial_score, 
                                          "delta": delta, 
                                          "shap_abs_mean": float(summary_df.loc[f_drop, "shap_abs_mean"])})
                    if verbose: 
                        print( f" try drop {f_drop}: " f"score {trial_score:.6f} (Δ {delta:+.6f})" )
                trial_df = pd.DataFrame(trial_results)
            else:
                trial_df = loo_sweep_parallel(candidates = candidates, X = X, y = y, estimator = estimator, 
                                            splits = splits, scoring = scoring, current_score = current_score,
                                            current_features = current_features, summary_df = summary_df, extra_cols = ('shap_abs_mean',), 
                                            n_jobs = min(len(candidates), max_workers), base_seed = base_seed)
            ok = trial_df[trial_df["delta"] >= -tolerance].copy()
            if ok.empty:
                if verbose:
                    print(f"[Cluster {cid}] no removable feature, stop")
                break
            ok = ok.sort_values(
                by=["delta", "shap_abs_mean"],
                ascending=[False, True]
            )
            chosen = ok.iloc[0]["feature"]
            current_features.remove(chosen)
            current_score = ok.iloc[0]["trial_score"]

            history.append({
                "phase": 1,
                "cluster": int(cid),
                "feature_dropped": chosen,
                "new_score": current_score,
                "delta": ok.iloc[0]["delta"],
                "n_features": len(current_features),
            })

            if verbose:
                print(
                    f"  -> drop {chosen}, "
                    f"new score {current_score:.6f}"
                )

    history_df = pd.DataFrame(history)
    return current_features, current_score, history_df

def phase2_selected_feature(X, y, shap_summary_df, splits, rf_params = None, tolerance = 0.001, cv_quantile = 0.80, scoring = "roc_auc", verbose = False, n_jobs = -1, random_state = None):
    if n_jobs is not None:
        if n_jobs == -1:
            max_workers = os.cpu_count()
        else:
            max_workers = max(n_jobs, os.cpu_count())
    if random_state is None:
        base_seed = 42
    else:
        base_seed = random_state
    required = {"shap_abs_cv", "shap_abs_mean"}
    missing = required - set(shap_summary_df.columns)
    if missing:
        raise ValueError(f"shap_summary_df missing columns: {missing}")
    
    current_features = [
            f for f in shap_summary_df.index.tolist() if f in X.columns
        ]
    summary = shap_summary_df.copy()

    cv_threshold = float(summary["shap_abs_cv"].quantile(cv_quantile))
    candidates = summary.index[summary["shap_abs_cv"] >= cv_threshold].tolist()

    if verbose:
        print(f"[Phase2] Candidates: {len(candidates)} / {len(current_features)} "
              f"(cv >= {cv_threshold:.6f})")
    if rf_params is None:
        rf_params = dict(
            n_estimators=1000,
            max_features="sqrt",
            max_depth=10,
            n_jobs = 1,
            class_weight="balanced",
            random_state = base_seed,
        )
    estimator = RandomForestClassifier(**rf_params)

    current_score = _evaluate_fixed_cv(X, y, current_features, estimator, splits, scoring, base_seed = base_seed)
    if verbose:
        print(f"[Phase2] Baseline score = {current_score:.6f}")

    history = []
    it = 0

    while True:
        it += 1
        candidates_now = [f for f in candidates if f in current_features]
        if not candidates_now:
            if verbose:
                print("[Phase2] No candidates left, stop.")
            break
        if n_jobs is None:
            trial_rows = []
            for f_drop in candidates_now:
                trial_features = [f for f in current_features if f != f_drop] 
                trial_score = _evaluate_fixed_cv(X, y, trial_features, estimator, splits, scoring, base_seed = base_seed) 
                delta = trial_score - current_score
                trial_rows.append({ "feature": f_drop, 
                                   "trial_score": trial_score, 
                                   "delta": delta, 
                                   "shap_abs_mean": float(summary.loc[f_drop, "shap_abs_mean"]), 
                                   "shap_abs_cv": float(summary.loc[f_drop, "shap_abs_cv"]), })
                if verbose:
                    print(f" [it={it}] try drop {f_drop}: " f"{trial_score:.6f} (Δ {delta:+.6f})")
            trial_df = pd.DataFrame(trial_rows)
        else:
            trial_df = loo_sweep_parallel(
                candidates = candidates_now, X = X, y = y, current_features = current_features, 
                estimator = estimator, splits = splits, scoring = scoring, summary_df = summary, 
                extra_cols = ('shap_abs_mean', 'shap_abs_cv'), n_jobs = min(len(candidates_now), max_workers),
                current_score = current_score, base_seed = base_seed
            )

        ok = trial_df[trial_df["delta"] >= -tolerance].copy()

        if ok.empty:
            if verbose:
                print("[Phase2] No removable feature under tolerance, stop.")
            break

        ok = ok.sort_values(
            by=["delta", "shap_abs_mean", "shap_abs_cv"],
            ascending=[False, True, False]
        )

        chosen = ok.iloc[0]["feature"]
        chosen_score = float(ok.iloc[0]["trial_score"])
        chosen_delta = float(ok.iloc[0]["delta"])

        # 永久刪除 & 更新 baseline
        current_features.remove(chosen)
        current_score = chosen_score

        history.append({
            "phase": 2,
            "iter": it,
            "feature_dropped": chosen,
            "new_score": current_score,
            "delta": chosen_delta,
            "n_features": len(current_features),
            "cv_threshold": cv_threshold,
            "shap_abs_mean": float(ok.iloc[0]["shap_abs_mean"]),
            "shap_abs_cv": float(ok.iloc[0]["shap_abs_cv"]),
        })

        if verbose:
            print(f"  -> drop {chosen}, new score {current_score:.6f}, "
                  f"n_features={len(current_features)}")

    history_df = pd.DataFrame(history)
    return current_features, current_score, history_df

def phase3_selected_feature(X, y, shap_summary_df, splits, rf_params = None, tolerance = 0.001, keep_cumshare = 0.90, scoring = 'roc_auc', verbose = False, n_jobs = -1, random_state = None):
    if n_jobs is not None:
        if n_jobs == -1:
            max_workers = os.cpu_count()
        else:
            max_workers = max(n_jobs, os.cpu_count())
    if random_state is None:
        base_seed = 42
    else:
        base_seed = random_state
    if rf_params is None:
        rf_params = dict(
            n_estimators=1000,
            max_features="sqrt",
            max_depth=10,
            n_jobs = 1,
            class_weight="balanced",
            random_state = base_seed,
        )
    estimator = RandomForestClassifier(**rf_params)

    current_features = [f for f in shap_summary_df.index.tolist() if f in X.columns]

    mean_col = "shap_abs_mean"

    if mean_col not in shap_summary_df.columns:
        raise ValueError(f"shap_summary_df must contain '{mean_col}'")

    df = shap_summary_df.loc[
        [f for f in shap_summary_df.index if f in current_features],
        [mean_col]
    ].copy()

    df = df.sort_values(mean_col, ascending=False)
    total = df[mean_col].sum()

    if total <= 0:
        candidates = []
    else:
        df["share"] = df[mean_col] / total
        df["cum_share"] = df["share"].cumsum()
        head = df.index[df["cum_share"] <= keep_cumshare].tolist()
        candidates = [f for f in df.index.tolist() if f not in head]

    if not candidates:
        estimator = RandomForestClassifier(**rf_params)
        score = _evaluate_fixed_cv(X, y, current_features, estimator, splits, scoring, base_seed = base_seed)
        return current_features, score, pd.DataFrame([])
    
    if verbose:
        print(f"[Phase3] Candidates (tail) = {len(candidates)} / {len(current_features)} "
              f"(keep_cumshare={keep_cumshare:.2f})")
    current_score = _evaluate_fixed_cv(X, y, current_features, estimator, splits, scoring, base_seed = base_seed)
    if verbose:
        print(f"[Phase3] Baseline score = {current_score:.6f}")

    history = []
    it = 0
    while True:
        it += 1

        candidates_now = [f for f in candidates if f in current_features]
        if not candidates_now:
            if verbose:
                print("[Phase3] No candidates left, stop.")
            break
        if n_jobs is None:
            trial_rows = [] 
            for f_drop in candidates_now: 
                trial_features = [f for f in current_features if f != f_drop] 
                trial_score = _evaluate_fixed_cv(X, y, trial_features, estimator, splits, scoring, base_seed = base_seed) 
                delta = trial_score - current_score 
                trial_rows.append({ "feature": f_drop, "trial_score": trial_score, "delta": delta, "shap_abs_mean": float(shap_summary_df.loc[f_drop, "shap_abs_mean"]), }) 
                if verbose: 
                    print(f" [it={it}] try drop {f_drop}: " f"{trial_score:.6f} (Δ {delta:+.6f})") 
            trial_df = pd.DataFrame(trial_rows)
        else:
            trial_df = loo_sweep_parallel(candidates = candidates_now, X = X, y = y,
                                        current_features = current_features, estimator = estimator, 
                                        splits = splits, scoring = scoring, current_score = current_score, 
                                        summary_df = shap_summary_df, extra_cols = ('shap_abs_mean',), 
                                        n_jobs = min(len(candidates_now), max_workers), base_seed = base_seed)

        ok = trial_df[trial_df["delta"] >= -tolerance].copy()

        if ok.empty:
            if verbose:
                print("[Phase3] No removable feature under tolerance, stop.")
            break

        ok = ok.sort_values(
            by=["shap_abs_mean", "delta"],
            ascending=[True, False]
        )

        chosen = ok.iloc[0]["feature"]
        chosen_score = float(ok.iloc[0]["trial_score"])
        chosen_delta = float(ok.iloc[0]["delta"])

        current_features.remove(chosen)
        current_score = chosen_score

        history.append({
            "phase": 3,
            "iter": it,
            "feature_dropped": chosen,
            "new_score": current_score,
            "delta": chosen_delta,
            "n_features": len(current_features),
            "keep_cumshare": keep_cumshare,
            "shap_abs_mean": float(ok.iloc[0]["shap_abs_mean"]),
        })

        if verbose:
            print(f"  -> drop {chosen}, new score {current_score:.6f}, "
                  f"n_features={len(current_features)}")

    history_df = pd.DataFrame(history)
    return current_features, current_score, history_df

def shap_selection(X, y, feature, rf_params = None, n_splits = 5, scoring = 'roc_auc', cluster = True, coef_var = True, mean = True, random_state = None, cluster_kwargs = None,
                phase1_kwargs = None, phase2_kwargs = None, phase3_kwargs = None):
    features = feature.copy()
    origin_shap = oof_shap_matrix_rf(X[features], y, rf_params = rf_params, n_splits = n_splits, random_state = random_state)
    splits = fixed_splits(X, y, n_splits = n_splits, random_state = random_state)
    if cluster_kwargs is None:
        cluster_kwargs = {}
    summary_df = shap_summary(origin_shap, cluster = cluster, **cluster_kwargs)
    if cluster:
        if phase1_kwargs is None:
            phase1_kwargs = {}
        features, _, _ = phase1_cluster_pruning(X[features], y, summary_df, splits = splits, rf_params = rf_params, scoring = scoring, **phase1_kwargs)
        summary_df = shap_summary(oof_shap_matrix_rf(X[features], y, rf_params = rf_params, n_splits = n_splits, random_state = random_state), cluster = False)
    if coef_var:
        if phase2_kwargs is None:
            phase2_kwargs = {}
        features, _, _ = phase2_selected_feature(X[features], y, summary_df, splits = splits, rf_params = rf_params, scoring = scoring, **phase2_kwargs)
        summary_df = shap_summary(oof_shap_matrix_rf(X[features], y, rf_params = rf_params, n_splits = n_splits, random_state = random_state), cluster = False)
    if mean:
        if phase3_kwargs is None:
            phase3_kwargs = {}    
        features, _, _ = phase3_selected_feature(X[features], y, summary_df, splits = splits, rf_params = rf_params, scoring = scoring, **phase3_kwargs)
    return features

def feature_selection(X, y, initial_K, scoring = 'roc_auc', filter = ("mRMR", "FCBF", "CMIM", "JMI", "reliefF"), corr_threshold = 0.7, wrapper = 'threshold', shap_selected = True, n_jobs = -1, 
                      random_state = None, rf_params = None, filter_kwargs = None, wrapper_kwargs = None, shap_kwargs = None):    
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
    features = select_feature_through_wrapper(X, y, initial_K, filter = filter, threshold = corr_threshold, scoring = scoring, wrapper = wrapper, n_jobs = n_jobs, 
                                              random_state = random_state, rf_params = rf_params, filter_kwargs = filter_kwargs, **wrapper_kwargs)
    if shap_selected:
        if shap_kwargs is None:
            shap_kwargs = {}
        features = shap_selection(X, y, features, rf_params = rf_params, scoring = scoring, random_state = random_state, **shap_kwargs)

    return features