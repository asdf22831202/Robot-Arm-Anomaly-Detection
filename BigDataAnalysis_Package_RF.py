import os, json, gc, joblib
import numpy as np
from collections import defaultdict
from scipy import signal
from scipy.ndimage import grey_erosion, grey_dilation, grey_opening, grey_closing
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft
import pywt
from scipy.signal import decimate
import antropy as ant
from PyEMD import EMD
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, recall_score
from Feature_Filter import feature_selection
import re

def read_data(folder_path):
    data_dict = {}
    #subfolders = sorted([f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))])
    subfolders = sorted(os.listdir(folder_path))
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            files = sorted([f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))])
            for i, file_name in enumerate(files):
                file_path = os.path.join(subfolder_path, file_name)
                data = np.loadtxt(file_path, skiprows=1)
                key = f"data_{folder_path.split('/')[2]}_{subfolder}_{i + 1}" 
                data_dict[key] = data
        elif os.path.isfile(subfolder_path):
            data = np.loadtxt(subfolder_path, skiprows=1)
            key = f"data_root_{subfolder}" 
            data_dict[key] = data
    return data_dict

def read_txt(file_path):
    data_dict = {}
    if os.path.isfile(file_path):
        data = np.loadtxt(file_path, skiprows=1)
        key = os.path.splitext(os.path.basename(file_path))[0]  # e.g. "datafile1.txt" -> "datafile1"
        data_dict[key] = data
    else:
        raise ValueError("The provided path is not a valid file.")
    return data_dict

def tune_data(data_dict, min_len=16175, prefix="data_"):
    cleaned_dict = {}

    for key, data in data_dict.items():
        if key.startswith(prefix) and isinstance(data, np.ndarray):
            m = data.shape[0]
            if m < min_len:
                continue  
            cleaned_dict[key] = data  
    return cleaned_dict

def psd_by_label_interpolate(data_dict, axis=0, duration=5.0, nperseg=2048, nfft=2048, fmax=None):

    label_to_interp_psd = defaultdict(list)
    nyquist_list = []

    for key, data in data_dict.items():
        label = key.split('_')[2]

        x = data[:, axis]
        fs_i = len(x) / duration

        freq_i, Pxx_i = signal.welch(x, fs=fs_i, window='hann', nperseg=min(nperseg, len(x)), nfft=nfft, detrend='constant')

        nyquist_list.append(freq_i[-1])
        label_to_interp_psd[label].append((freq_i, Pxx_i))

    if not label_to_interp_psd:
        raise ValueError("No valid data found.")

    f_nyq_common = min(nyquist_list)
    if fmax is not None:
        f_nyq_common = min(f_nyq_common, fmax)

    fs_ref = np.mean([len(v) / duration for v in data_dict.values()])
    df = fs_ref / nfft

    freq_common = np.arange(0, f_nyq_common, df)

    label_to_mean_psd = {}

    for label, psd_list in label_to_interp_psd.items():
        interp_psds = []

        for freq_i, Pxx_i in psd_list:
            interp_Pxx = np.interp(freq_common, freq_i, Pxx_i)
            interp_psds.append(interp_Pxx)

        label_to_mean_psd[label] = np.mean(interp_psds, axis=0)

    label_list = sorted(label_to_mean_psd.keys())

    return freq_common, label_list, label_to_mean_psd

def mean_psd(psd, abn_labels):
    abn_psd = dict(psd) 
    abn_labels = [str(x) for x in abn_labels]

    psd_list = []
    for lab in abn_labels:
        psd_list.append(abn_psd[lab])

    abn_psd = np.mean(np.stack(psd_list, axis=0), axis=0)
    return abn_psd

def psd_to_db(P, eps=1e-12):
    return 10 * np.log10(np.asarray(P, dtype=float) + eps)

def compute_diff_db(freq, psd_a, psd_b, eps=1e-12, use_abs=True, fmin=None, fmax=None):
    freq = np.asarray(freq, dtype=float)
    a_db = psd_to_db(psd_a, eps=eps)
    b_db = psd_to_db(psd_b, eps=eps)
    diff = a_db - b_db
    if use_abs:
        diff = np.abs(diff)

    return freq, diff

def find_mask_over_threshold(diff_db, thresh_db=3.0):
    return np.asarray(diff_db) > float(thresh_db)

def mask_to_bands(freq, mask, min_width_hz=10.0):

    freq = np.asarray(freq, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    idx = np.where(mask)[0]
    if idx.size == 0:
        return []

    bands = []
    start = idx[0]
    prev = idx[0]

    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            f1, f2 = freq[start], freq[prev]
            if (f2 - f1) >= min_width_hz:
                bands.append((float(f1), float(f2)))
            start = prev = i

    f1, f2 = freq[start], freq[prev]
    if (f2 - f1) >= min_width_hz:
        bands.append((float(f1), float(f2)))

    return bands

def bands_and_centers_for_label(freq, psd_a, psd_ref, *, thresh_db=3, min_width_hz=7, fmin=0, fmax=1250):

    f, diff_db = compute_diff_db(freq, psd_a=psd_a, psd_b=psd_ref, use_abs=True, fmin=fmin, fmax=fmax)
    m = find_mask_over_threshold(diff_db, thresh_db=thresh_db)
    bands = mask_to_bands(f, m, min_width_hz=min_width_hz)
    centers = [0.5 * (f1 + f2) for (f1, f2) in bands]
    return bands, centers
    
def collect_axis_centers(freq, psd_dict, ref_label="80", abn_labels=("65", "95", "130"), pooled_label="mean", pooled_psd=None, thresh_db=3, min_width_hz=7, min_width_hz_pooled=None, fmin=0, fmax=1250):
    ref_label = str(ref_label)
    abn_labels = [str(x) for x in abn_labels]
    if min_width_hz_pooled is None:
        min_width_hz_pooled = min_width_hz

    psd_ref = psd_dict[ref_label]

    out_centers = {}
    out_bands = {}

    for lab in abn_labels:
        bands, centers = bands_and_centers_for_label(freq, psd_dict[lab], psd_ref, thresh_db=thresh_db, min_width_hz=min_width_hz, fmin=fmin, fmax=fmax)
        out_bands[lab] = bands
        out_centers[lab] = centers

    if pooled_psd is not None:
        bands, centers = bands_and_centers_for_label( freq, pooled_psd, psd_ref, thresh_db=thresh_db, min_width_hz=min_width_hz_pooled, fmin=fmin, fmax=fmax)
        out_bands[pooled_label] = bands
        out_centers[pooled_label] = centers

    return out_bands, out_centers

def bandpower_ratio(x, fs, f_lo, f_hi):
    X = np.fft.rfft(x)
    P = (np.abs(X) ** 2)
    f = np.fft.rfftfreq(len(x), d=1/fs)

    band = (f >= f_lo) & (f <= f_hi)
    total = P.sum() + 1e-12
    return P[band].sum() / total

def pick_imf_by_bandpower(imfs, fs, f_lo, f_hi, topk=1):
    ratios = []
    for k in range(imfs.shape[0]):
        r = bandpower_ratio(imfs[k], fs, f_lo, f_hi)
        ratios.append(r)
    ratios = np.array(ratios)
    idx_sorted = np.argsort(-ratios)
    return idx_sorted[:topk], ratios

def select_imfs_for_label(imfs, fs, bands, topk=3, ratio_abs_thr=0.05):

    selected_set = set()
    detail = []

    for (f_lo, f_hi) in bands:
        idxs, ratios = pick_imf_by_bandpower(imfs, fs, f_lo=f_lo, f_hi=f_hi, topk=topk)

        kept = []
        for i in idxs:
            r = float(ratios[int(i)])
            if (r >= ratio_abs_thr):
                selected_set.add(int(i))
                kept.append((int(i), r, "thr"))
        if (len(kept) == 0) and (len(idxs) > 0):
            i0 = int(idxs[0])
            r0 = float(ratios[i0])
            selected_set.add(i0)
            kept.append((i0, r0, "fallback_top1"))

        detail.append({
            "band": (f_lo, f_hi),
            "top": [(int(i), float(ratios[int(i)])) for i in idxs],
            "kept": kept
        })

    selected = sorted(selected_set)
    return selected, detail

def rms(x):
    return float(np.sqrt(np.mean(x**2)))

def mm_operator_rms_1d(x: np.ndarray, size: int):

    x_ero = grey_erosion(x, size=size)
    x_dil = grey_dilation(x, size=size)
    x_opn = grey_opening(x, size=size)
    x_cls = grey_closing(x, size=size)

    return rms(x_ero),rms(x_dil),rms(x_opn), rms(x_cls)

def hjorth_params(x: np.ndarray):
    x = np.asarray(x)
    dx = np.diff(x)
    ddx = np.diff(dx)

    var_x = np.var(x)
    var_dx = np.var(dx) if dx.size else 0.0
    var_ddx = np.var(ddx) if ddx.size else 0.0

    activity = var_x
    mobility = np.sqrt(var_dx / (var_x + 1e-12))
    complexity = np.sqrt(var_ddx / (var_dx + 1e-12)) / (mobility + 1e-12)
    return activity, mobility, complexity

def histogram_upper_lower(x):
    
    n = x.size

    x_max = np.max(x)
    x_min = np.min(x)
    delta = (x_max - x_min) / (n - 1)

    HU = x_max + delta / 2.0
    HL = x_min - delta / 2.0
    return HU, HL

def extract_features(signal):
    fs = signal.shape[0]/5
    features = {}
    direction = ['x', 'y', 'z']
    for idx, dir in enumerate(direction):
        x = signal[:, idx]        # 原始振動
        dx = np.diff(x)*fs
        mean_x = np.mean(x)
        rms_x = np.sqrt(np.mean(x**2))
        std_x = np.std(x)
        var_x = np.var(x)
        abs_mean_x = np.mean(np.abs(x))
        peak_x = np.max(np.abs(x))
        # 這裡計算時域的統計量
        ## 基礎統計量
        features[f'Mean_{dir}'] = mean_x
        features[f'RMS_{dir}'] = rms_x
        features[f'Std_{dir}'] = std_x
        features[f'Var_{dir}'] = var_x
        ## 分布特徵
        features[f'Skewness_{dir}'] = skew(x)
        features[f'Kurtosis_{dir}'] = kurtosis(x)
        ## 隨機性量測
        pdf, _ = np.histogram(x, bins=100, density=True)
        features[f'Entropy_{dir}'] = entropy(pdf + 1e-12)
        ## 無因次指標
        features[f"Shape_Factor_{dir}"] = rms_x / (abs_mean_x + 1e-12)
        features[f'CrestFactor_{dir}'] = peak_x / (rms_x + 1e-12)
        features[f'Impulse_Factor_{dir}'] = peak_x / (abs_mean_x + 1e-12)
        features[f'Margin_Factor_{dir}'] = peak_x / ((np.mean(np.sqrt(np.abs(x)))**2) + 1e-12)
        ## 直方圖指標
        Hu, Hl = histogram_upper_lower(x)
        Hr = Hu - Hl
        features[f'HU_{dir}'] = Hu
        features[f'HL_{dir}'] = Hl
        features[f'HR_{dir}'] = Hr
        ## Hjorth 係數
        act, mob, comp = hjorth_params(x)
        features[f"Hjorth_Activity_{dir}"] = float(act)
        features[f"Hjorth_Mobility_{dir}"] = float(mob)
        features[f"Hjorth_Complexity_{dir}"] = float(comp)


        # 這裡是頻域特徵提取
        ## 頻譜統計量
        fc = np.sum(dx * x[1:]) / (2 * np.pi * np.sum(x[1:] ** 2) + 1e-12)
        msf = np.sum(dx ** 2) / (4 * (np.pi ** 2) * np.sum(x[1:] ** 2) + 1e-12)
        rmsf = np.sqrt(msf)
        rvf = np.sqrt(max(msf - fc ** 2, 0))
        fft_vals = np.abs(np.fft.rfft(x))
        fft_val_mu = fft_vals.mean()
        fft_val_std = fft_vals.std(ddof = 0)
        fft_prob = (fft_vals ** 2) / (np.sum((fft_vals ** 2)) + 1e-12)
        spectral_entropy = -np.sum(fft_prob * np.log(fft_prob + 1e-12))
        hist, _ = np.histogram(fft_vals, bins = 100, density=True)
        shannon_entropy = entropy(hist + 1e-12)
        SS = np.mean((fft_vals - fft_val_mu) ** 3) / (fft_val_std ** 3 + 1e-12)
        SK = np.mean((fft_vals - fft_val_mu) ** 4) / (fft_val_std ** 4 + 1e-12) - 3
        features[f'Freq_Center_{dir}'] = fc
        features[f'RMSF_{dir}'] = rmsf
        features[f'RVF_{dir}'] = rvf
        ## 高級頻譜指標   
        features[f"Spectral_Skewness_{dir}"] = SS
        features[f'Spectral_Kurtosis_{dir}'] = SK
        features[f'Spectral_Entropy_{dir}'] = spectral_entropy
        features[f'Shannon_Entropy_{dir}'] = shannon_entropy

        # 時頻域表示法
        ## 小波分解 D3 層詳細係數
        coeffs = pywt.wavedec(x, 'db4', level=3)
        d3 = coeffs[1]
        features[f'Wavelet_D3_Mean_{dir}'] = np.mean(d3)
        features[f'Wavelet_D3_Var_{dir}'] = np.var(d3)
        features[f'Wavelet_D3_Skewness_{dir}'] = skew(d3) if len(d3) > 3 else 0
        features[f'Wavelet_D3_Kurtosis_{dir}'] = kurtosis(d3) if len(d3) > 3 else 0

        # 相位空間不相似度測量
        ## 近似 Entropy
        # 將 data 降採樣，每5個點只取一個，因ApEn的計算量太大，是20000*20000，降採樣後變成4000*4000，理論上來說會大幅縮減計算時間
        x_ds = decimate(x, q=5, zero_phase=True)

        features[f'ApEn_{dir}'] = ant.app_entropy(x_ds, order = 2)
        ## 因為碎形維度 (Fractal Dimension)、相關維度 (Correlation dimension)和最大李雅普諾夫指數裡面有個參數 m 需要多次嘗試才能找到最好的 m，
        ## 那這兩個維度在原文中的表現並不好，所以我不打算花時間去找最適合的 m，而且計算這兩個太花時間了，我單獨計算過了，
        ## 每計算一個要跑30秒，如果到時候兩個都用上，那特徵提取就要花一分鐘

        features[f'Spectral_Energy_{dir}'] = np.sum(fft_vals**2)  # Total spectral energy

    return features

def extract_abnormal_feature(signal, label, centers_by_axis, bands_by_axis):
    fs = signal.shape[0]/5
    features = {}
    direction = ['x', 'y', 'z']
    for l in label:
        for idx, dir in enumerate(direction):
            x = signal[:, idx] 
            centers = centers_by_axis.get(dir, {}).get(str(l), [])
            if centers:
                for j, c in enumerate(centers, start = 1):
                    if c is None or c <= 0:
                        continue
                    size = int(round(fs / c)*0.6)
                    features[f'MM_erosion_{dir}_s{j}_{l}'], features[f'MM_dilation_{dir}_s{j}_{l}'], features[f'MM_opening_{dir}_s{j}_{l}'], features[f'MM_closing_{dir}_s{j}_{l}'] = mm_operator_rms_1d(x, size)
                    
            emd = EMD()
            imfs = emd.emd(x)
            bands = bands_by_axis.get(dir, {}).get(str(l), [])
            selected_imfs, _ = select_imfs_for_label(imfs, fs, bands, topk=3)
            if selected_imfs:
                x_recon = imfs[selected_imfs].sum(axis=0)
                features[f'IMF_Mean_{dir}_{l}'] = np.mean(x_recon)
                features[f'IMF_Var_{dir}_{l}'] = np.var(x_recon)
                features[f'IMF_Skewness_{dir}_{l}'] = skew(x_recon)
                features[f'IMF_Kurtosis_{dir}_{l}'] = kurtosis(x_recon)
            else:
                continue

    return features

def parse_key(key):
    parts = key.split('_') 
    group = parts[1]       
    load = int(parts[2])
    index = int(parts[3])
    return group, load, index

def build_centers_bands_for_group(data_dict, group, fmax = 1250, min_width_hz = 7):
    if group.startswith('X'):
        ref_label  = '80'
        abn_labels = ('65', '95', '130')
    elif group.startswith('Y'):
        ref_label = '260'
        abn_labels = ('220', '300', '380')

    sub = {k: v for k, v in data_dict.items() if parse_key(k)[0] == group}

    x_freq, _, x_psd = psd_by_label_interpolate(sub, axis=0, fmax=fmax)
    y_freq, _, y_psd = psd_by_label_interpolate(sub, axis=1, fmax=fmax)
    z_freq, _, z_psd = psd_by_label_interpolate(sub, axis=2, fmax=fmax)

    x_mean_psd = mean_psd(x_psd, abn_labels)
    y_mean_psd = mean_psd(y_psd, abn_labels)
    z_mean_psd = mean_psd(z_psd, abn_labels)

    axis_pack = {
        "x": (x_freq, x_psd, x_mean_psd),
        "y": (y_freq, y_psd, y_mean_psd),
        "z": (z_freq, z_psd, z_mean_psd),
    }

    bands_by_axis = {}
    centers_by_axis = {}
    for axis_name, (freq, psd_dict, pooled_psd) in axis_pack.items():
        bands_dict, centers_dict = collect_axis_centers(
            freq, psd_dict,
            ref_label=ref_label,
            abn_labels=abn_labels,
            pooled_psd=pooled_psd,
            min_width_hz=min_width_hz
        )
        bands_by_axis[axis_name] = bands_dict
        centers_by_axis[axis_name] = centers_dict

    return centers_by_axis, bands_by_axis

def chunk_list(lst, chunk_size: int):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def _base_chunk(chunk_items):
    out = []
    for key, arr in chunk_items:
        direction, load, index = parse_key(key)
        feats = extract_features(arr)
        feats["Direction"] = direction
        feats["Load"] = load
        feats["Index"] = index
        out.append(feats)
    return out

def _abn_chunk(chunk_items, centers_bands_local):
    out = []
    for key, arr in chunk_items:
        direction, load, index = parse_key(key)

        centers_by_axis, bands_by_axis = centers_bands_local[direction]
        labels = labels_for_sample(load, direction)

        abn = extract_abnormal_feature(arr, labels, centers_by_axis, bands_by_axis)

        abn["Direction"] = direction
        abn["Load"] = load
        abn["Index"] = index
        out.append(abn)
    return out

def labels_for_sample(load, direction):
    load = int(load)
    if direction.startswith('X'):
        if load == 80:
            return ['mean', '65', '95', '130']
        elif load in (65, 95, 130):
            return ['mean', str(load)]
    elif direction.startswith('Y'):
        if load == 260:
            return ['mean', '220', '300', '380']
        elif load in (220, 300, 380):
            return ['mean', str(load)]
        
def assign_gt_health(row):
    return int((row['Direction'] in ['Xa', 'Xb'] and row['Load'] == 80) or 
               (row['Direction'] in ['Ya', 'Yb'] and row['Load'] == 260))

def build_variants_for_fold(train_df, test_df, abn_df, group):
    if group.startswith('X'):
        normal = 80
        abn_loads = (65, 95, 130)
    elif group.startswith('Y'):
        normal = 260
        abn_loads = (220, 300, 380)
              
    out = {}
    cols = train_df.columns.to_list()
    cols_mean = [c for c in abn_df.columns if c.endswith("_mean")]
    train_mean = train_df.join(abn_df[cols_mean], how="left")
    test_mean  = test_df.join(abn_df[cols_mean], how="left")
    valid_mean_cols = [c for c in cols_mean if not train_mean[c].isna().all()]

    out["mean"] = (train_mean[cols + valid_mean_cols], test_mean[cols + valid_mean_cols])

    for L in abn_loads:
        cand_cols = [c for c in abn_df.columns if c.endswith(f"_{L}")]
        train_pair = train_df[train_df["Load"].isin([normal, L])].join(abn_df[cand_cols], how="left")
        test_pair = test_df.join(abn_df[cand_cols], how="left")
        valid_cols = [c for c in cand_cols if not train_pair[c].isna().all()]
        out[str(L)] = (train_pair[cols + valid_cols], test_pair[cols + valid_cols])

    return out

def scale_variant(train_var, test_var):
    target_col="GT_Health"
    key_cols = ['Direction', 'Load', 'Index']
    X_train = train_var.drop(columns=key_cols + [target_col])
    y_train = train_var[target_col]

    X_test  = test_var.drop(columns=key_cols + [target_col])
    y_test  = test_var[target_col]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    X_train_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_df  = pd.DataFrame(X_test_scaled,  index=X_test.index,  columns=X_test.columns)

    return {
        "X_train": X_train,
        "X_train_scaled_df": X_train_df,
        "X_test_scaled_df": X_test_df,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler
    }

def build_cv_pack_for_group(base_df, abn_df, group: str, n_splits=5, seed=42):
    df = base_df[base_df["Direction"] == group].copy()

    strat_key = df["Load"].astype(str)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    cv_pack = {}
    for fold_id, (tr_idx, te_idx) in enumerate(skf.split(df, strat_key), start=1):
        train_df = df.iloc[tr_idx]
        test_df  = df.iloc[te_idx]

        variants = build_variants_for_fold(train_df, test_df, abn_df, group)

        fold_pack = {}
        for variant_name, (train_var, test_var) in variants.items():
            scaled = scale_variant(train_var, test_var)
            scaled["train_index"] = train_var.index
            scaled["test_index"] = test_var.index
            fold_pack[variant_name] = scaled

        cv_pack[fold_id] = fold_pack

    return cv_pack
def generate_variant(base_df, abn_df, group):
    df = base_df[base_df["Direction"] == group].copy()
    abn = abn_df[base_df["Direction"] == group].copy()
    if group.startswith('X'):
        normal = 80
        abn_loads = (65, 95, 130)
    elif group.startswith('Y'):
        normal = 260
        abn_loads = (220, 300, 380)
    out = {}
    cols = df.columns.to_list()
    cols_mean = [c for c in abn.columns if c.endswith("_mean")]
    mean_df = df.join(abn[cols_mean], how="left")
    valid_mean_cols = [c for c in cols_mean if not mean_df[c].isna().all()]
    out["mean"] = (mean_df[cols + valid_mean_cols])
    for L in abn_loads:
        cand_cols = [c for c in abn.columns if c.endswith(f"_{L}")]
        df_abn = df[df["Load"].isin([normal, L])].join(abn[cand_cols], how="left")
        valid_cols = [c for c in cand_cols if not df_abn[c].isna().all()]
        out[str(L)] = (df_abn[cols + valid_cols])
    return out
def scale_train_data(train_df):
    target_col="GT_Health"
    key_cols = ['Direction', 'Load', 'Index']
    X_train = train_df.drop(columns=key_cols + [target_col])
    y_train = train_df[target_col]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    return {
        "X_train_raw": X_train,
        "X_train_scaled_df": X_train_df,
        "y_train": y_train
    }
def build_train_pack(base_df, abn_df, group):
    out = generate_variant(base_df, abn_df, group)
    pack = {}
    for keys, values in out.items():
        scaled = scale_train_data(values)
        pack[keys] = scaled
    return pack
def construct_HI(X_train_mean, y_train_mean, train_idx, val_idx):
    X_train = X_train_mean.iloc[train_idx]
    X_val = X_train_mean.iloc[val_idx]
    train_hp_mean_scaler = StandardScaler() 
    X_train_scaled = train_hp_mean_scaler.fit_transform(X_train) 
    X_val_scaled = train_hp_mean_scaler.transform(X_val)
    y_train = y_train_mean.iloc[train_idx]
    val_orig_idx = y_train_mean.iloc[val_idx].index

    clf = RandomForestClassifier(n_estimators=1000, class_weight="balanced", random_state=42, n_jobs = -1)
    clf.fit(X_train_scaled, y_train)

    y_proba = clf.predict_proba(X_val_scaled)[:, 1]

    return val_orig_idx, y_proba

def construct_HI_with_params(X_raw_df, y_series, train_idx, val_idx, rf_params):
    X_train = X_raw_df.iloc[train_idx]
    X_val   = X_raw_df.iloc[val_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    y_train = y_series.iloc[train_idx]
    val_orig_idx = y_series.iloc[val_idx].index

    clf = RandomForestClassifier(**rf_params)
    clf.fit(X_train_scaled, y_train)

    y_proba = clf.predict_proba(X_val_scaled)[:, 1]
    return val_orig_idx, y_proba

def compute_threshold_from_oof(oof, y):
    hi_abn  = oof.loc[y == 0].dropna()
    hi_norm = oof.loc[y == 1].dropna()
    norm_min = float(hi_norm.min())
    abn_max  = float(hi_abn.max())
    if norm_min > abn_max:
        return dict(gray_low=abn_max, gray_high=norm_min, mode="gap")
    else:
        gray_high = float(np.percentile(hi_abn, 99))
        gray_low  = float(np.percentile(hi_norm, 1))
        return dict(gray_low=gray_low, gray_high=gray_high, mode="percentile")
    
def run_pipeline_on_fold(fold_pack, group, seed, rf_params = None):
    if group.startswith('X'):
        variants = ["mean", "65", "95", "130"]
        default_fs = {"mean": 65, "65": 45, "95": 45, "130": 45}
    elif group.startswith('Y'):
        variants = ["mean", "220", "300", "380"]
        default_fs = {"mean": 65, "220": 45, "300": 45, "380": 45}

    if rf_params is None:
        rf_params = dict(n_estimators=1000, class_weight="balanced", random_state=seed, n_jobs=-1)  

    base_estimator = RandomForestClassifier(**rf_params) 
    selected_feats = {}

    for v in variants:
        Xtr_s_df = fold_pack[v]["X_train_scaled_df"]
        ytr   = fold_pack[v]["y_train"]
        k = default_fs.get(v)
        selected_feats[v] = feature_selection(Xtr_s_df, ytr, k)

    hi_oof = {}
    thr = {}

    for v in variants:
        Xtr_raw = fold_pack[v].get("X_train", None)
        ytr = fold_pack[v]["y_train"]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = pd.Series(index=Xtr_raw.index, dtype=float)
        for _, (train_idx, val_idx) in enumerate(skf.split(Xtr_raw[selected_feats[v]], ytr), 1):
            val_orig_idx, y_proba = construct_HI(Xtr_raw[selected_feats[v]], ytr, train_idx, val_idx)
            oof.loc[val_orig_idx] = y_proba * 100
        hi_oof[v] = oof
        hi_abn = oof.loc[ytr == 0].dropna()
        hi_norm = oof.loc[ytr == 1].dropna()
        norm_min = float(hi_norm.min())
        abn_max  = float(hi_abn.max())
        if norm_min > abn_max:
            thr[v] = dict(gray_low=abn_max, gray_high=norm_min, mode="gap")
        else:
            gray_high = float(np.percentile(hi_abn, 99))
            gray_low  = float(np.percentile(hi_norm, 1))
            thr[v] = dict(gray_low=gray_low, gray_high=gray_high, mode="percentile")
        del Xtr_raw

    clfs = {}
    for v in variants:
        Xtr_s_df = fold_pack[v]["X_train_scaled_df"][selected_feats[v]]
        ytr = fold_pack[v]["y_train"]
        clf = clone(base_estimator)
        clf.fit(Xtr_s_df.to_numpy(), ytr)
        clfs[v] = clf
        del Xtr_s_df
   
    Xte_mean = fold_pack["mean"]["X_test_scaled_df"]
    y_true = fold_pack["mean"]["y_test"].copy()

    HI_mean = clfs["mean"].predict_proba(Xte_mean[selected_feats["mean"]].to_numpy())[:, 1] * 100.0
    HI_mean = pd.Series(HI_mean, index=Xte_mean.index)
    decision = pd.Series("GRAY", index=Xte_mean.index)
    decision[HI_mean >= thr["mean"]["gray_high"]] = "NORMAL"
    decision[HI_mean <= thr["mean"]["gray_low"]]  = "ABNORMAL"
    gray_idx = decision[decision == "GRAY"].index
    if len(gray_idx) == 0:
        y_pred = decision.map({"NORMAL": 1, "ABNORMAL": 0}).astype(int)
        metrics = {
            "acc": accuracy_score(y_true.loc[y_pred.index], y_pred),
            "auc": roc_auc_score(y_true.loc[y_pred.index], HI_mean.loc[y_pred.index]),
            "recall": recall_score(y_true.loc[y_pred.index], y_pred),
            "cm": confusion_matrix(y_true.loc[y_pred.index], y_pred)
        }
        del clfs, hi_oof
        gc.collect()
        return y_true.loc[y_pred.index], y_pred, metrics
    del Xte_mean
    second_variants = [v for v in variants if v != "mean"]
    HI_second = {}
    gray_count = np.zeros(len(gray_idx), dtype=int)
    any_strong = np.zeros(len(gray_idx), dtype=bool)
    for v in second_variants:
        Xte_v = fold_pack[v]["X_test_scaled_df"].loc[gray_idx, selected_feats[v]]
        HI_v = clfs[v].predict_proba(Xte_v.to_numpy())[:, 1] * 100.0
        HI_second[v] = pd.Series(HI_v, index=gray_idx)
        hi_arr = HI_second[v].to_numpy()
        hi = np.asarray(hi_arr, dtype=float)
        t = thr[v]
        gray_count += ((hi > t["gray_low"]) & (hi < t["gray_high"])).astype(int)
        any_strong |= (hi <= t["gray_low"]).astype(bool)
    decision.loc[gray_idx] = np.where(any_strong | (gray_count >= 2), "ABNORMAL", "NORMAL")
    y_pred = decision.map({"NORMAL": 1, "ABNORMAL": 0}).astype(int)
    del clfs
    metrics = {
        "acc": accuracy_score(y_true.loc[y_pred.index], y_pred),
        "auc": roc_auc_score(y_true.loc[y_pred.index], HI_mean.loc[y_pred.index]),
        "recall": recall_score(y_true.loc[y_pred.index], y_pred),
        "cm": confusion_matrix(y_true.loc[y_pred.index], y_pred),
    }
    del hi_oof
    del HI_second
    gc.collect()
    return y_true.loc[y_pred.index], y_pred, metrics

def plot_radar(group_names, title, group_means_df):
    radar_df = group_means_df.loc[group_names].copy().rename(columns={
        "acc": "Accuracy",
        "auc": "AUC",
        "npv": "NPV",
        "recall": "Recall"
    })

    labels = radar_df.columns.tolist()
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1] 

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for name, row in radar_df.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=name)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title(title, size=14)
    ax.set_yticklabels([])
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.show()

def finalize_and_save(train_pack_group, group, out_dir, seed = None, centers_bands_group=None):
    os.makedirs(out_dir, exist_ok=True)

    if group.startswith("X"):
        variants = ["mean", "65", "95", "130"]
        default_fs = {"mean": 65, "65": 45, "95": 45, "130": 45}
    else:
        variants = ["mean", "220", "300", "380"]
        default_fs = {"mean": 65, "220": 45, "300": 45, "380": 45}

    rf_params = dict(
        n_estimators=1000,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1
    )
    base_estimator = RandomForestClassifier(**rf_params)
    selected_feats = {}
    for v in variants:
        X_scaled_full = train_pack_group[v]["X_train_scaled_df"]  # 這是 pack 階段的 scaler 做的
        y = train_pack_group[v]["y_train"]
        k = default_fs[v]
        selected_feats[v] = feature_selection(X_scaled_full, y, k)
        
    final_scalers = {}
    for v in variants:
        X_raw_full = train_pack_group[v]["X_train_raw"][selected_feats[v]]
        scaler = StandardScaler()
        scaler.fit(X_raw_full)
        final_scalers[v] = scaler

    thresholds = {}
    for v in variants:
        X_raw_full = train_pack_group[v]["X_train_raw"][selected_feats[v]]
        y = train_pack_group[v]["y_train"]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = pd.Series(index=X_raw_full.index, dtype=float)

        for train_idx, val_idx in skf.split(X_raw_full, y):
            val_orig_idx, y_proba = construct_HI_with_params(
                X_raw_full, y, train_idx, val_idx, rf_params=rf_params
            )
            oof.loc[val_orig_idx] = y_proba * 100.0

        thresholds[v] = compute_threshold_from_oof(oof, y)
    
    models = {}
    for v in variants:
        X_scaled = train_pack_group[v]["X_train_scaled_df"][selected_feats[v]].to_numpy()
        y = train_pack_group[v]["y_train"]

        clf = clone(base_estimator)
        clf.fit(X_scaled, y)
        models[v] = clf
    
    group_dir = os.path.join(out_dir, group)
    os.makedirs(group_dir, exist_ok=True)

    if centers_bands_group is not None:
        joblib.dump(centers_bands_group, os.path.join(group_dir, "centers_bands.pkl"))

    for v in variants:
        v_dir = os.path.join(group_dir, v)
        os.makedirs(v_dir, exist_ok=True)
        scaler_path = os.path.join(v_dir, 'scaler.pkl')
        joblib.dump(final_scalers[v], scaler_path)

        feats_path = os.path.join(v_dir, "selected_features.json") 
        with open(feats_path, "w", encoding="utf-8") as f:
            json.dump({"variant": v, "features": selected_feats[v]}, f, ensure_ascii=False, indent=2)
        model_path = os.path.join(v_dir, 'rf_model.pkl')
        joblib.dump(models[v], model_path)

        thr_path = os.path.join(v_dir, 'hi_threshold.json')
        with open(thr_path, 'w', encoding = 'utf-8') as f:
            json.dump(thresholds[v], f, ensure_ascii=False, indent = 2)

def load_group(out_dir, group):

    group_dir = os.path.join(out_dir, group)

    if not os.path.exists(group_dir):
        raise FileNotFoundError(f"Group folder not found: {group_dir}")

    artifacts = {}
    centers_bands_path = os.path.join(group_dir, "centers_bands.pkl")
    centers_bands_group = None
    if os.path.exists(centers_bands_path):
        centers_bands_group = joblib.load(centers_bands_path)
    artifacts = {"centers_bands": centers_bands_group}
    for variant in os.listdir(group_dir):
        v_dir = os.path.join(group_dir, variant)

        if not os.path.isdir(v_dir):
            continue 

        model_path = os.path.join(v_dir, "rf_model.pkl")
        scaler_path = os.path.join(v_dir, "scaler.pkl")
        threshold_path = os.path.join(v_dir, "hi_threshold.json")
        features_path = os.path.join(v_dir, "selected_features.json")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        with open(threshold_path, "r", encoding="utf-8") as f:
            threshold = json.load(f)

        with open(features_path, "r", encoding="utf-8") as f:
            selected_features = json.load(f)["features"]

        artifacts[variant] = {
            "model": model,
            "scaler": scaler,
            "threshold": threshold,
            "selected_features": selected_features
        }

    return artifacts

class FeatureContext:
    def __init__(self, signal):
        self.signal = signal
        self.fs = signal.shape[0] / 5.0
        self._cache = {}  # key -> value

    def axis(self, d):
        idx = {"x": 0, "y": 1, "z": 2}[d]
        return self.signal[:, idx]

    def get(self, key):
        return self._cache.get(key, None)

    def set(self, key, val):
        self._cache[key] = val

    def x(self, d):
        key = ("x", d)
        v = self.get(key)
        if v is None:
            v = self.axis(d)
            self.set(key, v)
        return v

    def dx(self, d):
        key = ("dx", d)
        v = self.get(key)
        if v is None:
            x = self.x(d)
            v = np.diff(x) * self.fs
            self.set(key, v)
        return v

    def basic_stats(self, d):
        key = ("basic", d)
        v = self.get(key)
        if v is None:
            x = self.x(d)
            mean_x = float(np.mean(x))
            rms_x = float(np.sqrt(np.mean(x**2)))
            std_x = float(np.std(x))
            var_x = float(np.var(x))
            abs_mean_x = float(np.mean(np.abs(x)))
            peak_x = float(np.max(np.abs(x)))
            v = (mean_x, rms_x, std_x, var_x, abs_mean_x, peak_x)
            self.set(key, v)
        return v

    def fft_vals(self, d):
        key = ("fft_vals", d)
        v = self.get(key)
        if v is None:
            x = self.x(d)
            v = np.abs(np.fft.rfft(x))
            self.set(key, v)
        return v

    def fft_mu_std(self, d):
        key = ("fft_mu_std", d)
        v = self.get(key)
        if v is None:
            fftv = self.fft_vals(d)
            mu = float(fftv.mean())
            sd = float(fftv.std(ddof=0))
            v = (mu, sd)
            self.set(key, v)
        return v

    def d3(self, d):
        key = ("d3", d)
        v = self.get(key)
        if v is None:
            x = self.x(d)
            coeffs = pywt.wavedec(x, "db4", level=3)
            v = coeffs[1] 
            self.set(key, v)
        return v

    def hist_x(self, d, bins=100):
        key = ("hist_x", d, bins)
        v = self.get(key)
        if v is None:
            x = self.x(d)
            hist, _ = np.histogram(x, bins=bins, density=True)
            v = hist
            self.set(key, v)
        return v

    def hist_fft(self, d, bins=100):
        key = ("hist_fft", d, bins)
        v = self.get(key)
        if v is None:
            fftv = self.fft_vals(d)
            hist, _ = np.histogram(fftv, bins=bins, density=True)
            v = hist
            self.set(key, v)
        return v

class AbnormalContext:
    def __init__(self, signal, centers_by_axis, bands_by_axis):
        self.signal = signal
        self.fs = signal.shape[0] / 5.0
        self.centers_by_axis = centers_by_axis
        self.bands_by_axis = bands_by_axis
        self._cache = {}

    def x(self, axis):
        idx = {"x":0, "y":1, "z":2}[axis]
        return self.signal[:, idx]

    def imfs(self, axis):
        key = ("imfs", axis)
        v = self._cache.get(key)
        if v is None:
            emd = EMD()
            v = emd.emd(self.x(axis))
            self._cache[key] = v
        return v

    def mm_all(self, axis, size):
        key = ("mm", axis, int(size))
        v = self._cache.get(key)
        if v is None:
            v = mm_operator_rms_1d(self.x(axis), int(size))  # 你原本的函數
            self._cache[key] = v
        return v

    def get_center(self, axis, label, s):
        centers = self.centers_by_axis.get(axis, {}).get(str(label), [])
        if not centers or s < 1 or s > len(centers):
            return None
        c = centers[s-1]
        if c is None or c <= 0:
            return None
        return float(c)

    def get_bands(self, axis, label):
        return self.bands_by_axis.get(axis, {}).get(str(label), [])

def make_registry():
    reg = {}

    # ---- 時域 ----
    reg["Mean"] = lambda ctx, d: ctx.basic_stats(d)[0]
    reg["RMS"]  = lambda ctx, d: ctx.basic_stats(d)[1]
    reg["Std"]  = lambda ctx, d: ctx.basic_stats(d)[2]
    reg["Var"]  = lambda ctx, d: ctx.basic_stats(d)[3]
    reg["Skewness"] = lambda ctx, d: float(skew(ctx.x(d)))
    reg["Kurtosis"] = lambda ctx, d: float(kurtosis(ctx.x(d)))

    reg["Entropy"] = lambda ctx, d: float(entropy(ctx.hist_x(d) + 1e-12))

    # 無因次指標
    def shape_factor(ctx, d):
        _, rms_x, _, _, abs_mean_x, _ = ctx.basic_stats(d)
        return float(rms_x / (abs_mean_x + 1e-12))
    reg["Shape_Factor"] = shape_factor

    def crest_factor(ctx, d):
        _, rms_x, _, _, _, peak_x = ctx.basic_stats(d)
        return float(peak_x / (rms_x + 1e-12))
    reg["CrestFactor"] = crest_factor

    def impulse_factor(ctx, d):
        _, _, _, _, abs_mean_x, peak_x = ctx.basic_stats(d)
        return float(peak_x / (abs_mean_x + 1e-12))
    reg["Impulse_Factor"] = impulse_factor

    def margin_factor(ctx, d):
        x = ctx.x(d)
        peak_x = ctx.basic_stats(d)[5]
        denom = (np.mean(np.sqrt(np.abs(x)))**2) + 1e-12
        return float(peak_x / denom)
    reg["Margin_Factor"] = margin_factor

    # 直方圖指標
    def hu(ctx, d):
        x = ctx.x(d)
        Hu, Hl = histogram_upper_lower(x)
        return float(Hu)
    def hl(ctx, d):
        x = ctx.x(d)
        Hu, Hl = histogram_upper_lower(x)
        return float(Hl)
    def hr(ctx, d):
        x = ctx.x(d)
        Hu, Hl = histogram_upper_lower(x)
        return float(Hu - Hl)
    reg["HU"] = hu
    reg["HL"] = hl
    reg["HR"] = hr

    # Hjorth
    def hjorth_activity(ctx, d):
        act, mob, comp = hjorth_params(ctx.x(d))
        return float(act)
    def hjorth_mobility(ctx, d):
        act, mob, comp = hjorth_params(ctx.x(d))
        return float(mob)
    def hjorth_complexity(ctx, d):
        act, mob, comp = hjorth_params(ctx.x(d))
        return float(comp)
    reg["Hjorth_Activity"] = hjorth_activity
    reg["Hjorth_Mobility"] = hjorth_mobility
    reg["Hjorth_Complexity"] = hjorth_complexity

    # ---- 頻域 ----
    def freq_center(ctx, d):
        x = ctx.x(d)
        dx = ctx.dx(d)
        num = np.sum(dx * x[1:])
        den = (2 * np.pi * np.sum(x[1:] ** 2) + 1e-12)
        return float(num / den)
    reg["Freq_Center"] = freq_center

    def msf(ctx, d):
        x = ctx.x(d)
        dx = ctx.dx(d)
        return float(np.sum(dx ** 2) / (4 * (np.pi ** 2) * np.sum(x[1:] ** 2) + 1e-12))

    reg["RMSF"] = lambda ctx, d: float(np.sqrt(msf(ctx, d)))

    def rvf(ctx, d):
        fc = freq_center(ctx, d)
        m = msf(ctx, d)
        return float(np.sqrt(max(m - fc ** 2, 0)))
    reg["RVF"] = rvf

    # FFT 高級頻譜指標
    def spectral_entropy(ctx, d):
        fftv = ctx.fft_vals(d)
        p = (fftv ** 2) / (np.sum(fftv ** 2) + 1e-12)
        return float(-np.sum(p * np.log(p + 1e-12)))
    reg["Spectral_Entropy"] = spectral_entropy

    reg["Shannon_Entropy"] = lambda ctx, d: float(entropy(ctx.hist_fft(d) + 1e-12))

    def spectral_skewness(ctx, d):
        fftv = ctx.fft_vals(d)
        mu, sd = ctx.fft_mu_std(d)
        return float(np.mean((fftv - mu) ** 3) / (sd ** 3 + 1e-12))
    reg["Spectral_Skewness"] = spectral_skewness

    def spectral_kurtosis(ctx, d):
        fftv = ctx.fft_vals(d)
        mu, sd = ctx.fft_mu_std(d)
        return float(np.mean((fftv - mu) ** 4) / (sd ** 4 + 1e-12) - 3.0)
    reg["Spectral_Kurtosis"] = spectral_kurtosis

    reg["Spectral_Energy"] = lambda ctx, d: float(np.sum(ctx.fft_vals(d) ** 2))

    # ---- 小波 ----
    reg["Wavelet_D3_Mean"] = lambda ctx, d: float(np.mean(ctx.d3(d)))
    reg["Wavelet_D3_Var"]  = lambda ctx, d: float(np.var(ctx.d3(d)))
    reg["Wavelet_D3_Skewness"] = lambda ctx, d: float(skew(ctx.d3(d))) if len(ctx.d3(d)) > 3 else 0.0
    reg["Wavelet_D3_Kurtosis"] = lambda ctx, d: float(kurtosis(ctx.d3(d))) if len(ctx.d3(d)) > 3 else 0.0

    # ---- ApEn ----
    def apen(ctx, d):
        x = ctx.x(d)
        x_ds = decimate(x, q=5, zero_phase=True)
        return float(ant.app_entropy(x_ds, order=2))
    reg["ApEn"] = apen

    return reg


REG = make_registry()
_MM_RE  = re.compile(r"MM_(erosion|dilation|opening|closing)_(x|y|z)_s(\d+)_(mean|\d+)")
_IMF_RE = re.compile(r"IMF_(Mean|Var|Skewness|Kurtosis)_(x|y|z)_(mean|\d+)")
def parse_abn_feature_name(name: str):
    m = _MM_RE.fullmatch(name)
    if m:
        op, axis, s, lab = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        return {"kind":"MM", "op":op, "axis":axis, "s":s, "label":str(lab)}

    m = _IMF_RE.match(name)
    if m:
        stat, axis, lab = m.group(1), m.group(2), m.group(3)
        return {"kind":"IMF", "stat":stat, "axis":axis, "label":str(lab)}

    return None

def extract_selected_features_full(signal, selected_features, centers_by_axis, bands_by_axis, variant_label, strict=False):
    ctx_base = FeatureContext(signal)  
    ctx_abn = AbnormalContext(signal, centers_by_axis, bands_by_axis)

    out = {}

    for feat in selected_features:
        try:
            base, axis = feat.rsplit("_", 1)
            if axis in ("x", "y", "z") and base in REG:
                out[feat] = REG[base](ctx_base, axis)
                continue
        except ValueError:
            pass  

        info = parse_abn_feature_name(feat)
        if info is None:
            if strict:
                raise KeyError(f"Unknown feature name: {feat}")
            out[feat] = np.nan
            continue

        if variant_label is not None and str(info.get("label")) != str(variant_label):
            out[feat] = np.nan
            continue

        if info["kind"] == "MM":
            axis = info["axis"]
            lab  = info["label"]
            s    = info["s"]
            op   = info["op"]

            c = ctx_abn.get_center(axis, lab, s)
            if c is None:
                out[feat] = np.nan
                continue

            size = int(round(ctx_abn.fs / c) * 0.6)
            if size <= 0:
                out[feat] = np.nan
                continue

            erosion, dilation, opening, closing = ctx_abn.mm_all(axis, size)
            out[feat] = {"erosion": erosion, "dilation": dilation, "opening": opening, "closing": closing}[op]
            continue

        if info["kind"] == "IMF":
            axis = info["axis"]
            lab  = info["label"]
            stat = info["stat"]

            bands = ctx_abn.get_bands(axis, lab)
            if not bands:
                out[feat] = np.nan
                continue

            imfs = ctx_abn.imfs(axis) 
            selected_imfs, _ = select_imfs_for_label(imfs, ctx_abn.fs, bands, topk=3)
            if not selected_imfs:
                out[feat] = np.nan
                continue

            x_recon = imfs[selected_imfs].sum(axis=0)
            if stat == "Mean":
                out[feat] = float(np.mean(x_recon))
            elif stat == "Var":
                out[feat] = float(np.var(x_recon))
            elif stat == "Skewness":
                out[feat] = float(skew(x_recon))
            elif stat == "Kurtosis":
                out[feat] = float(kurtosis(x_recon))
            else:
                out[feat] = np.nan

    return out


def featurize_dataset_full_from_dict(test_dict, selected_features, centers_by_axis, bands_by_axis, variant_label=None):
    rows = []
    index = []

    for k, v in test_dict.items():
        feats = extract_selected_features_full(v, selected_features, centers_by_axis=centers_by_axis, bands_by_axis=bands_by_axis, variant_label=variant_label)
        rows.append(feats)
        index.append(k)

    df = pd.DataFrame(rows, index=index)
    df = df.reindex(columns=selected_features)
    return df

def predict_by_rf(test_path, param_path, fold = True, group_prefix="X"):
    if fold:
        test = read_data(test_path)
    else:
        test = read_txt(test_path)
    if group_prefix.startswith("X"):
        artifacts_group = load_group(param_path, "Xa")
        second_variants = ["65", "95", "130"]
    else:
        artifacts_group = load_group(param_path, "Ya")
        second_variants = ["220", "300", "380"]
    art_mean = artifacts_group["mean"]
    sel_mean = art_mean["selected_features"]
    X_test_df_mean = featurize_dataset_full_from_dict(test, sel_mean, artifacts_group['centers_bands'][0], artifacts_group['centers_bands'][1], 'mean')
    X_mean = X_test_df_mean.reindex(columns=sel_mean)
    HI_mean = art_mean["model"].predict_proba(art_mean["scaler"].transform(X_mean))[:, 1] * 100.0
    HI_mean = pd.Series(HI_mean, index=X_mean.index, name="HI_mean")
    thr_mean = art_mean["threshold"]
    decision = pd.Series("Gray", index=X_mean.index, name="decision")
    decision[HI_mean >= thr_mean["gray_high"]] = "Normal"
    decision[HI_mean <= thr_mean["gray_low"]]  = "Abnormal"
    gray_idx = decision[decision == "Gray"].index
    if len(gray_idx) == 0:
        return pd.DataFrame({"HI_mean": HI_mean, "decision": decision}, index=X_mean.index)

    gray_count = np.zeros(len(gray_idx), dtype=int)
    any_strong = np.zeros(len(gray_idx), dtype=bool)

    HI_second = {}
    for v in second_variants:
        art_v = artifacts_group[v]
        sel_v = art_v["selected_features"]
        X_test_df = featurize_dataset_full_from_dict(test, sel_v, artifacts_group['centers_bands'][0], artifacts_group['centers_bands'][1], v)
        X_v = X_test_df.loc[gray_idx].reindex(columns=sel_v)
        HI_v = art_v["model"].predict_proba(art_v["scaler"].transform(X_v))[:, 1] * 100.0
        HI_v = pd.Series(HI_v, index=gray_idx, name=f"HI_{v}")
        HI_second[v] = HI_v

        thr_v = art_v["threshold"]
        hi = HI_v.to_numpy(dtype=float)

        gray_count += ((hi > thr_v["gray_low"]) & (hi < thr_v["gray_high"])).astype(int)
        any_strong |= (hi <= thr_v["gray_low"])
    decision.loc[gray_idx] = np.where(any_strong | (gray_count >= 2), "Abnormal", "Normal")

    out = pd.DataFrame({"HI_mean": HI_mean, "decision": decision}, index=X_mean.index)
    for v in second_variants:
        out[f"HI_{v}"] = HI_second[v].reindex(out.index)
    return out