import os
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import joblib

def read_data(folder_path):
    '''讀取資料夾中所有資料，無論是放在子資料夾或直接放在主資料夾中，並存成 dict。'''
    data_dict = {}
    subfolders_or_files = sorted(os.listdir(folder_path))

    for item in subfolders_or_files:
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            # 處理子資料夾
            files = sorted([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
            for i, file_name in enumerate(files):
                file_path = os.path.join(item_path, file_name)
                data = np.loadtxt(file_path, skiprows=1)
                key = f"data_{item}_{i + 1}"
                data_dict[key] = data
        elif os.path.isfile(item_path):
            # 處理直接放在主資料夾中的檔案
            data = np.loadtxt(item_path, skiprows=1)
            key = f"data_root_{item}"
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

def tune_data(data_dict, min_len=16175, target_len=16175):
    '''調整資料，將資料長度小於min_len的刪除，並將其餘資料長度大於min_len的資料調整為target_len。'''
    cleaned_dict = {}

    for key, data in data_dict.items():
        if isinstance(data, np.ndarray):
            m = data.shape[0]
            if m < min_len:
                # print(f"移除變數：{key}（長度為 {m}）")
                continue  
            elif m > target_len:
                data = data[:target_len, :] 
            cleaned_dict[key] = data  
    return cleaned_dict

def split_data(data_dict, test_size = 0.2):
    '''將資料分為訓練集和測試集，並返回兩個字典。'''
    data_items = list(data_dict.items()) 
    train_items, test_items = train_test_split(data_items, test_size=0.2)
    return dict(train_items), dict(test_items)

def PSD_CSD_train(data_dict, nperseg = 256, mask = None):
    '''計算每個變數的PSD和CSD，並將其標準化。返回標準化後的資料和標籤。'''
    Pxx_list, Pyy_list, Pzz_list = [], [], []
    Pxy_list, Pxz_list, Pyz_list = [], [], []
    y_list = []
    fs = min(len(v) for v in data_dict.values())/5
    for var_name, data in data_dict.items():
        parts = var_name.split("_")
        response = int(parts[1])  
    
        f, P = signal.welch(data[:, 0], fs=fs, nperseg=nperseg)
        if mask:
            freq_mask = (f >= mask[0]) & (f <= mask[1])
            P = P[freq_mask]
        Pxx_list.append(P)

        _, P = signal.welch(data[:, 1], fs=fs, nperseg=nperseg)
        if mask:
            P = P[freq_mask]
        Pyy_list.append(P)

        _, P = signal.welch(data[:, 2], fs=fs, nperseg=nperseg)
        if mask:
            P = P[freq_mask]
        Pzz_list.append(P)

        _, P = signal.csd(data[:, 0], data[:, 1], fs=fs, nperseg=nperseg)
        if mask:
            P = P[freq_mask]
        Pxy_list.append(np.abs(P))

        _, P = signal.csd(data[:, 0], data[:, 2], fs=fs, nperseg=nperseg)
        if mask:
            P = P[freq_mask]
        Pxz_list.append(np.abs(P))

        _, P = signal.csd(data[:, 1], data[:, 2], fs=fs, nperseg=nperseg)
        if mask:
            P = P[freq_mask]
        Pyz_list.append(np.abs(P))


        y_list.append(response)

    Pxx = np.stack(Pxx_list)
    Pyy = np.stack(Pyy_list)
    Pzz = np.stack(Pzz_list)
    Pxy = np.stack(Pxy_list)
    Pxz = np.stack(Pxz_list)
    Pyz = np.stack(Pyz_list)



    Pxx_norm = (Pxx - np.mean(Pxx, axis=0))/ np.std(Pxx, axis=0)
    Pyy_norm = (Pyy - np.mean(Pyy, axis=0))/ np.std(Pyy, axis=0)
    Pzz_norm = (Pzz - np.mean(Pzz, axis=0))/ np.std(Pzz, axis=0)
    Pxy_norm = (Pxy - np.mean(Pxy, axis=0))/ np.std(Pxy, axis=0)
    Pxz_norm = (Pxz - np.mean(Pxz, axis=0))/ np.std(Pxz, axis=0)
    Pyz_norm = (Pyz - np.mean(Pyz, axis=0))/ np.std(Pyz, axis=0)

    N = len(y_list)
    X = np.zeros((N, Pxx.shape[1], 3, 3))


    X[:, :, 0, 0] = Pxx_norm
    X[:, :, 1, 1] = Pyy_norm
    X[:, :, 2, 2] = Pzz_norm
    X[:, :, 1, 0] = X[:, :, 0, 1] = Pxy_norm
    X[:, :, 2, 0] = X[:, :, 0, 2] = Pxz_norm
    X[:, :, 2, 1] = X[:, :, 1, 2] = Pyz_norm
    


    scaler = {
        'Pxx': {'mean': np.mean(Pxx, axis=0), 'std': np.std(Pxx, axis=0)},
        'Pyy': {'mean': np.mean(Pyy, axis=0), 'std': np.std(Pyy, axis=0)},
        'Pzz': {'mean': np.mean(Pzz, axis=0), 'std': np.std(Pzz, axis=0)},
        'Pxy': {'mean': np.mean(Pxy, axis=0), 'std': np.std(Pxy, axis=0)},
        'Pxz': {'mean': np.mean(Pxz, axis=0), 'std': np.std(Pxz, axis=0)},
        'Pyz': {'mean': np.mean(Pyz, axis=0), 'std': np.std(Pyz, axis=0)}
    }
    
    return X, np.array(y_list), scaler

def PSD_CSD_test(data_dict, scaler, nperseg = 256, mask = None):
    '''計算每個變數的PSD和CSD，並利用訓練資料的平均和標準差將其標準化。返回標準化後的資料和標籤。'''
    name_list = []
    Pxx_list, Pyy_list, Pzz_list = [], [], []
    Pxy_list, Pxz_list, Pyz_list = [], [], []
    y_list = []
    fs = min(len(v) for v in data_dict.values())/5
    for var_name, data in data_dict.items():
        parts = var_name.split("_")
        response = int(parts[1])  

    
        f, P = signal.welch(data[:, 0], fs=fs, nperseg=nperseg)
        if mask:
            freq_mask = (f >= mask[0]) & (f <= mask[1])
            P = P[freq_mask]
        Pxx_list.append(P)
        _, P = signal.welch(data[:, 1], fs=fs, nperseg=nperseg)
        if mask:
            P = P[freq_mask]
        Pyy_list.append(P)

        _, P = signal.welch(data[:, 2], fs=fs, nperseg=nperseg)
        if mask:
            P = P[freq_mask]
        Pzz_list.append(P)

        _, P = signal.csd(data[:, 0], data[:, 1], fs=fs, nperseg=nperseg)
        if mask:
            P = P[freq_mask]
        Pxy_list.append(np.abs(P))

        _, P = signal.csd(data[:, 0], data[:, 2], fs=fs, nperseg=nperseg)
        if mask:
            P = P[freq_mask]
        Pxz_list.append(np.abs(P))

        _, P = signal.csd(data[:, 1], data[:, 2], fs=fs, nperseg=nperseg)
        if mask:
            P = P[freq_mask]
        Pyz_list.append(np.abs(P))

        name_list.append(var_name)
        y_list.append(response)

    Pxx = np.stack(Pxx_list)
    Pyy = np.stack(Pyy_list)
    Pzz = np.stack(Pzz_list)
    Pxy = np.stack(Pxy_list)
    Pxz = np.stack(Pxz_list)
    Pyz = np.stack(Pyz_list)
    Pxx_norm = (Pxx - scaler['Pxx']['mean']) / scaler['Pxx']['std']
    Pyy_norm = (Pyy - scaler['Pyy']['mean']) / scaler['Pyy']['std']
    Pzz_norm = (Pzz - scaler['Pzz']['mean']) / scaler['Pzz']['std']
    Pxy_norm = (Pxy - scaler['Pxy']['mean']) / scaler['Pxy']['std']
    Pxz_norm = (Pxz - scaler['Pxz']['mean']) / scaler['Pxz']['std']
    Pyz_norm = (Pyz - scaler['Pyz']['mean']) / scaler['Pyz']['std']
    N = len(y_list)
    X = np.zeros((N, Pxx.shape[1], 3, 3))
    X[:, :, 0, 0] = Pxx_norm
    X[:, :, 1, 1] = Pyy_norm
    X[:, :, 2, 2] = Pzz_norm
    X[:, :, 1, 0] = X[:, :, 0, 1] = Pxy_norm
    X[:, :, 2, 0] = X[:, :, 0, 2] = Pxz_norm
    X[:, :, 2, 1] = X[:, :, 1, 2] = Pyz_norm
    return X, np.array(y_list), name_list

def PSD_CSD_predict(data_dict, scaler, nperseg = 256, mask = None, label = 'X'):
    '''計算每個變數的PSD和CSD，並利用訓練資料的平均和標準差將其標準化。返回標準化後的資料和標籤。'''
    Pxx_list, Pyy_list, Pzz_list = [], [], []
    Pxy_list, Pxz_list, Pyz_list = [], [], []
    N = len(data_dict)
    if label == 'X':
        fs = 4000
    else:
        fs = 4200
    for var_name, data in data_dict.items():
        f, P = signal.welch(data[:, 0], fs=fs, nperseg=nperseg)
        if mask:
            freq_mask = (f >= mask[0]) & (f <= mask[1])
            P = P[freq_mask]
        Pxx_list.append(P)
        _, P = signal.welch(data[:, 1], fs=fs, nperseg=nperseg)
        if mask:
            P = P[freq_mask]
        Pyy_list.append(P)

        _, P = signal.welch(data[:, 2], fs=fs, nperseg=nperseg)
        if mask:
            P = P[freq_mask]
        Pzz_list.append(P)

        _, P = signal.csd(data[:, 0], data[:, 1], fs=fs, nperseg=nperseg)
        if mask:
            P = P[freq_mask]
        Pxy_list.append(np.abs(P))

        _, P = signal.csd(data[:, 0], data[:, 2], fs=fs, nperseg=nperseg)
        if mask:
            P = P[freq_mask]
        Pxz_list.append(np.abs(P))

        _, P = signal.csd(data[:, 1], data[:, 2], fs=fs, nperseg=nperseg)
        if mask:
            P = P[freq_mask]
        Pyz_list.append(np.abs(P))

    Pxx = np.stack(Pxx_list)
    Pyy = np.stack(Pyy_list)
    Pzz = np.stack(Pzz_list)
    Pxy = np.stack(Pxy_list)
    Pxz = np.stack(Pxz_list)
    Pyz = np.stack(Pyz_list)
    Pxx_norm = (Pxx - scaler['Pxx']['mean']) / scaler['Pxx']['std']
    Pyy_norm = (Pyy - scaler['Pyy']['mean']) / scaler['Pyy']['std']
    Pzz_norm = (Pzz - scaler['Pzz']['mean']) / scaler['Pzz']['std']
    Pxy_norm = (Pxy - scaler['Pxy']['mean']) / scaler['Pxy']['std']
    Pxz_norm = (Pxz - scaler['Pxz']['mean']) / scaler['Pxz']['std']
    Pyz_norm = (Pyz - scaler['Pyz']['mean']) / scaler['Pyz']['std']
    X = np.zeros((N, Pxx.shape[1], 3, 3))
    X[:, :, 0, 0] = Pxx_norm
    X[:, :, 1, 1] = Pyy_norm
    X[:, :, 2, 2] = Pzz_norm
    X[:, :, 1, 0] = X[:, :, 0, 1] = Pxy_norm
    X[:, :, 2, 0] = X[:, :, 0, 2] = Pxz_norm
    X[:, :, 2, 1] = X[:, :, 1, 2] = Pyz_norm
    return X

def data_transform_discrete(X, y, batch_size, label = 'X'):
    '''將資料轉換為PyTorch的DataLoader格式，並將標籤進行編碼。'''
    if label == 'X':
        label_map = {65: 0, 80: 1, 95: 2, 130: 3}
    elif label == 'Y':
        label_map = {220: 0, 260: 1, 300: 2, 380: 3}
    y_encoded = np.array([label_map[val] for val in y])
    X = X.reshape(-1, 1, X.shape[1], X.shape[2], X.shape[3])
    X= torch.from_numpy(X).float()
    y = torch.from_numpy(y_encoded).long()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
    return loader

def add_coordconv_channels(data):
    '''將頻率、行、列的坐標添加到資料中，並將其標準化到[-1, 1]的範圍內。'''
    N, _, D, H, W = data.size()
    
    # Normalize x/y/freq to [-1, 1]
    z_range = torch.linspace(-1, 1, D)  # frequency
    y_range = torch.linspace(-1, 1, H)  # row
    x_range = torch.linspace(-1, 1, W)  # col

    # Create meshgrid in (D, H, W) order
    zz, yy, xx = torch.meshgrid(z_range, y_range, x_range, indexing='ij')  # (D, H, W)

    # Reshape to (1, 1, D, H, W) then expand to (N, 1, D, H, W)
    zz = zz.unsqueeze(0).unsqueeze(0).expand(N, -1, -1, -1, -1)
    yy = yy.unsqueeze(0).unsqueeze(0).expand(N, -1, -1, -1, -1)
    xx = xx.unsqueeze(0).unsqueeze(0).expand(N, -1, -1, -1, -1)

    # Concatenate with input → (N, 4, D, H, W)
    return torch.cat([data, zz, yy, xx], dim=1)

def data_transform_continue(X, y, batch_size, coord = False, shuffle = False):
    '''將資料轉換為PyTorch的DataLoader格式。'''
    X= torch.from_numpy(X).float()
    y = torch.from_numpy(y).float() 
    X = X.reshape(-1, 1, X.shape[1], X.shape[2], X.shape[3])
    if coord:
        X = add_coordconv_channels(X)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle = shuffle)

    return loader

def train_model_CNND3_discrete(train_loader, model, criterion, optimizer, epochs, path = 'model.pth', device = 'cuda'):
    '''訓練模型，並將模型儲存到指定的路徑。'''
    model.train()
    for epoch in range(epochs):
        trainAcc = 0
        samples = 0
        losses = []
        for batch_num, input_data in enumerate(train_loader):

            x, y = input_data

            x = x.to(device).float()
            y = y.to(device)
            y_pre = model(x)
            loss = criterion(y_pre, y.long())
            # print("loss:", loss.item())
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainAcc += (y_pre.argmax(dim=1) == y).sum().item()
            samples += y.size(0)
    torch.save(model.state_dict(), path)
    return np.round(trainAcc/samples, 2)

def evaluate_model_discrete(model, test_loader, path, device='cuda'):
    model.load_state_dict(torch.load(path))
    model.eval()
    correct = 0
    total = 0
    merge_map = {0: 0, 1: 1, 2: 0, 3: 2}
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device).float()
            y = y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            pred_merged = torch.tensor([merge_map[p.item()] for p in pred], device=device)
            y_merged = torch.tensor([merge_map[t.item()] for t in y], device=device)
            correct += (pred_merged == y_merged).sum().item()
            total += y.size(0)
    acc = correct / total
    return np.round(acc, 2)

def groupwise_mse_loss(y_pred, y_true):
    unique_labels = y_true.unique()
    total_loss = 0.0
    for lbl in unique_labels:
        mask = (y_true == lbl)
        if mask.sum() > 0:
            total_loss += F.mse_loss(y_pred[mask], y_true[mask])
    return total_loss / len(unique_labels)

def train_model_CNND3_con(train_loader,  model, optimizer, epochs, path = 'model.pth', device = 'cuda', label = 'X', clamp = None):
    model.to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)
    model.train()
    if label == 'Y':
        Y = True
    else:
        Y = False
    if Y:
        #label_weights = {220: 1.0, 260: 1.0, 300: 1.0, 380: 0.2}  
        for epoch in range(epochs):
            all_preds = []
            all_targets = []
            for batch_num, input_data in enumerate(train_loader):
                
                x, y = input_data

                y_norm = (y - 220) / (380 - 220)
                x = x.to(device).float()
                y = y.to(device).float().view(-1, 1)
                y_norm = y_norm.to(device).float().view(-1, 1)


                y_pre = model(x)


                #loss = weighted_mse_loss(y_pre, y_norm, label_weights)
                loss = groupwise_mse_loss(y_pre, y_norm)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                y_pre_real = y_pre * (380 - 220) + 220
                if clamp:
                    y_pre_real = torch.clamp(y_pre_real, min=clamp[0], max=clamp[1])
                all_preds.append(y_pre_real.detach().cpu().numpy())
                all_targets.append(y.detach().cpu().numpy())
            scheduler.step()
            y_all = np.vstack(all_targets).flatten()
            y_pred_all = np.vstack(all_preds).flatten()
            mse = mean_squared_error(y_all, y_pred_all)

    else:
        for epoch in range(epochs):
            all_preds = []
            all_targets = []
            for batch_num, input_data in enumerate(train_loader):
                
                x, y = input_data

                y_norm = (y - 65) / (130 - 65)
                x = x.to(device).float()
                y = y.to(device).float().view(-1, 1)
                y_norm = y_norm.to(device).float().view(-1, 1)


                y_pre = model(x)

                loss = groupwise_mse_loss(y_pre, y_norm)
                #loss = criterion(y_pre, y_norm)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    
                y_pre_real = y_pre * (130 - 65) + 65
                if clamp:
                    y_pre_real = torch.clamp(y_pre_real, min=clamp[0], max=clamp[1])
                all_preds.append(y_pre_real.detach().cpu().numpy())
                all_targets.append(y.detach().cpu().numpy())
            scheduler.step()
            y_all = np.vstack(all_targets).flatten()
            y_pred_all = np.vstack(all_preds).flatten()
            mse = mean_squared_error(y_all, y_pred_all)

    torch.save(model.state_dict(), path)
    return np.round(mse, 2)

def evaluate_model_CNND3_con(model, test_loader, path, device='cuda', label = 'X', clamp = None, inner = False):
    model.load_state_dict(torch.load(path))
    model.eval()
    model.to(device)

    preds = []
    targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device).float()
            y = y.to(device).float().view(-1, 1)

            y_pred = model(x)
            if label == 'Y':
                y_pred_real = y_pred * (380 - 220) + 220
                if clamp:
                    y_pred_real = torch.clamp(y_pred_real, min=clamp[0], max=clamp[1])
            else:
                y_pred_real = y_pred * (130 - 65) + 65
                if clamp:
                    y_pred_real = torch.clamp(y_pred_real, min=clamp[0], max=clamp[1])

            preds.append(y_pred_real.cpu().numpy())
            targets.append(y.cpu().numpy())

    y_pred_all = np.vstack(preds).flatten()
    y_true_all = np.vstack(targets).flatten()

    mse = mean_squared_error(y_true_all, y_pred_all)
    mae = mean_absolute_error(y_true_all, y_pred_all)
    if inner:
        true_labels = np.unique(y_true_all)
        print("\n各 label 的 MSE：")
        for lbl in true_labels:
            mask = (y_true_all == lbl)
            mse_lbl = mean_squared_error(y_true_all[mask], y_pred_all[mask])
            print(f"label {int(lbl)} 的 MSE = {mse_lbl:.2f}")
    return np.round(mse, 2),  y_true_all, y_pred_all

def evaluate_model_CNND3_con_no_loader(model, X_test, y_test,
                                       device='cuda', label='X', clamp=None,
                                       inner=False):
    model.eval()
    model.to(device)
    X_test = X_test.reshape(-1, 1, X_test.shape[1], X_test.shape[2], X_test.shape[3])
    X_test = add_coordconv_channels(torch.from_numpy(X_test).float())
    X_tensor = torch.tensor(X_test).float().to(device)
    y_tensor = torch.tensor(y_test).float().view(-1, 1).to(device)

    with torch.no_grad():
        y_pred = model(X_tensor)

    if label == 'Y':
        y_pred_real = y_pred * (380 - 220) + 220
        y_true_real = y_tensor
        if clamp:
            y_pred_real = torch.clamp(y_pred_real, min=clamp[0], max=clamp[1])
    else:
        y_pred_real = y_pred * (130 - 65) + 65
        y_true_real = y_tensor
        if clamp:
            y_pred_real = torch.clamp(y_pred_real, min=clamp[0], max=clamp[1])

    y_pred_all = y_pred_real.cpu().numpy().flatten()
    y_true_all = y_true_real.cpu().numpy().flatten()

    mse = mean_squared_error(y_true_all, y_pred_all)

    if inner:
        print("\n各 label 的 MSE：")
        for lbl in np.unique(y_true_all):
            mask = y_true_all == lbl
            mse_lbl = mean_squared_error(y_true_all[mask], y_pred_all[mask])
            print(f"  label {int(lbl)} 的 MSE = {mse_lbl:.2f}")

    return mse, y_true_all, y_pred_all

class CNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(5, 3, 3), padding=(2,0,0))
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1,1,1))
        self.bn2 = nn.BatchNorm3d(32)
        self.pool = nn.MaxPool3d((2, 1, 1))

        self.fc1 = nn.LazyLinear(64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.drop_fc = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop_fc(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN3D_con(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(7, 3, 3), padding=(2, 1, 1))
        #self.bn1 = nn.GroupNorm(4, 16) 
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(5, 3, 3), padding=(1, 1, 1))
        #self.bn2 = nn.GroupNorm(4, 32)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3))
        self.bn3 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d((2, 1, 1))
        #self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.LazyLinear(64)  
        self.bn_fc1 = nn.BatchNorm1d(256)  
        self.drop_fc = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)  
        self.fc4 = nn.Linear(16, 1)  

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        #x = self.gap(x) 
        x = F.relu(self.bn_fc1(self.fc1(x)))
        #x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class CNN3D_con_coord(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(4, 16, kernel_size=(7, 3, 3), padding=(2, 0, 0))
        #self.bn1 = nn.GroupNorm(4, 16) 
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(5, 1, 1), padding=(1, 0, 0))
        #self.bn2 = nn.GroupNorm(4, 32)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.bn3 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d((2, 1, 1))
        self.pool_2 = nn.AvgPool3d((2, 1, 1))
        #self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.LazyLinear(256)  
        self.bn_fc1 = nn.BatchNorm1d(256)  
        self.drop_fc = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)  
        self.fc4 = nn.Linear(16, 1)  

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool_2(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool_2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        #x = self.gap(x) 
        x = F.relu(self.bn_fc1(self.fc1(x)))
        #x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def validation(data_dict, K,  label, pattern, path, mask = None, clamp = None, size = 0.2, nperseg = 256, epochs = 500, val_label = None, inner = False):
    device = 'cuda'
    train_result = []
    test_result = []
    for i in range(K):
        train_dict, test_dict = split_data(data_dict, test_size = size)
        X_train, y_train, scaler = PSD_CSD_train(train_dict, nperseg = nperseg, mask = mask)
        X_test, y_test = PSD_CSD_test(test_dict, scaler, nperseg, mask = mask)
        if pattern == 'discrete':
            train_loader = data_transform_discrete(X_train, y_train, batch_size = 16, label = label)
            test_loader = data_transform_discrete(X_test, y_test, batch_size = 16, label = label)
            CNN3D_model = CNN3D().to(device)        
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(CNN3D_model.parameters(), lr=0.0001)
            train_result.append(train_model_CNND3_discrete(train_loader, CNN3D_model, criterion, optimizer, epochs = epochs, path = path, device = device))
            test_result.append(evaluate_model_discrete(CNN3D_model, test_loader, path = path, device = device))
        elif pattern == 'continue':
            train_loader = data_transform_continue(X_train, y_train, batch_size = 16, coord=False)
            test_loader = data_transform_continue(X_test, y_test, batch_size = 16, coord=False)
            CNN3D_model = CNN3D_con().to(device)
            optimizer = torch.optim.Adam(CNN3D_model.parameters(), lr=0.001, weight_decay=1e-5)
            train_result.append(train_model_CNND3_con(train_loader, CNN3D_model, optimizer, epochs = epochs, path = path, device = device, label = label, clamp =clamp))
            mse, y_true_all, y_pred_all = evaluate_model_CNND3_con(CNN3D_model, test_loader, path=path, device=device, label=label, clamp =clamp, inner = inner)
            test_result.append(mse)
            if val_label is not None:
                mask_val = (y_true_all == val_label)
                errors_val = np.abs(y_true_all[mask_val] - y_pred_all[mask_val])

                print(f"\n第{i+1}次 label={val_label} 預測誤差：")
                for j, (true_val, pred_val, err) in enumerate(zip(y_true_all[mask_val], y_pred_all[mask_val], errors_val)):
                    print(f"  樣本 {j+1}: 真實 = {true_val:.2f}, 預測 = {pred_val:.2f}, 誤差 = {err:.2f}")
        elif pattern == 'coord':
            train_loader = data_transform_continue(X_train, y_train, batch_size = 16, coord=True)
            test_loader = data_transform_continue(X_test, y_test, batch_size = 16, coord=True)    
            CNN3D_model = CNN3D_con_coord().to(device)
            optimizer = torch.optim.Adam(CNN3D_model.parameters(), lr=0.001, weight_decay=1e-5)
            train_result.append(train_model_CNND3_con(train_loader, CNN3D_model, optimizer, epochs = epochs, path = path, device = device, label = label, clamp =clamp))
            mse, y_true_all, y_pred_all = evaluate_model_CNND3_con(CNN3D_model, test_loader, path=path, device=device, label=label, clamp =clamp, inner = inner)
            test_result.append(mse)
            if val_label is not None:
                mask_val = (y_true_all == val_label)
                errors_val = np.abs(y_true_all[mask_val] - y_pred_all[mask_val])

                print(f"\n第{i+1}次 label={val_label} 預測誤差：")
                for j, (true_val, pred_val, err) in enumerate(zip(y_true_all[mask_val], y_pred_all[mask_val], errors_val)):
                    print(f"  樣本 {j+1}: 真實 = {true_val:.2f}, 預測 = {pred_val:.2f}, 誤差 = {err:.2f}")
    df = pd.DataFrame({
        '訓練': train_result,
        '測試': test_result
    }, index=[f'第{i+1}次' for i in range(len(train_result))])
    df.loc['平均'] = df.mean()

    print(df)

def validation_kfold(data_dict, K, label, pattern, path, mask = None, clamp = None, nperseg=256, epochs=500, val_label = None, inner = False):
    device = 'cuda'
    train_result = []
    test_result = []

    all_keys = list(data_dict.keys())
    kf = KFold(n_splits=K, shuffle=True)

    for i, (train_idx, test_idx) in enumerate(kf.split(all_keys)):
        train_keys = [all_keys[j] for j in train_idx]
        test_keys = [all_keys[j] for j in test_idx]
        train_dict = {k: data_dict[k] for k in train_keys}
        test_dict = {k: data_dict[k] for k in test_keys}

        X_train, y_train, scaler = PSD_CSD_train(train_dict, nperseg=nperseg, mask=mask)
        X_test, y_test = PSD_CSD_test(test_dict, scaler, nperseg=nperseg, mask=mask)

        if pattern == 'discrete':
            train_loader = data_transform_discrete(X_train, y_train,label=label, batch_size=16)
            test_loader = data_transform_discrete(X_test, y_test,label=label, batch_size=16)
            model = CNN3D().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            train_result.append(train_model_CNND3_discrete(train_loader, model, criterion, optimizer, epochs=epochs, path=path, device=device))
            test_result.append(evaluate_model_discrete(model, test_loader, path=path, device=device))

        elif pattern == 'continue':
            train_loader = data_transform_continue(X_train, y_train, batch_size=16, coord=False)
            test_loader = data_transform_continue(X_test, y_test, batch_size=16, coord=False)
            model = CNN3D_con().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            train_result.append(train_model_CNND3_con(train_loader, model, optimizer, epochs=epochs, path=path, device=device, label=label, clamp =clamp))
            mse, y_true_all, y_pred_all = evaluate_model_CNND3_con(model, test_loader, path=path, device=device, label=label, clamp =clamp, inner=inner)
            test_result.append(mse)
            if val_label is not None:
                mask_val = (y_true_all == val_label)
                errors_val = np.abs(y_true_all[mask_val] - y_pred_all[mask_val])

                print(f"\n第{i+1}次 label={val_label} 預測誤差：")
                for j, (true_val, pred_val, err) in enumerate(zip(y_true_all[mask_val], y_pred_all[mask_val], errors_val)):
                    print(f"  樣本 {j+1}: 真實 = {true_val:.2f}, 預測 = {pred_val:.2f}, 誤差 = {err:.2f}")

        elif pattern == 'coord':
            train_loader = data_transform_continue(X_train, y_train, batch_size=16, coord=True)
            test_loader = data_transform_continue(X_test, y_test, batch_size=16, coord=True)
            model = CNN3D_con_coord().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            train_result.append(train_model_CNND3_con(train_loader, model, optimizer, epochs=epochs, path=path, device=device, label=label, clamp =clamp))
            mse, y_true_all, y_pred_all = evaluate_model_CNND3_con(model, test_loader, path=path, device=device, label=label, clamp =clamp, inner=inner)
            test_result.append(mse)
            if val_label is not None:
                mask_val = (y_true_all == val_label)
                errors_val = np.abs(y_true_all[mask_val] - y_pred_all[mask_val])

                print(f"\n第{i+1}次 label={val_label} 預測誤差：")
                for j, (true_val, pred_val, err) in enumerate(zip(y_true_all[mask_val], y_pred_all[mask_val], errors_val)):
                    print(f"  樣本 {j+1}: 真實 = {true_val:.2f}, 預測 = {pred_val:.2f}, 誤差 = {err:.2f}")

    df = pd.DataFrame({
        '訓練': train_result,
        '測試': test_result
    }, index=[f'第{i+1}次' for i in range(K)])
    df.loc['平均'] = df.mean()

    print(df)

def validation_skfold(data_dict, K, label, pattern, path, mask = None, clamp = None, nperseg=256, epochs=500, val_label = None, inner = False):
    device = 'cuda'
    train_result = []
    test_result = []

    all_keys = list(data_dict.keys())
    labels = [int(k.split('_')[1]) for k in all_keys]
    #skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=K, shuffle=True)
    for i, (train_idx, test_idx) in enumerate(skf.split(all_keys, labels)):
        train_keys = [all_keys[j] for j in train_idx]
        test_keys = [all_keys[j] for j in test_idx]
        train_dict = {k: data_dict[k] for k in train_keys}
        test_dict = {k: data_dict[k] for k in test_keys}

        X_train, y_train, scaler = PSD_CSD_train(train_dict, nperseg=nperseg, mask=mask)
        X_test, y_test, name_test = PSD_CSD_test(test_dict, scaler, nperseg=nperseg, mask=mask)

        if pattern == 'discrete':
            train_loader = data_transform_discrete(X_train, y_train,label=label, batch_size=16)
            test_loader = data_transform_discrete(X_test, y_test, label=label,batch_size=16)
            model = CNN3D().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            train_result.append(train_model_CNND3_discrete(train_loader, model, criterion, optimizer, epochs=epochs, path=path, device=device))
            test_result.append(evaluate_model_discrete(model, test_loader, path=path, device=device))

        elif pattern == 'continue':
            train_loader = data_transform_continue(X_train, y_train, batch_size=16, coord=False, shuffle = True)
            test_loader = data_transform_continue(X_test, y_test, batch_size=16, coord=False, shuffle = True)
            model = CNN3D_con().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            train_result.append(train_model_CNND3_con(train_loader, model, optimizer, epochs=epochs, path=path, device=device, label=label, clamp =clamp))
            mse, y_true_all, y_pred_all = evaluate_model_CNND3_con(model, test_loader, path=path, device=device, label=label, clamp =clamp, inner=inner)
            test_result.append(mse)
            if val_label is not None:
                mask_val = (y_true_all == val_label)
                errors_val = np.abs(y_true_all[mask_val] - y_pred_all[mask_val])
                max_idx = np.argmax(errors_val)
                print(f"\n第{i+1}次 label={val_label} 預測誤差：")
                for j, (true_val, pred_val, err) in enumerate(zip(y_true_all[mask_val], y_pred_all[mask_val], errors_val)):
                    print(f"  樣本 {j+1}: 真實 = {true_val:.2f}, 預測 = {pred_val:.2f}, 誤差 = {err:.2f}")
                print(f"\n誤差最大的是第 {max_idx} 筆，名稱為 {name_test[max_idx]}")

        elif pattern == 'coord':
            train_loader = data_transform_continue(X_train, y_train, batch_size=16, coord=True, shuffle = True)
            test_loader = data_transform_continue(X_test, y_test, batch_size=16, coord=True, shuffle = True)
            model = CNN3D_con_coord().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            train_result.append(train_model_CNND3_con(train_loader, model, optimizer, epochs=epochs, path=path, device=device, label=label, clamp =clamp))
            mse, y_true_all, y_pred_all = evaluate_model_CNND3_con(model, test_loader, path=path, device=device, label=label, clamp =clamp, inner=inner)
            test_result.append(mse)
            if val_label is not None:
                mask_val = (y_true_all == val_label)
                errors_val = np.abs(y_true_all[mask_val] - y_pred_all[mask_val])
                print(f"\n第{i+1}次 label={val_label} 預測誤差：")
                for j, (true_val, pred_val, err) in enumerate(zip(y_true_all[mask_val], y_pred_all[mask_val], errors_val)):
                    print(f"  樣本 {j+1}: 真實 = {true_val:.2f}, 預測 = {pred_val:.2f}, 誤差 = {err:.2f}")
    df = pd.DataFrame({
        '訓練': train_result,
        '測試': test_result
    }, index=[f'第{i+1}次' for i in range(K)])
    df.loc['平均'] = df.mean()
    print(df)

def validation_skfold_eva(data_dict, K, label, pattern, path, mask = None, clamp = None, nperseg=256, epochs=500, val_label = None, inner = False):
    device = 'cuda'
    train_result = []
    test_result = []

    all_keys = list(data_dict.keys())
    labels = [int(k.split('_')[1]) for k in all_keys]
    #skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=K, shuffle=True)
    for i, (train_idx, test_idx) in enumerate(skf.split(all_keys, labels)):
        train_keys = [all_keys[j] for j in train_idx]
        test_keys = [all_keys[j] for j in test_idx]
        train_dict = {k: data_dict[k] for k in train_keys}
        test_dict = {k: data_dict[k] for k in test_keys}

        X_train, y_train, scaler = PSD_CSD_train(train_dict, nperseg=nperseg, mask=mask)
        X_test, y_test, name_test = PSD_CSD_test(test_dict, scaler, nperseg=nperseg, mask=mask)

        if pattern == 'discrete':
            train_loader = data_transform_discrete(X_train, y_train, batch_size=16)
            test_loader = data_transform_discrete(X_test, y_test, batch_size=16)
            model = CNN3D().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            train_result.append(train_model_CNND3_discrete(train_loader, model, criterion, optimizer, epochs=epochs, path=path, device=device))
            test_result.append(evaluate_model_discrete(model, test_loader, path=path, device=device))

        elif pattern == 'continue':
            train_loader = data_transform_continue(X_train, y_train, batch_size=16, coord=False, shuffle = True)
            test_loader = data_transform_continue(X_test, y_test, batch_size=16, coord=False, shuffle = True)
            model = CNN3D_con().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            train_result.append(train_model_CNND3_con(train_loader, model, optimizer, epochs=epochs, path=path, device=device, label=label, clamp =clamp))
            mse, y_true_all, y_pred_all = evaluate_model_CNND3_con(model, test_loader, path=path, device=device, label=label, clamp =clamp, inner=inner)
            test_result.append(mse)
            if val_label is not None:
                mask_val = (y_true_all == val_label)
                errors_val = np.abs(y_true_all[mask_val] - y_pred_all[mask_val])
                max_idx = np.argmax(errors_val)
                print(f"\n第{i+1}次 label={val_label} 預測誤差：")
                for j, (true_val, pred_val, err) in enumerate(zip(y_true_all[mask_val], y_pred_all[mask_val], errors_val)):
                    print(f"  樣本 {j+1}: 真實 = {true_val:.2f}, 預測 = {pred_val:.2f}, 誤差 = {err:.2f}")
                print(f"\n誤差最大的是第 {max_idx} 筆，名稱為 {name_test[max_idx]}")

        elif pattern == 'coord':
            train_loader = data_transform_continue(X_train, y_train, batch_size=16, coord=True, shuffle = True)
            test_loader = data_transform_continue(X_test, y_test, batch_size=16, coord=True, shuffle = False)
            model = CNN3D_con_coord().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            train_result.append(train_model_CNND3_con(train_loader, model, optimizer, epochs=epochs, path=path, device=device, label=label, clamp =clamp))
            mse, y_true_all, y_pred_all = evaluate_model_CNND3_con_no_loader(
                model, X_test, y_test,
                device=device, label=label, clamp=clamp,
                inner=inner
            )
            test_result.append(mse)
            if val_label is not None:
                mask_val = (y_true_all == val_label)
                val_indices = np.where(mask_val)[0]
                errors_val = np.abs(y_true_all[mask_val] - y_pred_all[mask_val])
                max_idx = np.argmax(errors_val)
                original_idx = val_indices[max_idx]
                print(f"\n第{i+1}次 label={val_label} 預測誤差：")
                for j, (true_val, pred_val, err) in enumerate(zip(y_true_all[mask_val], y_pred_all[mask_val], errors_val)):
                    print(f"  樣本 {j+1}: 真實 = {true_val:.2f}, 預測 = {pred_val:.2f}, 誤差 = {err:.2f}")
                print(f"\n誤差最大的是第 {original_idx} 筆，名稱為 {name_test[original_idx]}")

    df = pd.DataFrame({
        '訓練': train_result,
        '測試': test_result
    }, index=[f'第{i+1}次' for i in range(K)])
    df.loc['平均'] = df.mean()

    print(df)

def train_dis(path, label, model_path, epochs, scaler_path=None):
    data = read_data(path)
    if label == 'X':
        data_dict = tune_data(data, 20000, 20000)
    elif label == 'Y':
        data_dict = tune_data(data, 21000, 21000)
    X_train, y_train, scaler = PSD_CSD_train(data_dict)
    if scaler_path is None:
        base, _ = os.path.splitext(model_path)  
        scaler_path = base + "_scaler.pkl"     
    joblib.dump(scaler, scaler_path)
    train_loader = data_transform_discrete(X_train, y_train, batch_size = 16, label = label)
    CNN3D_model = CNN3D().to('cuda')        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(CNN3D_model.parameters(), lr=0.0001)
    train_model_CNND3_discrete(train_loader, CNN3D_model, criterion, optimizer, epochs, path = model_path)
    print(f"訓練完成，模型已儲存至 {model_path}")
    print(f"Scaler 已儲存至 {scaler_path}")
def train_con(path, label, model_path, epochs, scaler_path=None):
    data = read_data(path)
    if label == 'X':
        data_dict = tune_data(data, 20000, 20000)
    elif label == 'Y':
        data_dict = tune_data(data, 21000, 21000)
    X_train, y_train, scaler = PSD_CSD_train(data_dict)
    if scaler_path is None:
        base, _ = os.path.splitext(model_path)  
        scaler_path = base + "_scaler.pkl"     
    joblib.dump(scaler, scaler_path)
    train_loader = data_transform_continue(X_train, y_train, batch_size=16, coord=True, shuffle = True)  
    model = CNN3D_con_coord().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    train_model_CNND3_con(train_loader, model, optimizer, epochs = epochs, path = model_path, label = label)
    print(f"訓練完成，模型已儲存至 {model_path}")
    print(f"Scaler 已儲存至 {scaler_path}")
def predict_dis(path, scaler, model_path, label, folder = False):
    if folder:   
        data = read_data(path)
    else:
        data = read_txt(path)
    if label == 'X':
        data_dict = tune_data(data, 20000, 20000)
    elif label == 'Y':
        data_dict = tune_data(data, 21000, 21000)
    if not data_dict:
        msg = "X 軸資料長度不足 20000 筆，無法預測" if label == "X" else "Y 軸資料長度不足 21000 筆，無法預測"
        return None, [], msg
    valid_keys = list(data_dict.keys())
    X = PSD_CSD_predict(data_dict, scaler, label = label)
    X = X.reshape(-1, 1, X.shape[1], X.shape[2], X.shape[3])
    X= torch.from_numpy(X).float()
    data_set = TensorDataset(X)
    if folder:
        test_loader = DataLoader(data_set, batch_size = 16, shuffle=False)
    else:
        test_loader = DataLoader(data_set, batch_size = 1, shuffle=False)

    model = CNN3D().to('cuda')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    preds = []
    with torch.no_grad():
        for (x, ) in test_loader:
            x = x.to('cuda').float()
            output = model(x)
            pred = output.argmax(dim=1)
            preds.append(pred.cpu().numpy())
    if label == 'X':
        reverse_map = {0: 65, 1: 80, 2: 95, 3: 130}
    elif label == 'Y':
        reverse_map = {0: 220, 1: 260, 2: 300, 3: 380}
    preds = np.concatenate(preds).tolist()
    mapped_preds = [reverse_map[p] for p in preds]
    return mapped_preds, valid_keys, None
def predict_con(path, scaler, model_path, label, folder = False):
    if folder:   
        data = read_data(path)
    else:
        data = read_txt(path)
    if label == 'X':
        data_dict = tune_data(data, 20000, 20000)
    elif label == 'Y':
        data_dict = tune_data(data, 21000, 21000)
    if not data_dict:
        msg = "X 軸資料長度不足 20000 筆，無法預測" if label == "X" else "Y 軸資料長度不足 21000 筆，無法預測"
        return None, [], msg
    valid_keys = list(data_dict.keys())
    X = PSD_CSD_predict(data_dict, scaler, label = label)
    X= torch.from_numpy(X).float()
    X = X.reshape(-1, 1, X.shape[1], X.shape[2], X.shape[3])
    X = add_coordconv_channels(X)
    data_set = TensorDataset(X)
    if folder:
        test_loader = DataLoader(data_set, batch_size = 16, shuffle=False)
    else:
        test_loader = DataLoader(data_set, batch_size = 1, shuffle=False)
    model = CNN3D_con_coord().to('cuda')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    preds = []
    with torch.no_grad():
        for (x, ) in test_loader:
            x = x.to('cuda').float()

            y_pred = model(x)
            if label == 'Y':
                y_pred_real = y_pred * (380 - 220) + 220
            else:
                y_pred_real = y_pred * (130 - 65) + 65
            preds.append(y_pred_real.cpu().numpy())
    return [round(float(x), 2) for x in np.concatenate(preds).flatten()], valid_keys, None