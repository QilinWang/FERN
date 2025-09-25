import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# ====================================================================================
# DATA LOADING (Minimized from data_mgr.py)
# ====================================================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        # Assuming data is already a tensor on the correct device
        self.data = data 
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        return seq_x, seq_y

def load_etth2_data(file_path, seq_len, pred_len, batch_size, device='cuda', scale=False):
    # Load data using pandas
    try:
        df_raw = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {file_path}. Please download ETTh2.csv.")
        
    # Prepare data
    if 'date' in df_raw.columns:
        df_data = df_raw.drop('date', axis=1)
    else:
        df_data = df_raw
        
    data = df_data.values.astype(np.float32)
    channels = data.shape[1]
    
    # Define splits (Standard ETT chronological splits)
    # ETTh2: 12 months train, 4 months val, 4 months test (hourly data)
    num_train = 12 * 30 * 24
    num_vali = 4 * 30 * 24
    num_test = 4 * 30 * 24
    
    # Define indices for splitting, ensuring lookback window is available
    train_end = num_train
    val_start = num_train - seq_len
    val_end = num_train + num_vali
    test_start = num_train + num_vali - seq_len
    test_end = num_train + num_vali + num_test

    # Ensure we don't exceed the bounds of the data
    if test_end > len(data):
        test_end = len(data)
    
    train_data = data[0:train_end]
    val_data = data[val_start:val_end]
    test_data = data[test_start:test_end]
    
    # Calculate global std dev on unscaled training data for EPT metric
    global_std = np.std(train_data, axis=0)

    # Scaling (The paper emphasizes training ETT on original scales, so scale=False by default)
    if scale:
        scaler = StandardScaler()
        scaler.fit(train_data)
        train_data_processed = scaler.transform(train_data)
        val_data_processed = scaler.transform(val_data)
        test_data_processed = scaler.transform(test_data)
    else:
        train_data_processed = train_data
        val_data_processed = val_data
        test_data_processed = test_data
    
    # Convert to tensors and move to device
    train_tensor = torch.tensor(train_data_processed, dtype=torch.float32, device=device)
    val_tensor = torch.tensor(val_data_processed, dtype=torch.float32, device=device)
    test_tensor = torch.tensor(test_data_processed, dtype=torch.float32, device=device)
    
    # Create Datasets and Loaders
    train_dataset = TimeSeriesDataset(train_tensor, seq_len, pred_len)
    val_dataset = TimeSeriesDataset(val_tensor, seq_len, pred_len)
    test_dataset = TimeSeriesDataset(test_tensor, seq_len, pred_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        
    return train_loader, val_loader, test_loader, global_std, channels

# ====================================================================================
# METRICS (Minimized from metrics.py)
# ====================================================================================

# --- Sliced Wasserstein Distance (SWD) ---

def rand_proj(input_dim: int, num_proj: int, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=device); g.manual_seed(seed)
    proj = torch.randn(input_dim, num_proj, device=device, generator=g)
    proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-9)
    return proj

def _project_along_axis(x: torch.Tensor, proj_mat: torch.Tensor, feature_axis: int) -> torch.Tensor:
    x_feat_last = torch.moveaxis(x, feature_axis, -1)
    z = x_feat_last @ proj_mat
    z = torch.moveaxis(z, -1, feature_axis)
    return z

class SWDMetric(nn.Module):
    # Calculates SWD between (B, S, D) tensors. Projects features (D), sorts horizon (S).
    def __init__(self, feature_dim: int, num_proj: int, seed: int, device: torch.device):
        super().__init__()
        self.feature_axis = 2 # D
        self.point_axis = 1   # S
        
        proj_mat = rand_proj(feature_dim, num_proj, seed=seed, device=device)
        self.register_buffer("proj_mat", proj_mat)

    def forward(self, y_pred: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
        assert y_pred.shape == y_real.shape
        
        # 1) Project features (D) -> L projections
        z_pred = _project_along_axis(y_pred, self.proj_mat, self.feature_axis)
        z_real = _project_along_axis(y_real, self.proj_mat, self.feature_axis)
        
        # 2) Sort along horizon (S)
        z_pred_sorted, _ = torch.sort(z_pred, dim=self.point_axis)
        z_real_sorted, _ = torch.sort(z_real, dim=self.point_axis)
        
        # 3) Calculate squared difference and mean (Squared SWD-2)
        diff = z_pred_sorted - z_real_sorted
        sq_swd2 = diff.pow(2).mean()
        return sq_swd2

# --- Effective Prediction Time (EPT) ---

class EPTMetric(nn.Module):
    # Calculates EPT: average steps before error exceeds 1 std dev of true data.
    def __init__(self, global_std: np.ndarray, device: torch.device):
        super().__init__()
        # Threshold epsilon is 1 standard deviation per channel
        self.register_buffer("epsilon", torch.tensor(global_std, dtype=torch.float32, device=device))

    def forward(self, y_pred: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
        # Input shapes: (B, S, D)
        B, S, D = y_pred.shape
        
        # Calculate absolute error
        abs_error = torch.abs(y_pred - y_real)
        
        # Check if error exceeds threshold (epsilon broadcasts across B and S)
        exceeds_threshold = abs_error > self.epsilon.view(1, 1, D)
        
        # Find the first time step (s) where the threshold is exceeded
        indices = torch.arange(S, device=y_pred.device).view(1, S, 1).expand(B, S, D)
        
        # Mask indices where threshold is NOT exceeded, set them to S
        masked_indices = torch.where(exceeds_threshold, indices, S)
        
        # Find the minimum index along dim S. If never exceeded, EPT is S.
        ept_bd = torch.min(masked_indices, dim=1).values # (B, D)
        
        # Average EPT across all batches and dimensions
        avg_ept = ept_bd.float().mean()
        return avg_ept

def calculate_metrics(pred, true, global_std, device, num_proj=1500, seed=42):
    mse_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()
    swd_fn = SWDMetric(feature_dim=pred.shape[2], num_proj=num_proj, seed=seed, device=device)
    ept_fn = EPTMetric(global_std=global_std, device=device)

    mse = mse_fn(pred, true).item()
    mae = mae_fn(pred, true).item()
    swd = swd_fn(pred, true).item()
    ept = ept_fn(pred, true).item()
    
    return {"MSE": mse, "MAE": mae, "SWD": swd, "EPT": ept}