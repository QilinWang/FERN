
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple 
import polars as pl
import numpy as np  
from typing import Optional, Dict, Any, List, Literal, Union, Callable, Annotated
from pydantic import BaseModel, Field, validator, computed_field, ConfigDict 
import study.configs as configs 
import textwrap
import pendulum
from functools import cached_property
import os

np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)

def to_pydantic_dict(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else (
        obj.dict() if hasattr(obj, "dict") else obj
    )

class TimeSeriesDataset(Dataset):
    """
    A PyTorch Dataset for time series data. 
    """
    def __init__(self, data_x, data_y, seq_len, label_len, pred_len, device='cuda'):
        self.data_x = data_x
        self.data_y = data_y
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        if seq_len <= 0 or label_len < 0 or pred_len <= 0:
            raise ValueError("Sequence lengths must be positive integers.") 

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        input_start = index
        input_end = input_start + self.seq_len
        target_start = input_end - self.label_len
        target_end = target_start + self.label_len + self.pred_len

        seq_x = self.data_x[input_start:input_end]
        seq_y = self.data_y[target_start:target_end]
        #!!! close for optimization
        # if not isinstance(seq_x, torch.Tensor):
        #     seq_x = torch.tensor(seq_x, dtype=torch.float32, device=self.device)
        
        # if not isinstance(seq_y, torch.Tensor):
        #     seq_y = torch.tensor(seq_y, dtype=torch.float32, device=self.device)

        return seq_x, seq_y
    
##### Config Classes #####
class SyntheticSourceConfig(BaseModel):
    type: Literal['synthetic'] = 'synthetic'
    name: Literal['lorenz', 'henon', 'rossler', 'hyper_rossler', 
        'logistic', 'duffing', 'lorenz96','chua'] = Field(
        description="Name of the dynamical system, e.g., 'lorenz'")
    params: BaseModel 
    
class CSVSourceConfig(BaseModel):
    type: Literal['csv'] = 'csv'
    name: str = Field(description="A unique name for this CSV dataset")
    path: str = Field(description="Filepath to the CSV")
    date_column: Optional[str] = 'date'
    replace_value: Optional[float] = -9999
    random_feature_num_seed_pair: Optional[Tuple[int, int]] = None

# This is the single, unified input configuration class
DatasetConfig =   Union[SyntheticSourceConfig, CSVSourceConfig] 
      
class DataSplits(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        validate_assignment=True,
    )
    train_data: torch.Tensor
    val_data: torch.Tensor 
    test_data: torch.Tensor  

    def __str__(self) -> str:
        torch.set_printoptions( precision=3)
        lines = [
            f"📊 Data Split shapes",
            f"   Train: {self.train_data.shape} | Val: {self.val_data.shape} | Test: {self.test_data.shape}",
            f"   Sample train data: {self.train_data[0:1, :].cpu().numpy()}",
            f"   Sample val data: {self.val_data[0:1, :].cpu().numpy()}",
            f"   Sample test data: {self.test_data[0:1, :].cpu().numpy()}",
        ]
        return "\n".join(lines)

class WindowedDatasets(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        validate_assignment=True,
    )
    train_dataset: TimeSeriesDataset
    val_dataset: TimeSeriesDataset
    test_dataset: TimeSeriesDataset 

    def __str__(self) -> str:
        first_train_input, first_train_target = self.train_dataset[0]
        first_val_input, first_val_target = self.val_dataset[0]
        first_test_input, first_test_target = self.test_dataset[0]
        train_shape = first_train_input.shape
        val_shape = first_val_input.shape
        test_shape = first_test_input.shape
        lines = [
            f"📊 TDE Data shapes",
            f"   Train First Input: {train_shape} | Val First Input: {val_shape} | Test First Input: {test_shape}",
            f"   Train First Target: {first_train_target.shape} | Val First Target: {first_val_target.shape} | Test First Target: {first_test_target.shape}",
        ] 
        return "\n".join(lines)

class DataLoaders(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        validate_assignment=True,
    )
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    def __str__(self) -> str:
        torch.set_printoptions(precision=3, sci_mode=False)
        np.set_printoptions(precision=3, suppress=True)

        output = []
        output.append(f"Number of batches in train_loader: {len(self.train_loader)}")
        for batch_idx, (data_batch, target_batch) in enumerate(self.train_loader):
            output.append(f"Batch {batch_idx}: Data shape {data_batch.shape}, Target shape {target_batch.shape}")
            output.append(f"Data sample: {data_batch[0, 1:2, :].cpu().numpy()}, Target sample: {target_batch[0, 1:2, :].cpu().numpy()}")
            if batch_idx == 0:
                break
        return "\n".join(output)
  
class DataSource(BaseModel):
    """
    A self-contained object representing a single data source, bundling
    its configuration with its raw data array.
    """
    config: DatasetConfig  # e.g., SyntheticSourceConfig or CSVSourceConfig
    data: np.ndarray

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    # --- Properties are defined ONCE here ---
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @cached_property
    def global_std(self) -> torch.Tensor:
        """Calculates std dev efficiently, only once per instance."""
        return np.std(self.data, axis=0)

    # --- The __str__ method provides the detailed printout ---
    def __str__(self) -> str:
        """This provides the detailed, multi-line printout for this source."""
        lines = [
            f"📊 DataSource ({self.config.name})",
            f"   ├── Shape: {self.shape}, Source Type: {self.config.type}", 
        ]
        if self.config.type == 'csv':
            lines.append(f"   └── File Path: {self.config.path}")
        elif self.config.type == 'synthetic':
            lines.append(f"   └── Parameters: {self.config.params.dict()}")
        lines.append(f"   └── Global Std: {self.global_std}")
            
        return "\n".join(lines)
    
class DataBundle(BaseModel):
    """
    A single, stateful container that flows through the data processing pipeline.
    It holds all inputs, configurations, and intermediate/final results.
    """
    def __or__(self, func: Callable[['DataBundle'], 'DataBundle']):
        """Enables the elegant `|` pipeline syntax."""
        return func(self)
    model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=False,
        )
    # --- Initial Inputs (Configuration) ---
    train_source_config: DatasetConfig # DatasetConfig # e.g., SyntheticSourceConfig or CSVSourceConfig
    val_source_config: DatasetConfig
    test_source_config: DatasetConfig
    train_split: Tuple[float, float] = (0.0, 0.7)
    val_split: Tuple[float, float] = (0.7, 0.8)
    test_split: Tuple[float, float] = (0.8, 1.0)
    scaler: StandardScaler = StandardScaler()
    no_scale: bool = True
    no_scale_val_test: bool = True
    scalar_mean: np.ndarray | torch.Tensor | None = None
    scalar_std:  np.ndarray | torch.Tensor | None = None
    scalar_min: np.ndarray | torch.Tensor | None = None
    scalar_max: np.ndarray | torch.Tensor | None = None
    training_config: configs.BaseTrainingConfig # e.g., BaseTrainingConfig holding seq_len, etc.

    # --- Fields to be populated by the pipeline ---   
    experiment_name: Optional[str] = None 
    train_source: Optional[DataSource] = None 
    val_source: Optional[DataSource] = None
    test_source: Optional[DataSource] = None

    data_splits: Optional[DataSplits] = None 

    windowed_datasets: Optional[WindowedDatasets] = None 
     
    dataloaders: Optional[DataLoaders] = None    # Will hold LoadersContainer
    
    # --- Properties --- 
    
    
    @property
    def primary_dataset_name(self) -> str:
        # This logic lives in one place
        return self.train_source.config.name

    @property
    def primary_global_std_dev(self) -> np.ndarray:
        return self.train_source.global_std
    
    @property
    def device(self) -> str:
        return self.training_config.device
    
    def print_summary(self):
        """
        Prints a summary of the data bundle, intelligently grouping sources
        that are the same object.
        """
        print("="*50)
        print("DataBundle Processing Summary")
        print("="*50)

        # --- Print Train Source (always) ---
        print("Train Source:")
        print(self.train_source)
        print("-" * 20)

        # --- Conditionally Print Validation Source ---
        if self.val_source is self.train_source:
            print("Validation Source: (Same as Train Source)")
        else:
            print("Validation Source:")
            print(self.val_source)
        print("-" * 20)

        # --- Conditionally Print Test Source ---
        if self.test_source is self.train_source:
            print("Test Source: (Same as Train Source)")
        elif self.test_source is self.val_source:
            print("Test Source: (Same as Validation Source)")
        else:
            print("Test Source:")
            print(self.test_source)
        
        print("="*50)
  
# --- Assume these helper functions exist, extracted from the old class ---
def _load_data_from_source(data_cfg: DatasetConfig) -> np.ndarray: 
    print(f"-> Loading from {data_cfg.name}")
    if data_cfg.type == 'synthetic':
        data = data_cfg.params.generate()
    elif isinstance(data_cfg, CSVSourceConfig): 
        print(f"   Scanning {data_cfg.path} with Polars...")
        lazy_frame = pl.scan_csv(data_cfg.path, 
            null_values=str(data_cfg.replace_value) 
            if data_cfg.replace_value is not None else None)
        
        if data_cfg.date_column and data_cfg.date_column in lazy_frame.columns:
            print(f"   Excluding date column: '{data_cfg.date_column}'")
            lazy_frame = lazy_frame.select(pl.exclude(data_cfg.date_column))
        
        # print("   Dropping nulls and collecting data...")
        df_polars = lazy_frame.drop_nulls().collect()
        print(f"   Final features being used: {df_polars.columns}")

        data = df_polars.to_numpy()
        data = data.astype(np.float32)

        if data_cfg.replace_value is not None: # Check if the value still exists in the final tensor
            if np.any(data == data_cfg.replace_value):
                raise ValueError(
                    f"CONSERVATIVE CHECK FAILED: The value '{data_cfg.replace_value}' was found in the final data "
                    f"tensor even after cleaning. This can happen if the raw file contains the value "
                    f"with a different string format (e.g., '{int(data_cfg.replace_value)}' vs '{data_cfg.replace_value}')."
                )
            print(f"   ✓ Conservative check passed: '{data_cfg.replace_value}' not found in final tensor.")
         
    else:
        raise ValueError(f"Unsupported data source type: {data_cfg.type}")
    return data
 
def stage_load_raw_data(bundle: DataBundle) -> DataBundle:
    """Loads data from disk or generates it based on initial configs.""" 
    # print("Loading Train source...")
    train_data_arr = _load_data_from_source(bundle.train_source_config)
    bundle.train_source = DataSource(config=bundle.train_source_config, data=train_data_arr)
    
    print("Loading Validation source...")
    if bundle.val_source_config == bundle.train_source_config:
        print(" -> Validation config is identical to Train config. Reusing.") 
        bundle.val_source = bundle.train_source # Point to the *exact same* DataSource object.
    else:
        print(" -> Validation config is different. Loading new data.")
        val_data_arr = _load_data_from_source(bundle.val_source_config)
        bundle.val_source = DataSource(config=bundle.val_source_config, data=val_data_arr)
 
    print("Loading Test source...")
    if bundle.test_source_config == bundle.train_source_config:
        print(" -> Test config is identical to Train config. Reusing.")
        bundle.test_source = bundle.train_source
    elif bundle.test_source_config == bundle.val_source_config:
        print(" -> Test config is identical to Validation config. Reusing.")
        bundle.test_source = bundle.val_source
    else:
        print(" -> Test config is different. Loading new data.")
        test_data_arr = _load_data_from_source(bundle.test_source_config)
        bundle.test_source = DataSource(config=bundle.test_source_config, data=test_data_arr)
 
    bundle.print_summary()
    return bundle

def stage_generate_experiment_name(bundle: DataBundle) -> DataBundle:
    """Generates a unique name for the experiment run."""
    print("\n--- Stage: Generating Experiment Name ---")
    cfg = bundle.training_config
    name = bundle.primary_dataset_name
    now_str = pendulum.now().format("YYYYMMDD_HHmm")
    bundle.experiment_name = f"{cfg.model_type}_{name}_seq{cfg.seq_len}_pred{cfg.pred_len}_{now_str}"
    print(f"  -> Experiment Name: {bundle.experiment_name}")
    return bundle


def stage_split_and_scale_data(bundle: DataBundle) -> DataBundle:
    """Splits data into train/val/test sets and applies scaling.""" 
    print("\n--- Stage: Splitting and Scaling Data ---") 
    
    def get_slice(source_data: np.ndarray, split_tuple: Tuple[float, float]) -> np.ndarray:
        length = len(source_data)
        start, end = int(length * split_tuple[0]), int(length * split_tuple[1])
        return source_data[start:end]
    
    def _standardize_data(data: np.ndarray, scaler: StandardScaler, to_fit: bool, no_scale: bool) -> np.ndarray:
        """
        Standardize the data using the scaler.
        If no_scale is True, return the original data.
        """
        if no_scale: return data
        if to_fit:
            scaler.fit(data)
        return scaler.transform(data)
        # return scaler.fit_transform(data) if to_fit else scaler.transform(data)

    train_arr = get_slice(bundle.train_source.data, bundle.train_split)
    val_arr = get_slice(bundle.val_source.data, bundle.val_split)
    test_arr = get_slice(bundle.test_source.data, bundle.test_split)
     
    scaled_arr = _standardize_data(train_arr, bundle.scaler, to_fit=True, no_scale=bundle.no_scale) 
    if bundle.no_scale:
        bundle.scalar_mean = train_arr.mean(0).astype(np.float32)
        bundle.scalar_std  = train_arr.std(0).astype(np.float32)
    else:
        bundle.scalar_mean = bundle.scaler.mean_.astype(np.float32)   # <-- NEW
        bundle.scalar_std  = bundle.scaler.scale_.astype(np.float32)  # <-- NEW
        
    scaled_val_arr = _standardize_data(val_arr, bundle.scaler, to_fit=False, no_scale=bundle.no_scale)
    scaled_test_arr = _standardize_data(test_arr, bundle.scaler, to_fit=False, no_scale=bundle.no_scale)
    
    
    # NEW: per-column min/max on TRAIN in original units
    bundle.scalar_min = train_arr.min(axis=0).astype(np.float32)
    bundle.scalar_max = train_arr.max(axis=0).astype(np.float32)
    
    
    bundle.data_splits = DataSplits(
        train_data=torch.tensor(scaled_arr, dtype=torch.float32, device=bundle.device),
        val_data=torch.tensor(scaled_val_arr, dtype=torch.float32, device=bundle.device),
        test_data=torch.tensor(scaled_test_arr, dtype=torch.float32, device=bundle.device)
    ) 
    print(bundle.data_splits)
    print(f"The mean of the train data: {bundle.scalar_mean}")
    print(f"The std of the train data: {bundle.scalar_std}")
    print(f"Train min : {bundle.scalar_min}")
    print(f"Train max : {bundle.scalar_max}")
    
    # NEW: warnings for large absolute values on TRAIN
    # threshold = getattr(bundle, "warn_abs_threshold", 2000.0)
    # exceed_mask = (np.abs(bundle.scalar_min) > threshold) | (np.abs(bundle.scalar_max) > threshold)
    # exceed_cols = np.where(exceed_mask)[0]
    # if exceed_cols.size > 0:
    #     print(f"[WARN] {exceed_cols.size} feature(s) exceed |value| > {threshold} on TRAIN data. Indices: {exceed_cols.tolist()}")
    #     for j in exceed_cols:
    #         print(f"  - col {j}: min={float(bundle.scalar_min[j]):.3f}, max={float(bundle.scalar_max[j]):.3f}")
    # else:
    #     print(f"All train feature ranges are within ±{threshold}.")
        
    return bundle

def stage_move_to_window_view(bundle: DataBundle) -> DataBundle:
    """Applies time-delay embedding to create TimeSeriesDatasets.""" 
    print("\n--- Stage: Moving to Window View ---") 
    cfg = bundle.training_config  

    bundle.windowed_datasets = WindowedDatasets(
        train_dataset=TimeSeriesDataset(data_x=bundle.data_splits.train_data, data_y=bundle.data_splits.train_data, 
            seq_len=cfg.seq_len, label_len=cfg.label_len, pred_len=cfg.pred_len, device=cfg.device ),
        val_dataset=TimeSeriesDataset(data_x=bundle.data_splits.val_data, data_y=bundle.data_splits.val_data, 
            seq_len=cfg.seq_len, label_len=cfg.label_len, pred_len=cfg.pred_len, device=cfg.device ),
        test_dataset=TimeSeriesDataset(data_x=bundle.data_splits.test_data, data_y=bundle.data_splits.test_data, 
            seq_len=cfg.seq_len, label_len=cfg.label_len, pred_len=cfg.pred_len, device=cfg.device )
    )
    print(bundle.windowed_datasets)
    return bundle

def stage_create_dataloaders(bundle: DataBundle) -> DataBundle:
    """Creates PyTorch DataLoaders."""
    print("\n--- Stage: Creating DataLoaders ---")
    cfg = bundle.training_config  
    bundle.dataloaders = DataLoaders(
        train_loader=DataLoader(bundle.windowed_datasets.train_dataset, 
            batch_size=cfg.batch_size, shuffle=True),
        val_loader=DataLoader(bundle.windowed_datasets.val_dataset, 
            batch_size=cfg.batch_size, shuffle=False),
        test_loader=DataLoader(bundle.windowed_datasets.test_dataset, 
            batch_size=cfg.batch_size, shuffle=False)
    )
    print(bundle.dataloaders)
    return bundle


def create_data_pipeline(
    train_config: DatasetConfig, 
    val_config: DatasetConfig,
    test_config: DatasetConfig,
    model_config: configs.BaseTrainingConfig,
    no_scale: bool = True,
    no_scale_val_test: bool = True
) -> DataBundle:
    """
    Initializes and runs the entire data processing pipeline.
    
    Returns:
        A fully populated DataBundle containing all data artifacts.
    """
    # 1. Create the initial bundle with all necessary configurations.
    initial_bundle = DataBundle(
        train_source_config=train_config,
        val_source_config=val_config,
        test_source_config=test_config,
        training_config=model_config,
        no_scale=no_scale,
        no_scale_val_test=no_scale_val_test
    )

    # 2. Execute the full pipeline in a clear, readable sequence.
    final_bundle = (
        initial_bundle
        | stage_load_raw_data
        | stage_split_and_scale_data
        | stage_move_to_window_view
        | stage_create_dataloaders  
        | stage_generate_experiment_name
    )
    
    print("\n✅ Data pipeline finished successfully.")
    return final_bundle
 
def select_random_features(
        relevant_columns: List[str], 
        random_feature_num_seed_pair: Tuple[int, int]
    ) -> List[str]:
    """
    Select random features from the dataset.
    """
    random_feature_num, random_seed = random_feature_num_seed_pair
    np.random.seed(random_seed)
    selected_columns = np.random.choice(relevant_columns, size=random_feature_num, replace=False)
    print(f"Original Num Columns: {len(relevant_columns)}, Selected Num Columns: {len(selected_columns)}")

    return selected_columns