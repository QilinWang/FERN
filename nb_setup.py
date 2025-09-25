# notebook_setup.py
# A unified script for initializing a notebook environment.

# ===================================================================
# region: Consolidated Imports
# ===================================================================
import sys
import os
import importlib
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

# IPython-specific imports for setup
try:
    import IPython
    from IPython.display import Markdown, display
except ImportError:
    IPython = None
    Markdown = None
    display = lambda *args, **kwargs: print("Warning: display() is not available outside of an IPython environment.")

# Project-specific imports

# from train_util import (run_single_seed, run_multiple_seeds, 
#     get_default_cfg_dict, run_pred_len,
# )

# from study.data_mgr import (
#     _load_data_from_source, create_data_pipeline, SyntheticSourceConfig, CSVSourceConfig,
# )

# from study.configs import (
#     FERNConfig, HoloMorphConfig, DLinearConfig, NaiveConfig, PatchTSTConfig, 
#     TimeMixerConfig, BaseTrainingConfig
# )

# from study.data_gen import (
#     LorenzParams, HenonParams, RosslerParams, HyperRosslerParams,
#     LogisticParams, DuffingParams, Lorenz96Params
# )

# from study.utils import (
#     CheckpointSettings,
#     save_model,
#     load_model,
#     tde_to_seq,
# )

# endregion

# ===================================================================
# region: Environment and Reloading Functions
# ===================================================================

def setup_environment(notebook_dir: Path = None, project_root_level: int = 1):
    """
    Sets up the Python path and enables autoreload in an IPython environment.

    Args:
        notebook_dir: The directory of the current notebook. Defaults to Path.cwd().
        project_root_level: How many levels to go up from the notebook_dir to find the project root.
                            1 means the parent directory is the root.
    """
    notebook_path = notebook_dir or Path.cwd()
    
    # Traverse up the specified number of levels to find the project root
    project_root = notebook_path
    for _ in range(project_root_level):
        project_root = project_root.parent

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"✅ Added '{project_root}' to sys.path")

    # Enable autoreload if in an IPython environment
    if IPython:
        ipy = IPython.get_ipython()
        if ipy is not None:
            ipy.run_line_magic("load_ext", "autoreload")
            ipy.run_line_magic("autoreload", "2")
            print("🔄 Autoreload enabled")
    else:
        print("⚠️ Could not enable autoreload (not in an IPython environment).")


def reload_modules():
    """
    Manually reloads key project modules. Useful for development in a notebook.
    """
    print("♻️ Reloading key modules...")
    # Import modules to be reloaded within the function scope
    import study.FERN as fr
    import study.configs as configs
    import study.data_mgr as data_mgr
    import study.data_gen as data_gen
    import study.utils as utils
    import study.train as train
    import study.metrics as metrics
    import study.other_models as other_models
    import study.train_util as train_util

    # Reload them
    importlib.reload(configs)
    importlib.reload(utils)
    importlib.reload(data_mgr)
    importlib.reload(data_gen)
    importlib.reload(fr)
    importlib.reload(train_util)
    importlib.reload(train)
    importlib.reload(metrics)
    importlib.reload(other_models)
    print("✅ Modules reloaded.")

# endregion

# ===================================================================
# region: __all__ Export Control
# ===================================================================

# Controls what `from notebook_setup import *` will import into the notebook.
__all__ = [
    # Setup functions
    # "setup_environment", "reload_modules",

    # # Common utilities and libraries
    # "Path", "re", "np", "pd", "torch", "Markdown", "display",
    # "List", "Dict", "Tuple", "Optional", "Callable",

    # # Training utilities
    # # "run_single_seed", "run_multiple_seeds", "get_default_cfg_dict", "run_pred_len",

    # # Data management
    # # "_load_data_from_source", "create_data_pipeline", "SyntheticSourceConfig", "CSVSourceConfig",

    # # Model Configurations
    # "configs", "FERNConfig", "HoloMorphConfig", "DLinearConfig", "NaiveConfig",
    # "PatchTSTConfig", "TimeMixerConfig", "BaseTrainingConfig",

    # # Synthetic Data Generators
    # "LorenzParams", "HenonParams", "RosslerParams", "HyperRosslerParams",
    # "LogisticParams", "DuffingParams", "Lorenz96Params",

    # # Checkpointing and utils
    # "CheckpointSettings", "save_model", "load_model", "tde_to_seq",
]
