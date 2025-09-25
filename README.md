FERN Codebase for ICLR 2026 Submission (Anonymized)

Also available at the anonymized link:
https://anonymous.4open.science/status/FERN-1F63

This codebase provides the complete implementation of the FERN model, alongside baseline models (DLinear, PatchTST, TimeMixer) used in the experiments. The primary implementation is in Python, using PyTorch.

We provide a training log in markdown alongside.

🔧 Setup and Usage

Clone the repository or unzip the downloaded folder.

Install the required libraries using the provided requirements file.


For notebook usage, we use something like this:
 
 
```python
from nb_setup import setup_environment, reload_modules
setup_environment()

import study.data_mgr as data_mgr
import study.data_gen as data_gen
import study.configs as configs
from study.configs import FERNConfig
import study.train_util as train_util
from pathlib import Path

# Base config
cfg_dict = train_util.get_default_cfg_dict(seq_len=336, pred_len=336, channels=7)

# Data source
train_source_config = data_mgr.CSVSourceConfig(
    name="etth2",
    type="csv",
    path="data/ETT/ETTh2.csv",
    date_column="date",
    replace_value=-9999,
)

# Pipeline  
data_bundle = data_mgr.create_data_pipeline(
    train_source_config, train_source_config, train_source_config, cfg_dict["fr"],
    no_scale=True,
    no_scale_val_test=True,
)

# Experiment runner
lsts, dfs = train_util.exp_runner(
    train_source_config=train_source_config,
    seq_len=336,
    pred_len_lst=[192],
    seeds=[7, 1955, 2023, 4],
    channels=7,
    display_digits=2,
    display_digits_aux=2,
    latex_digits=2,
    save_csv=True,
    to_latex=True,
    cfg_to_use=["fr"],  # options: "fr", "tm", "tst", "dl"
)
```


