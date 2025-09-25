from pathlib import Path
import time
import torch
import numpy as np
import pandas as pd
from typing import List, Dict,  Literal, Optional
import study.configs as configs
import study.train as train 
import study.data_mgr as data_mgr
import study.data_gen as data_gen 
from IPython.display import Markdown, display

SAVE_DIR = Path("results")

def run_single_seed(cfg: configs.ModelConfigType, seed: int, data_bundle) -> "SingleRunResults":
    """Runs a complete training process for a single seed and returns the results."""
    print(f"\n{'=' * 50}\n Running experiment with seed {seed} {'=' * 50}\n")

    torch.manual_seed(seed)
    np.random.seed(seed)

    trainer = train.ModelTrainer(config=cfg, data_bundle=data_bundle, seed=seed)

    single_run_result = train.SingleRunResults(
        seed=seed,
        checkpoint_settings=trainer.checkpoint,
        data_bundle=data_bundle,
    )

    start_time = time.time()
    final_bundle = trainer.train_model(data_bundle)
    end_time = time.time()

    single_run_result.final_bundle = final_bundle
    single_run_result.training_time = end_time - start_time
    single_run_result.report_training_time()

    return single_run_result
    
def run_multiple_seeds(
    cfg, 
    data_bundle,
    save_path: Path,
    include_std: bool = False,              # <<< allow AAAI‑friendly table w / or w/o ±
    to_latex: bool = True,                  # <<< emit LaTeX string for copy‑paste
    display_digits: int = 4, 
    display_digits_aux: int = 2,
    latex_digits: int = 3,
) -> list["SingleRunResults"]: 
    """Train on *all* seeds listed in `cfg.seeds`, then aggregate metrics."""
 
    total_time_acc = 0.0
    single_run_results: list["SingleRunResults"] = []
    for i, seed in enumerate(cfg.seeds):
        single_run_result = run_single_seed(cfg, seed, data_bundle)
        single_run_results.append(single_run_result)
        total_time_acc += single_run_result.training_time
    
    print(f"Total time taken: {total_time_acc:.2f} seconds")
    # --- build per-seed rows ---

    rows = []
    for result in single_run_results:
        m = result.best_test_metrics
        rows.append(
            {
                "seed": result.seed,
                "MSE": m.mse,
                "MAE": m.mae, # m.huber, 
                "SWD": m.swd,
                "EPT": m.ept,
            }
        )
    
    df_raw = pd.DataFrame(rows)
    
    # --- 3. mean / std summary ---
    mean = df_raw.mean(numeric_only=True)
    std  = df_raw.std(ddof=0, numeric_only=True)
    summary = mean.to_frame(name="value").T
    summary["seed"] = f"{cfg.pred_len}-{cfg.model_type}"

    if include_std:
        for col in ["MSE", "MAE", "SWD", "EPT"]: 
            summary[col] = summary.apply(
                lambda r, 
                c=col: f"{r[c]:.{display_digits}f} ± {std[c]:.{display_digits_aux}f}", 
                axis=1
            )
    else:
        for col in ["MSE", "MAE", "SWD", "EPT"]: 
            summary[col] = summary[col].apply(lambda x: f"{x:.{display_digits}f}")
    
    # --- 4. concat & (optionally) emit LaTeX ------------------------------
    df = pd.concat([df_raw, summary], ignore_index=True)

    if to_latex:
        latex_str = df.to_latex(
            index=False,
            escape=False,                     # <<< keep ± symbols if present
            float_format=f"%.{latex_digits}f"
        ) 
        display(Markdown(f"```latex\n{latex_str}\n```")) 

    return single_run_results, df

def exp_runner(
    train_source_config,
    seq_len: int,
    pred_len_lst: Optional[List[int]] = None, 
    seeds: Optional[List[int]] = None,
    channels: int = 7,
    display_digits: int = 2,
    display_digits_aux: int = 2,
    latex_digits: int = 3, 
    save_csv: bool = True,
    to_latex: bool = True,
    cfg_to_use: Optional[List[str]] = None,
    no_scale=True,
    no_scale_val_test=True,
):
    """
    Run experiments across models and horizons.

    Returns:
        results_dict: {pred_len -> {model_name -> list[SingleRunResults]}}
        master_df:    long dataframe with per-seed + avg rows, columns:
                      ['seed','MSE','MAE','SWD','EPT','Model','pred_len']
    """
    if pred_len_lst is None:
        pred_len_lst = [96, 192, 336, 720]
    if cfg_to_use is None:
        cfg_to_use = ["fr", "tm", "tst", "dl"]
    if seeds is None:
        seeds = [7, 1955, 2023, 4]

    csv_dir = Path("results") / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    results_dict: Dict[int, Dict[str, list]] = {}
    master_rows: List[pd.DataFrame] = []  # <<< single container

    for pred_len in pred_len_lst:
        print(f"\n{'#' * 60}\n>>> Running all models @ pred_len={pred_len}\n{'#' * 60}\n")
        results_dict[pred_len] = {}

        for cfg_name in cfg_to_use:
            # build cfgs for this horizon
            cfgs = get_default_cfg_dict(seq_len=seq_len, pred_len=pred_len, channels=channels)

            # rebuild data bundle to match this horizon
            data_bundle = data_mgr.create_data_pipeline(
                train_source_config, train_source_config, train_source_config,
                cfgs[cfg_name],
                no_scale=no_scale,
                no_scale_val_test=no_scale_val_test,
            )

            runs, df = run_multiple_seeds(
                cfg=cfgs[cfg_name],
                data_bundle=data_bundle,
                save_path=csv_dir,
                include_std=True,
                to_latex=False,   # we'll print combined later
                display_digits=display_digits,
                display_digits_aux=display_digits_aux,
                latex_digits=latex_digits,
            )
            results_dict[pred_len][cfg_name] = runs

            # annotate and stash into the single long DF
            df = df.copy()
            df["Model"] = cfg_name
            df["pred_len"] = pred_len
            master_rows.append(df)

            # per-run CSV (no overwrite)
            if save_csv:
                csv_path = csv_dir / f"{cfgs[cfg_name].model_type}_pred{pred_len}_{cfg_name}.csv"
                df.to_csv(csv_path, index=False)
                print(f"Saved CSV: {csv_path}") 

    # ===== AFTER ALL HORIZONS =====
    master_df = pd.concat(master_rows, ignore_index=True)
    source_name = train_source_config.name
    if to_latex:
        # (A) STACKED per model: seeds + avg rows for ALL horizons
        for model_key in cfg_to_use:
            mdl = master_df[master_df["Model"] == model_key]
            if mdl.empty:
                continue
            latex_stacked = mdl.to_latex(index=False, escape=False, float_format=f"%.{latex_digits}f")
            display(Markdown(f"```latex\n% STACKED (all seeds + avg) for {model_key} {source_name}\n{latex_stacked}\n```"))

        # (B) SHORT per model: only avg rows (across horizons)
        for model_key in cfg_to_use:
            mdl = master_df[(master_df["Model"] == model_key) &
                            (master_df["seed"].astype(str).str.startswith("avg-"))]
            if mdl.empty:
                continue
            latex_short = mdl.to_latex(index=False, escape=False, float_format=f"%.{latex_digits}f")
            display(Markdown(f"```latex\n% SHORT (avg rows) for {model_key} {source_name}\n{latex_short}\n```"))

    return results_dict, master_df



from typing import Dict, List

def get_default_cfg_dict(
    seq_len: int,
    pred_len: int,
    channels: int,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    seeds: List[int] = [  7, 1955, 2023, 4], # , #, 4], 20, , [2022,2023,2024,2025]
    epochs: int = 50,
    patience: int = 5,
    num_proj_swd: int = 336,
    dim_hidden: int = 144,
    dim_augment: int = 144,
    householder_reflects_latent: int = 6,
    householder_reflects_data: int = 8,
    scheduler_type: str = "none",
    warmup_epochs: int = 2,
    eta_min: float = 0.0,
    lr_scheduler_factor: float = 0.9,
    lr_scheduler_patience: int = 2,
    # loss weights
    mse_weight_backward: float = 0.0,
    mae_weight_backward: float = 0.0,
    huber_weight_backward: float = 1.0,
    swd_weight_backward: float = 0.0,
    quantile85_weight_backward: float = 0.0,
    quantile15_weight_backward: float = 0.0,
    quantile70_weight_backward: float = 0.0,
    quantile30_weight_backward: float = 0.0,
    mse_weight_validate: float = 0.1,
    mae_weight_validate: float = 1.0,
    huber_weight_validate: float = 0.0,
    swd_weight_validate: float = 0.1,
    quantile85_weight_validate: float = 0.0,
    quantile15_weight_validate: float = 0.0,
    quantile70_weight_validate: float = 0.0,
    quantile30_weight_validate: float = 0.0,
) -> Dict[str, "configs.ModelConfigType"]:

    common = dict(
        seq_len=seq_len,
        pred_len=pred_len,
        channels=channels,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seeds=seeds,
        epochs=epochs,
        patience=patience,
        num_proj_swd=num_proj_swd,
        scheduler_type=scheduler_type,
        warmup_epochs=warmup_epochs,
        eta_min=eta_min,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        # loss weights
        mse_weight_backward=mse_weight_backward,
        mae_weight_backward=mae_weight_backward,
        huber_weight_backward=huber_weight_backward,
        swd_weight_backward=swd_weight_backward,
        quantile85_weight_backward=quantile85_weight_backward,
        quantile15_weight_backward=quantile15_weight_backward,
        quantile70_weight_backward=quantile70_weight_backward,
        quantile30_weight_backward=quantile30_weight_backward,
        mse_weight_validate=mse_weight_validate,
        mae_weight_validate=mae_weight_validate,
        huber_weight_validate=huber_weight_validate,
        swd_weight_validate=swd_weight_validate,
        quantile85_weight_validate=quantile85_weight_validate,
        quantile15_weight_validate=quantile15_weight_validate,
        quantile70_weight_validate=quantile70_weight_validate,
        quantile30_weight_validate=quantile30_weight_validate,
    )

    cfg_dict = {
        "fr": configs.FERNConfig(
            **common,
            dim_hidden=dim_hidden,
            dim_augment=dim_augment,
            enable_koopman=True,
            use_complex_eigenvalues=True,
            enable_rotate_back_Koopman=True,
            decoder_mode=configs.FERNConfig.DecoderMode.FULL_ROTATION,
            householder_reflects_latent=householder_reflects_latent,
            householder_reflects_data=householder_reflects_data,
            ablate_deterministic_y0=False,
            ablate_single_encoding_layer=False,
            use_data_shift_in_z_push=True,
        ),
        "tm": configs.TimeMixerConfig(**common),
        "dl": configs.DLinearConfig(**common),
        "tst": configs.PatchTSTConfig(
            **common,
            task_name="long_term_forecast",
            factor=3,
        ),
        # --- NEW: Koopa ---
        "kp": configs.KoopaConfig(
            **common,
            mask_spectrum=[0],     # DC-only to invariant branch
            seg_len=24,
            num_blocks=2,
            dynamic_dim=64,
            hidden_dim=128,
            hidden_layers=2,
            multistep=True,
        ),
        # --- NEW: Attraos ---
        "atr": configs.AttraosConfig(
            **common,
            patch_len=24,
            e_layers=2,
            PSR_dim=3,
            PSR_delay=1,
            PSR_type="indep",
            dt_rank=32,
            d_state=64,
            FFT_evolve=True,
            multi_res=False,
        ),
    }

    return cfg_dict

 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib

def plot_final_iclr_figure(full_series, input_len, patch_len):
    """
    Generates the final, publication-quality figure with a more plausible
    forecast series that aligns better with the geometric guides.
    """
    # --- 1. ICLR-Ready Setup and Styling ---
    plt.style.use('seaborn-v0_8-paper')
    matplotlib.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 20,
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("FERN's Geometric Forecasting Mechanism", weight='bold')

    # Colorblind-friendly palette
    COLOR_INPUT = '#377eb8'    # Blue
    COLOR_PREDICT = '#ff7f00'  # Orange
    COLOR_TRUTH = '#4daf4a'    # Green
    COLOR_LATENT = '#984ea3'   # Purple
    LINEWIDTH = 2.2

    # --- 2. Panel (a): Input Series ---
    ax1 = axes[0]
    input_series = full_series[:input_len]
    ax1.plot(np.arange(input_len), input_series, color=COLOR_INPUT, linewidth=LINEWIDTH, label='Input Series')
    num_input_patches = input_len // patch_len
    for i in range(1, num_input_patches):
        ax1.axvline(x=i * patch_len - 0.5, color='grey', linestyle='--', linewidth=1.0, alpha=0.8)
    ax1.set_title('(a) Input Series', weight='bold')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.legend(loc='upper left')
    ax1.set_xlim(0, input_len)

    # --- 3. Panel (b): Latent Space Transformation ---
    ax2 = axes[1]
    np.random.seed(0)
    points = np.random.randn(200, 2)
    scale = np.array([[2.8, 0], [0, 0.9]])
    angle = np.pi / 4
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transform = rotation @ scale
    transformed_points = (transform @ points.T).T
    ax2.scatter(points[:, 0], points[:, 1], alpha=0.6, s=20, color=COLOR_INPUT, label='Isotropic Gaussian Latent')
    ax2.scatter(transformed_points[:, 0], transformed_points[:, 1], alpha=0.6, s=20, color=COLOR_LATENT, label='Anisotropic Encoded Latent')
    iso_radius = 2.5
    circ = Ellipse(xy=(0, 0), width=2*iso_radius, height=2*iso_radius, angle=0, facecolor=COLOR_INPUT, alpha=0.1)
    ax2.add_patch(circ)
    cov = np.cov(transformed_points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    ell_scale = 2.5
    ell_width = 2 * ell_scale * np.sqrt(eigvals.max())
    ell_height = 2 * ell_scale * np.sqrt(eigvals.min())
    ell_angle_deg = np.degrees(np.arctan2(eigvecs[1, eigvals.argmax()], eigvecs[0, eigvals.argmax()]))
    ell = Ellipse(xy=(0, 0), width=ell_width, height=ell_height, angle=ell_angle_deg, facecolor=COLOR_LATENT, alpha=0.15)
    ax2.add_patch(ell)
    ax2.set_title('(b) Latent Space Transformation', weight='bold')
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlabel('Latent Dimension 1')
    ax2.set_ylabel('Latent Dimension 2')
    ax2.legend(loc='upper right')

    # --- 4. Panel (c): Patchwise Geometric Forecast ---
    ax3 = axes[2]
    forecast_len = patch_len * 3
    forecast_x = np.arange(input_len, input_len + forecast_len)
    last_val = input_series[-1]
    
    # **MODIFIED LINE**: Create a U-shaped forecast to better match the ground truth trend
    # This ensures the forecast line aligns visually with all three ellipsoids.
    t_poly = np.linspace(-1.5, 1.5, forecast_len)
    parabolic_trend = 3.0 * (t_poly**2)
    t_sin = np.linspace(0, 3 * np.pi, forecast_len)
    forecast_y = last_val + parabolic_trend - 2.5 + np.sin(t_sin * 1.2) * 0.8 + (np.random.randn(forecast_len) * 0.4)
    
    ground_truth_y = full_series[input_len : input_len + forecast_len]
    
    context_x = np.arange(input_len - patch_len, input_len + 1)
    context_y = full_series[input_len - patch_len : input_len + 1]
    ax3.plot(context_x, context_y, color=COLOR_INPUT, linewidth=LINEWIDTH, label='Input')
    ax3.plot(forecast_x, forecast_y, color=COLOR_PREDICT, linewidth=LINEWIDTH, linestyle='--', label='Forecast')
    ax3.plot(forecast_x, ground_truth_y, color=COLOR_TRUTH, linewidth=LINEWIDTH, linestyle='-', label='Ground Truth')

    global_pred_std = np.std(forecast_y) + 1e-6
    for i in range(forecast_len // patch_len):
        start_idx, end_idx = i * patch_len, (i + 1) * patch_len
        patch_x_coords = forecast_x[start_idx:end_idx]
        patch_y_truth = ground_truth_y[start_idx:end_idx]
        patch_y_pred = forecast_y[start_idx:end_idx]
        
        center_x, center_y = np.mean(patch_x_coords), np.mean(patch_y_pred)
        
        slope = np.polyfit(patch_x_coords, patch_y_truth, 1)[0]
        angle_deg = np.degrees(np.arctan(slope))
        
        patch_pred_std = np.std(patch_y_pred)
        rel_unc = np.clip(patch_pred_std / global_pred_std, 0.7, 1.5)
        height = 2.5 * rel_unc

        ellipse = Ellipse(xy=(center_x, center_y), width=patch_len * 1.2, height=height,
                          angle=angle_deg, facecolor=COLOR_PREDICT, alpha=0.3)
        ax3.add_patch(ellipse)
        ax3.axvline(x=end_idx + input_len - patch_len - 0.5, color='grey', linestyle='--', linewidth=1.0, alpha=0.8)

    ax3.set_title('(c) Patchwise Geometric Forecast', weight='bold')
    ax3.set_xlabel('Time Step')
    ax3.legend(loc='upper left')
    ax3.set_xlim(input_len - patch_len, input_len + forecast_len)

    # --- 5. Final Polishing ---
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.tick_params(width=1.2)
        ax.grid(True, which='major', linestyle=':', linewidth=0.5, color='gainsboro')
        
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("fern_final_figure.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

# if __name__ == '__main__':
#     np.random.seed(42)
#     input_length = 96
#     patch_length = 24
#     total_length = 200
#     t = np.linspace(0, 10 * np.pi, total_length)
#     # Using the same data generation as before
#     time_series = 3 * np.sin(t * 0.5) + np.cos(t * 2.5) + t * 0.15 + np.random.randn(total_length) * 0.4
    
#     print("Generating the final, revised ICLR-style figure...")
#     plot_final_iclr_figure(time_series, input_length, patch_length)
#     print("Figure saved as 'fern_final_figure.pdf'")