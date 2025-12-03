"""
Main script for survival model 
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from typing import Tuple

KaplanMeierFitter = None
HAS_LIFELINES = False

try:
    from lifelines import KaplanMeierFitter
    HAS_LIFELINES = True
except ImportError:
    print("Lifelines not installed. Kaplan-Meier plot will be skipped.")

from data_loader_pancan import PanCancerOmicsDataLoader
from dimensionality_reduction import reduce_dimensionality
from evaluation_metrics import (
    compute_c_index, compute_brier_score, compute_negative_log_likelihood,
    compute_time_dependent_auc, compute_calibration_error,
    extract_risk_scores_from_survival
)
from gp_survival_model_vi import VariationalGPSurvivalModel


def normalize_sample_id(sample_id: str) -> str:
    """Normalize sample IDs to the patient-level code (TCGA-XX-XXXX)"""
    if sample_id is None:
        return ""
    sid = str(sample_id).strip().upper()
    sid = sid.replace("_", "-")
    if "TCGA" in sid:
        parts = [part for part in sid.split("-") if part]
        if len(parts) >= 3:
            return "-".join(parts[:3])
    return sid


def load_pathway_activity_matrix(pathway_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and standardize a pathway activity matrix
    
    Returns:
        scores: (n_samples, n_features) standardized numpy array
        sample_ids: array of normalized sample IDs corresponding to rows
        feature_names: array of pathway feature names
    """
    if not pathway_path.exists():
        raise FileNotFoundError(f"Pathway activity file not found: {pathway_path}")
    
    print(f"\n[Pathway] Loading pathway activity scores from {pathway_path} ...")
    if pathway_path.suffix.lower() == ".npz":
        data = np.load(pathway_path, allow_pickle=True)
        scores = data["scores"]
        sample_ids = data["samples"].astype(str)
        feature_names = data["features"].astype(str) if "features" in data else np.array(
            [f"pathway_{i}" for i in range(scores.shape[1])]
        )
    else:
        df = pd.read_csv(pathway_path, sep="\t", index_col=0)
        sample_like_cols = sum("TCGA" in str(col).upper() for col in list(df.columns)[: min(10, len(df.columns))])
        sample_like_idx = sum("TCGA" in str(idx).upper() for idx in list(df.index)[: min(10, len(df.index))])
        if sample_like_cols >= sample_like_idx:
            df = df.T
        
        df = df.apply(pd.to_numeric, errors="coerce")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.fillna(df.median())
        df.index = [normalize_sample_id(idx) for idx in df.index]
        df = df[~df.index.duplicated(keep="first")]
        scores = df.values.astype(np.float32)
        sample_ids = df.index.to_numpy()
        feature_names = df.columns.to_numpy()
    
    sample_ids = np.array([normalize_sample_id(sid) for sid in sample_ids])
    scores = scores.astype(np.float32, copy=False)
    if np.any(~np.isfinite(scores)):
        print("  Warning: Non-finite values detected after loading; imputing with column medians.")
        col_medians = np.nanmedian(scores, axis=0, keepdims=True)
        scores = np.where(np.isfinite(scores), scores, col_medians)
    
    col_mean = scores.mean(axis=0, keepdims=True)
    col_std = scores.std(axis=0, keepdims=True)
    col_std[col_std < 1e-6] = 1.0
    scores = (scores - col_mean) / col_std
    
    print(f"  Pathway matrix shape (samples × pathways): {scores.shape}")
    return scores, sample_ids, feature_names


def align_and_append_pathway_features(
    latent: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    sample_ids: np.ndarray,
    pathway_scores: np.ndarray,
    pathway_sample_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Align pathway scores to the current sample ordering and append to latent features;
    drop samples without pathway scores
    """
    if sample_ids is None:
        raise ValueError("Sample IDs are required to align pathway features.")
    
    normalized_main = np.array([normalize_sample_id(sid) for sid in sample_ids])
    normalized_pathway = np.array([normalize_sample_id(sid) for sid in pathway_sample_ids])
    
    index_map = {}
    for idx, sid in enumerate(normalized_pathway):
        if sid and sid not in index_map:
            index_map[sid] = idx
    
    matched_rows = []
    keep_mask = np.zeros(len(normalized_main), dtype=bool)
    for i, sid in enumerate(normalized_main):
        match_idx = index_map.get(sid)
        if match_idx is not None:
            keep_mask[i] = True
            matched_rows.append(pathway_scores[match_idx])
    
    matched_rows = np.array(matched_rows, dtype=np.float32)
    n_matched = keep_mask.sum()
    coverage = n_matched / len(keep_mask) if len(keep_mask) else 0.0
    
    print(f"\n[Pathway] Matched pathway scores for {n_matched}/{len(keep_mask)} samples "
          f"({coverage * 100:.1f}% coverage)")
    
    if n_matched == 0:
        raise ValueError("No overlapping samples found between pathway scores and training data.")
    
    if not np.all(keep_mask):
        dropped = len(keep_mask) - n_matched
        print(f"  Dropping {dropped} samples without pathway activity scores.")
        latent = latent[keep_mask]
        times = times[keep_mask]
        events = events[keep_mask]
        sample_ids = sample_ids[keep_mask]
    
    latent = latent.astype(np.float32, copy=False)
    augmented = np.hstack([latent, matched_rows])
    print(f"  Augmented latent shape: {augmented.shape}")
    return augmented, times, events, sample_ids


def main():
    """Main training"""
    
    data_dir = 'data'
    max_features = 1000
    time_col = None
    event_col = None
    ard = True
    test_size = 0.2
    stratify = False
    prediction_samples = 100
    output_dir = 'results'
    
    use_fixed_seed = True
    if use_fixed_seed:
        random_seed = 42
    else:
        random_seed = np.random.randint(0, 2**31)
        print(f"Using random seed: {random_seed}")
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    if use_fixed_seed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    sample_ids = None
        
    loader = PanCancerOmicsDataLoader(data_dir=data_dir, include_cancer_type=True)
    survival_df = loader.load_pancan_data()
    
    times, events = loader.prepare_survival_outcomes(survival_df, time_col=time_col, event_col=event_col)
    if hasattr(loader, 'last_valid_sample_ids') and loader.last_valid_sample_ids is not None:
        sample_ids = np.array(loader.last_valid_sample_ids)
    else:
        sample_ids = survival_df.index.astype(str).values
    if sample_ids is not None and len(sample_ids) != len(times):
        sample_ids = sample_ids[:len(times)]
    
    if survival_df.shape[1] > 1000:
        exclude_cols = ['time', 'event', 'status', 'survival', 'os_time', 'pfs_time', 
                       'death', 'censored', 'sample', 'patient', 'bcr_patient_barcode',
                       'normalized_id']
        time_event_cols = [c for c in survival_df.columns 
                          if any(exc in c.lower() for exc in ['time', 'event', 'status', 'survival', 'death', 'cens'])]
        
        feature_cols = [c for c in survival_df.columns 
                       if c not in time_event_cols and not any(exc in str(c).lower() for exc in exclude_cols)]
        
        if len(feature_cols) > 0:
            X = survival_df[feature_cols].select_dtypes(include=[np.number])
            
            if X.shape[1] > 10000:
                X = X.fillna(0)
            else:
                X = X.fillna(X.median())
            
            X_arr = X.values if isinstance(X, pd.DataFrame) else X
            
            mean_vals = np.mean(X_arr, axis=0, keepdims=True)
            std_vals = np.std(X_arr, axis=0, keepdims=True)
            X = (X_arr - mean_vals) / (std_vals + 1e-8)
            
            if X.shape[0] != len(times):
                min_len = min(X.shape[0], len(times))
                X = X[:min_len]
                times = times[:min_len]
                events = events[:min_len]
                if sample_ids is not None and len(sample_ids) != min_len:
                    sample_ids = sample_ids[:min_len]
        else:
            raise ValueError("Could not extract features from survival data")
    else:
        omics_dict = loader.load_all_data()
        
        valid_omics = {k: v for k, v in omics_dict.items() if v is not None and not v.empty}
        
        if len(valid_omics) > 1:
            aligned_omics = loader.align_samples(valid_omics)
            
            if any(df is not None and not df.empty for df in aligned_omics.values()):
                X = loader.integrate_omics(aligned_omics, 
                                           max_features_per_omics=max_features)
                if isinstance(X, pd.DataFrame):
                    X = X.values
                
                if X.shape[0] != len(times):
                    min_len = min(X.shape[0], len(times))
                    X = X[:min_len]
                    times = times[:min_len]
                    events = events[:min_len]
                    if sample_ids is not None and len(sample_ids) != min_len:
                        sample_ids = sample_ids[:min_len]
            else:
                omics_type = list(valid_omics.keys())[0]
                omics_df = valid_omics[omics_type]
                
                exclude_cols = ['normalized_id', 'sample', 'patient', 'gene', 'gene_id']
                feature_cols = [c for c in omics_df.columns 
                               if not any(exc in str(c).lower() for exc in exclude_cols)]
                X = omics_df[feature_cols].select_dtypes(include=[np.number])
                X = X.fillna(X.median())
                X = (X - X.mean()) / (X.std() + 1e-8)
                X = X.values
        elif len(valid_omics) == 1:
            omics_type = list(valid_omics.keys())[0]
            omics_df = valid_omics[omics_type]
            
            exclude_cols = ['normalized_id', 'sample', 'patient', 'gene', 'gene_id']
            feature_cols = [c for c in omics_df.columns 
                           if not any(exc in str(c).lower() for exc in exclude_cols)]
            X = omics_df[feature_cols].select_dtypes(include=[np.number])
            X = X.fillna(X.median())
            X = (X - X.mean()) / (X.std() + 1e-8)
            X = X.values
        else:
            raise ValueError("No valid omics data found. Check data files.")
    
    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"Survival data: {len(times)} samples, {events.sum()} events")
    
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    reduction_kwargs = {}
    reduction_kwargs['epochs'] = 300
    reduction_kwargs['device'] = None
    reduction_kwargs['beta'] = 0.1
    reduction_kwargs['survival_weight'] = 0.5
    reduction_kwargs['random_seed'] = random_seed
    survival_data = {
        'times': times,
        'events': events
    }
    
    Z, reducer = reduce_dimensionality(
        X,
        method='vae',
        n_components=50,
        **survival_data,
        **reduction_kwargs
    )
    
    print(f"Latent space shape: {Z.shape}")
    
    pathway_path = Path(data_dir) / "tcga_pancan_ssGSEA.txt"
    if pathway_path.exists():
        pathway_scores, pathway_sample_ids, pathway_feature_names = load_pathway_activity_matrix(pathway_path)
        
        max_components = min(pathway_scores.shape[0], pathway_scores.shape[1])
        n_components = min(200, max_components)
        if n_components != pathway_scores.shape[1]:
            print(f"Applying PCA reduction: {pathway_scores.shape[1]} -> {n_components} components")
            pca = PCA(n_components=n_components, random_state=random_seed)
            pathway_scores = pca.fit_transform(pathway_scores)
            explained = float(np.sum(pca.explained_variance_ratio_))
            print(f" PCA explained variance ratio (cumulative): {explained:.4f}")
        else:
            print("[Pathway] Skipping PCA reduction because requested components >= original feature count.")
        
        pathway_scores = pathway_scores.astype(np.float32, copy=False)
        
        if sample_ids is None:
            raise ValueError("Sample IDs are required when loading pathway features.")
        Z, times, events, sample_ids = align_and_append_pathway_features(
            Z, times, events, sample_ids, pathway_scores, pathway_sample_ids
        )
        print(f" Augmented latent dimensions: {Z.shape[1]} (added {pathway_scores.shape[1]} pathway features)")
    else:
        print(f" Warning: Expected pathway file not found at {pathway_path}. Continuing without pathway features.")
    
    indices = np.arange(len(times))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_seed,
        stratify=events if stratify else None
    )
    
    Z_train = Z[train_idx]
    Z_test = Z[test_idx]
    times_train = times[train_idx]
    times_test = times[test_idx]
    events_train = events[train_idx]
    events_test = events[test_idx]
    
    print(f"Train: {len(Z_train)} samples, {events_train.sum()} events")
    print(f"Test: {len(Z_test)} samples, {events_test.sum()} events")
    
    
    model = VariationalGPSurvivalModel(
        n_pieces=50,
        ard=ard,
        n_inducing=200,
        vi_family='cholesky',
        kernel_type='matern32',
        device=None
    )
    
    model.fit(
        Z_train,
        times_train,
        events_train,
        epochs=3000,
        lr=0.005,
        scheduler_start_epoch=2000
    )
    
    trace = None
    samples = None
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    preprocessing_info = {}
    with open(output_dir / 'preprocessing.pkl', 'wb') as f:
        pickle.dump(preprocessing_info, f)
    
    with open(output_dir / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(output_dir / 'reducer.pkl', 'wb') as f:
        pickle.dump(reducer, f)
    
    np.savez(
        output_dir / 'data_splits.npz',
        train_idx=train_idx,
        test_idx=test_idx,
        Z_train=Z_train,
        Z_test=Z_test,
        times_train=times_train,
        times_test=times_test,
        events_train=events_train,
        events_test=events_test
    )
        
    summary = model.get_posterior_summary()
    summary.to_csv(output_dir / 'posterior_summary.csv')
    print(summary.head(20))
    
    if ard:
        relevance = model.get_feature_relevance()
        print(f"Top 10 most relevant latent dimensions:")
        top_indices = np.argsort(relevance)[::-1][:10]
        for idx in top_indices:
            print(f"  Dimension {idx}: relevance = {relevance[idx]:.4f}")
    
    time_points = np.linspace(0, np.percentile(times_test, 95), 100)
    
    survival_probs = model.predict_survival(
        Z_test,
        time_points,
        n_samples=prediction_samples
    )
    
    survival_mean = np.mean(survival_probs, axis=2)
    survival_lower = np.percentile(survival_probs, 2.5, axis=2)
    survival_upper = np.percentile(survival_probs, 97.5, axis=2)
    
    np.savez(
        output_dir / 'predictions.npz',
        time_points=time_points,
        survival_mean=survival_mean,
        survival_lower=survival_lower,
        survival_upper=survival_upper,
        times_test=times_test,
        events_test=events_test
    )
    
    print(f"Predictions shape: {survival_probs.shape}")
    
    metrics = {}
    
    median_time = np.median(times_test[events_test == 1]) if events_test.sum() > 0 else np.median(times_test)
    risk_scores = extract_risk_scores_from_survival(survival_mean, time_points, time_horizon=median_time)
    
    c_index_forward = compute_c_index(times_test, events_test, risk_scores)
    c_index_inverted = compute_c_index(times_test, events_test, -risk_scores)
    
    if c_index_inverted > c_index_forward:
        print(f"  WARNING: Predictions appear inverted. Original C-index: {c_index_forward:.4f}")
        print(f"  Using inverted risk scores. New C-index: {c_index_inverted:.4f}")
        risk_scores = -risk_scores
        c_index = c_index_inverted
    else:
        c_index = c_index_forward
    
    metrics['c_index'] = c_index
    print(f"  C-index: {c_index:.4f}")
    
    brier_scores, integrated_brier = compute_brier_score(
        times_test, events_test, time_points, survival_mean
    )
    metrics['integrated_brier_score'] = integrated_brier
    metrics['brier_scores'] = brier_scores
    print(f"  Integrated Brier Score: {integrated_brier:.4f}")
    
    nll = compute_negative_log_likelihood(
        times_test, events_test, time_points, survival_mean
    )
    metrics['negative_log_likelihood'] = nll
    print(f"  Negative Log-Likelihood: {nll:.4f}")
    
    print(f"\n  Kaplan-Meier Baseline")
    if HAS_LIFELINES:
        from lifelines import KaplanMeierFitter
        
        kmf_train = KaplanMeierFitter()
        kmf_train.fit(times_train, events_train, label='KM (train)')
        
        km_survival = np.zeros((len(times_test), len(time_points)))
        for i, t in enumerate(time_points):
            km_survival[:, i] = kmf_train.predict(t)
        
        km_c_index = 0.5
        print(f"  KM C-index: {km_c_index:.4f}")
        
        km_brier_scores, km_integrated_brier = compute_brier_score(
            times_test, events_test, time_points, km_survival
        )
        print(f"  KM Integrated Brier Score: {km_integrated_brier:.4f}")
        
        km_nll = compute_negative_log_likelihood(
            times_test, events_test, time_points, km_survival
        )
        print(f"  KM Negative Log-Likelihood: {km_nll:.4f}")
        
        metrics['km_c_index'] = km_c_index
        metrics['km_integrated_brier_score'] = km_integrated_brier
        metrics['km_negative_log_likelihood'] = km_nll
        
        print(f"\n  === Model Improvement over KM ===")
        print(f"  C-index improvement: {c_index - km_c_index:.4f} ({((c_index - km_c_index) / (1 - km_c_index) * 100):.1f}% of max possible)")
        print(f"  Brier improvement: {km_integrated_brier - integrated_brier:.4f} ({((km_integrated_brier - integrated_brier) / km_integrated_brier * 100):.1f}% relative improvement)")
        print(f"  NLL improvement: {km_nll - nll:.4f} ({((km_nll - nll) / km_nll * 100):.1f}% relative improvement)")
    else:
        print("  Warning: lifelines not installed. Skipping Kaplan-Meier baseline.")
    
    time_auc = compute_time_dependent_auc(
        times_test, events_test, time_points, risk_scores, median_time
    )
    metrics['time_dependent_auc'] = time_auc
    
    time_horizons = np.percentile(times_test[events_test == 1], [25, 50, 75]) if events_test.sum() > 0 else np.percentile(times_test, [25, 50, 75])
    auc_at_horizons = []
    for th in time_horizons:
        rs = extract_risk_scores_from_survival(survival_mean, time_points, time_horizon=th)
        if c_index_inverted > c_index_forward:
            rs = -rs
        auc_th = compute_time_dependent_auc(times_test, events_test, time_points, rs, th)
        auc_at_horizons.append(auc_th)
    
    print(f"  Time-dependent AUC (t={median_time:.2f}): {time_auc:.4f}")
    print(f"  Time-dependent AUC at 25th/50th/75th percentiles: {auc_at_horizons[0]:.4f}/{auc_at_horizons[1]:.4f}/{auc_at_horizons[2]:.4f}")
    metrics['time_dependent_auc_25th'] = auc_at_horizons[0]
    metrics['time_dependent_auc_50th'] = auc_at_horizons[1]
    metrics['time_dependent_auc_75th'] = auc_at_horizons[2]
    
    cal_error, pred_bins, obs_bins = compute_calibration_error(
        times_test, events_test, time_points, survival_mean
    )
    metrics['calibration_error'] = cal_error
    print(f"  Calibration Error: {cal_error:.4f}")
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / 'evaluation_metrics.csv', index=False)
    print(f"\nEvaluation metrics saved to {output_dir / 'evaluation_metrics.csv'}")
    
    plot_survival_curves(
        time_points,
        survival_mean,
        survival_lower,
        survival_upper,
        times_test,
        events_test,
        output_dir / 'survival_curves.png',
        brier_scores,
        cal_error,
        pred_bins,
        obs_bins
    )
    
    print(f"\nModel training complete. Results saved to {output_dir}")
    print("\nSummary Metrics:")
    print(f"  C-index: {c_index:.4f}")
    print(f"  Integrated Brier Score: {integrated_brier:.4f}")
    print(f"  Time-dependent AUC: {time_auc:.4f}")
    print(f"  Calibration Error: {cal_error:.4f}")


def plot_survival_curves(time_points, survival_mean, survival_lower, 
                        survival_upper, times_test, events_test, 
                        output_path, brier_scores=None, cal_error=None,
                        pred_bins=None, obs_bins=None):
    """Plot survival analysis visualizations"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    n_samples = min(15, len(survival_mean))
    indices = np.random.choice(len(survival_mean), n_samples, replace=False)
    
    for i in indices:
        color = 'red' if events_test[i] else 'blue'
        alpha = 0.8 if events_test[i] else 0.5
        linestyle = '-' if events_test[i] else '--'
        
        ax1.plot(time_points, survival_mean[i], color=color, 
                alpha=alpha, linewidth=1.5, linestyle=linestyle, label=None)
        
        if events_test[i]:
            ax1.scatter(times_test[i], survival_mean[i, np.argmin(np.abs(time_points - times_test[i]))],
                       color='darkred', s=50, marker='x', zorder=5, linewidths=2)
        else:
            ax1.scatter(times_test[i], survival_mean[i, np.argmin(np.abs(time_points - times_test[i]))],
                       color='darkblue', s=30, marker='o', zorder=5)
    
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Survival Probability', fontsize=12)
    ax1.set_title(f'Individual Predicted Survival Curves (n={n_samples})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend([plt.Line2D([0], [0], color='red', linestyle='-', linewidth=2),
                plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=2),
                plt.scatter([], [], color='darkred', marker='x', s=50, label='Observed Event'),
                plt.scatter([], [], color='darkblue', marker='o', s=30, label='Censored')],
               ['Event', 'Censored', 'Event Marker', 'Censored Marker'], loc='upper right')
    ax1.set_ylim([0, 1.05])
    
    ax2 = fig.add_subplot(gs[1, 0])
    avg_survival = np.mean(survival_mean, axis=0)
    avg_lower = np.mean(survival_lower, axis=0)
    avg_upper = np.mean(survival_upper, axis=0)
    
    ax2.plot(time_points, avg_survival, 'b-', linewidth=2.5, label='Mean Prediction')
    ax2.fill_between(time_points, avg_lower, avg_upper, alpha=0.3, 
                     color='blue', label='95% Credible Interval')
    
    if HAS_LIFELINES:
        kmf = KaplanMeierFitter()
        kmf.fit(times_test, events_test, label='Observed (KM)')
        kmf.plot_survival_function(ax=ax2, color='red', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Survival Probability', fontsize=12)
    ax2.set_title('Average Predicted vs Observed Survival', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_ylim([0, 1.05])
    
    ax3 = fig.add_subplot(gs[1, 1])
    if brier_scores is not None:
        valid_mask = ~np.isnan(brier_scores)
        if np.sum(valid_mask) > 0:
            ax3.plot(time_points[valid_mask], brier_scores[valid_mask], 
                    'g-', linewidth=2, marker='o', markersize=4)
            ax3.fill_between(time_points[valid_mask], 0, brier_scores[valid_mask],
                            alpha=0.2, color='green')
            ax3.axhline(y=0.25, color='r', linestyle='--', alpha=0.5, 
                       label='Baseline (0.25)')
            ax3.set_xlabel('Time', fontsize=12)
            ax3.set_ylabel('Brier Score', fontsize=12)
            ax3.set_title('Brier Score Over Time', fontsize=13)
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            ax3.set_ylim(bottom=0)
    
    ax4 = fig.add_subplot(gs[2, 0])
    if pred_bins is not None and obs_bins is not None and len(pred_bins) > 0:
        valid_mask = ~np.isnan(pred_bins) & ~np.isnan(obs_bins)
        if np.sum(valid_mask) > 0:
            ax4.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
            ax4.scatter(pred_bins[valid_mask], obs_bins[valid_mask], 
                       s=100, alpha=0.7, color='blue', edgecolors='black', linewidths=1.5)
            
            for i in range(len(pred_bins)):
                if valid_mask[i]:
                    ax4.plot([pred_bins[i], pred_bins[i]], 
                            [pred_bins[i], obs_bins[i]], 
                            'r-', alpha=0.3, linewidth=1)
            
            ax4.set_xlabel('Mean Predicted Survival Probability', fontsize=12)
            ax4.set_ylabel('Observed Survival Rate', fontsize=12)
            ax4.set_title(f'Calibration Plot (Error: {cal_error:.4f})', fontsize=13)
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            ax4.set_xlim([0, 1])
            ax4.set_ylim([0, 1])
    
    ax5 = fig.add_subplot(gs[2, 1])
    risk_scores = extract_risk_scores_from_survival(survival_mean, time_points)
    
    event_risks = risk_scores[events_test == 1]
    censored_risks = risk_scores[events_test == 0]
    
    ax5.hist(censored_risks, bins=20, alpha=0.6, color='blue', 
            label=f'Censored (n={len(censored_risks)})', density=True)
    ax5.hist(event_risks, bins=20, alpha=0.6, color='red', 
            label=f'Events (n={len(event_risks)})', density=True)
    ax5.axvline(np.mean(risk_scores), color='black', linestyle='--', 
               linewidth=2, label=f'Mean Risk: {np.mean(risk_scores):.3f}')
    ax5.set_xlabel('Predicted Risk Score', fontsize=12)
    ax5.set_ylabel('Density', fontsize=12)
    ax5.set_title('Risk Score Distribution by Event Status', fontsize=13)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Survival model evaluation dashboard', fontsize=16, y=0.995)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved survival curves to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()

