import numpy as np
from typing import Tuple, Optional
from scipy import stats
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def compute_c_index(times_true: np.ndarray, events_true: np.ndarray, 
                    risk_scores: np.ndarray) -> float:
    """
    Compute C-index for survival predictions
    
    Args:
        times_true: True survival times
        events_true: True event indicators (1 = event, 0 = censored)
        risk_scores: Predicted risk scores 
    
    Returns:
        c_index: C-index between 0 and 1
    """
    n = len(times_true)
    if n < 2:
        return 0.5
    
    concordant = 0
    comparable = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if events_true[i] == 1 and events_true[j] == 1:
                if times_true[i] < times_true[j]:
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] < risk_scores[j]:
                        concordant += 0
                    else:
                        concordant += 0.5
                    comparable += 1
                elif times_true[j] < times_true[i]:
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1
                    elif risk_scores[j] < risk_scores[i]:
                        concordant += 0
                    else:
                        concordant += 0.5
                    comparable += 1
            
            elif events_true[i] == 1 and events_true[j] == 0:
                if times_true[i] <= times_true[j]:
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] < risk_scores[j]:
                        concordant += 0
                    else:
                        concordant += 0.5
                    comparable += 1
            
            elif events_true[i] == 0 and events_true[j] == 1:
                if times_true[j] <= times_true[i]:
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1
                    elif risk_scores[j] < risk_scores[i]:
                        concordant += 0
                    else:
                        concordant += 0.5
                    comparable += 1
    
    if comparable == 0:
        return 0.5
    
    return concordant / comparable


def compute_brier_score(times_true: np.ndarray, events_true: np.ndarray,
                       time_points: np.ndarray, 
                       survival_probs: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute Brier score at specific time points
    
    Args:
        times_true: True survival times
        events_true: True event indicators
        time_points: Time points at which survival probabilities are evaluated
        survival_probs: Predicted survival probabilities (n_samples, n_times)
    
    Returns:
        brier_scores: Brier score at each time point
        integrated_brier: Integrated Brier score
    """
    n_samples, n_times = survival_probs.shape
    brier_scores = np.zeros(n_times)
    
    for t_idx, t in enumerate(time_points):
        actual_survival = (times_true > t).astype(float)
        known_mask = (times_true > t) | ((events_true == 1) & (times_true <= t))
        
        if np.sum(known_mask) == 0:
            brier_scores[t_idx] = np.nan
            continue
        
        predicted = survival_probs[:, t_idx]
        actual = actual_survival
        
        brier_scores[t_idx] = np.mean((predicted[known_mask] - actual[known_mask]) ** 2)
    
    valid_mask = ~np.isnan(brier_scores)
    if np.sum(valid_mask) > 1:
        integrated_brier = np.trapz(brier_scores[valid_mask], time_points[valid_mask])
        integrated_brier /= (time_points[valid_mask].max() - time_points[valid_mask].min())
    else:
        integrated_brier = np.nan
    
    return brier_scores, integrated_brier


def compute_negative_log_likelihood(times_true: np.ndarray, events_true: np.ndarray,
                                   time_points: np.ndarray,
                                   survival_probs: np.ndarray) -> float:
    """
    Compute negative log-likelihood of survival predictions
    
    Args:
        times_true: True survival times
        events_true: True event indicators
        time_points: Time points for survival probabilities
        survival_probs: Predicted survival probabilities
    
    Returns:
        nll: Negative log-likelihood
    """
    n_samples = len(times_true)
    nll = 0.0
    
    for i in range(n_samples):
        time_idx = np.argmin(np.abs(time_points - times_true[i]))
        
        if time_idx >= len(time_points):
            time_idx = len(time_points) - 1
        
        if events_true[i] == 1:
            if time_idx > 0:
                s_t = survival_probs[i, time_idx]
                s_t_prev = survival_probs[i, time_idx - 1]
                h_t = (s_t_prev - s_t) / (s_t_prev + 1e-10)
                nll -= np.log(h_t + 1e-10)
            else:
                nll -= np.log(1 - survival_probs[i, 0] + 1e-10)
        else:
            nll -= np.log(survival_probs[i, time_idx] + 1e-10)
    
    return nll / n_samples


def compute_time_dependent_auc(times_true: np.ndarray, events_true: np.ndarray,
                               time_points: np.ndarray,
                               risk_scores: np.ndarray,
                               time_horizon: float) -> float:
    """
    Compute time-dependent AUC at a specific time horizon
    
    Args:
        times_true: True survival times
        events_true: True event indicators
        time_points: Time points
        risk_scores: Risk scores; should be extracted at time_horizon
        time_horizon: Time point at which to evaluate AUC
    
    Returns:
        auc: Time-dependent AUC at specified horizon
    """
    
    y_binary = ((times_true <= time_horizon) & (events_true == 1)).astype(int)
    
    valid_mask = (times_true > time_horizon) | ((events_true == 1) & (times_true <= time_horizon))
    
    n_positives = np.sum(y_binary[valid_mask])
    n_negatives = np.sum((1 - y_binary)[valid_mask])
    
    if n_positives == 0 or n_negatives == 0:
        return 0.5
    
    try:
        auc = roc_auc_score(y_binary[valid_mask], risk_scores[valid_mask])
        if auc < 0.5:
            auc = 1.0 - auc
        return auc
    except ValueError:
        return 0.5


def compute_calibration_error(times_true: np.ndarray, events_true: np.ndarray,
                              time_points: np.ndarray,
                              survival_probs: np.ndarray,
                              n_bins: int = 10) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute calibration error (MAE between predicted and observed)
    
    Args:
        times_true: True survival times
        events_true: True event indicators
        time_points: Time points
        survival_probs: Predicted survival probabilities
        n_bins: Number of bins for calibration
    
    Returns:
        calibration_error: Mean absolute calibration error
        pred_bins: Mean predicted probability per bin
        obs_bins: Observed survival rate per bin
    """
    calibration_errors = []
    pred_bins_list = []
    obs_bins_list = []
    
    eval_times = np.linspace(time_points[0], time_points[-1], min(5, len(time_points)))
    
    for eval_time in eval_times:
        time_idx = np.argmin(np.abs(time_points - eval_time))
        predicted = survival_probs[:, time_idx]
        actual = (times_true > eval_time).astype(float)
        
        known_mask = (times_true > eval_time) | ((events_true == 1) & (times_true <= eval_time))
        if np.sum(known_mask) < n_bins:
            continue
        
        pred_known = predicted[known_mask]
        actual_known = actual[known_mask]
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(pred_known, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        pred_bins = np.zeros(n_bins)
        obs_bins = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        
        for i in range(len(pred_known)):
            bin_idx = bin_indices[i]
            pred_bins[bin_idx] += pred_known[i]
            obs_bins[bin_idx] += actual_known[i]
            counts[bin_idx] += 1
        
        valid_bins = counts > 0
        pred_bins[valid_bins] /= counts[valid_bins]
        obs_bins[valid_bins] /= counts[valid_bins]
        
        cal_error = np.mean(np.abs(pred_bins[valid_bins] - obs_bins[valid_bins]))
        calibration_errors.append(cal_error)
        pred_bins_list.append(pred_bins)
        obs_bins_list.append(obs_bins)
    
    if len(calibration_errors) == 0:
        return np.nan, np.array([]), np.array([])
    
    mean_cal_error = np.mean(calibration_errors)
    mean_pred_bins = np.mean(pred_bins_list, axis=0)
    mean_obs_bins = np.mean(obs_bins_list, axis=0)
    
    return mean_cal_error, mean_pred_bins, mean_obs_bins


def extract_risk_scores_from_survival(survival_probs: np.ndarray, 
                                      time_points: np.ndarray,
                                      time_horizon: Optional[float] = None) -> np.ndarray:
    """
    Extract risk scores from survival probabilities.
    
    Args:
        survival_probs: Survival probabilities (n_samples, n_times)
        time_points: Time points
        time_horizon: Optional time horizon (uses median time if None)
    
    Returns:
        risk_scores: Risk scores (higher = higher risk)
    """
    if time_horizon is None:
        time_horizon = np.median(time_points)
    
    time_idx = np.argmin(np.abs(time_points - time_horizon))
    risk_scores = 1.0 - survival_probs[:, time_idx]
    
    return risk_scores

