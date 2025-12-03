"""
Bayesian GP + Neural Cox survival model
"""

import copy
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal, kl_divergence
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

LOG_2PI = math.log(2.0 * math.pi)


class NeuralCoxMainEffects(nn.Module):
    """
    Neural network main effects component for survival hazard
    """
    def __init__(self, input_dim: int, hidden_dims: Optional[list] = None, 
                 dropout: float = 0.2):
        """
        Args:
            input_dim: Input feature dimension (latent space size)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super(NeuralCoxMainEffects, self).__init__()
        
        self.input_dim = input_dim
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        
        self.hidden_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for i, h_dim in enumerate(hidden_dims):
            layer = nn.Sequential(
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.hidden_layers.append(layer)
            prev_dim = h_dim
        
        self.output_layer = nn.Linear(prev_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Kaiming normal"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict log-hazard from features
        
        Args:
            x: Input features (batch_size, input_dim)
        
        Returns:
            log_hazard: (batch_size,) log-hazard values
        """
        h = x
        for layer in self.hidden_layers:
            h = layer(h)
        
        return self.output_layer(h).squeeze(-1)


class VariationalGPSurvivalModel:
    """
    VI GP survival model with ARD kernel w/ inducing points
    """
    
    def __init__(self, n_pieces: int = 10, ard: bool = True, n_inducing: int = 50,
                 vi_family: str = 'lowrank', device: str = None,
                 kernel_type: str = 'matern32',
                 component_scale: float = 1.0):
        """
        Args:
            n_pieces: Number of pieces for piecewise constant baseline hazard
            ard: Whether to use ARD kernel
            n_inducing: Number of inducing points for sparse GP
            vi_family: Variational family for q(u):
                - 'lowrank': Low-rank + diagonal 
                - 'cholesky': Full Cholesky factor
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.n_pieces = n_pieces
        self.ard = ard
        self.n_inducing = n_inducing
        self.vi_family = vi_family
        self.kernel_type = kernel_type
        self.component_scale = component_scale
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Variational family: {vi_family}")
        print(f"Kernel: {kernel_type}")
        print(f"Neural Cox Main Effects: Enabled")
        
        self.latent_dim = None
        self.times = None
        self.events = None
        self.Z_train = None
        self.breakpoints = None
        self.pieces = None
        self.time_mean = None
        self.time_std = None
        
        self.neural_cox_main_effects = None
        
        self.variational_params = {}
        self.model = None
    
    def _create_baseline_hazard_pieces(self, times: np.ndarray) -> np.ndarray:
        """Create time intervals for piecewise constant baseline hazard"""
        event_times = times[times > 0]
        if len(event_times) == 0:
            event_times = times
        
        quantiles = np.linspace(0, 1, self.n_pieces + 1)
        breakpoints = np.quantile(event_times, quantiles)
        breakpoints = np.unique(breakpoints)
        
        return breakpoints
    
    def _assign_to_pieces(self, times: np.ndarray, breakpoints: np.ndarray) -> np.ndarray:
        """Assign each observation to a piecewise interval"""
        pieces = np.searchsorted(breakpoints, times, side='right') - 1
        pieces = np.clip(pieces, 0, len(breakpoints) - 2)
        return pieces
    
    def _matern32_kernel(self, X1: torch.Tensor, X2: torch.Tensor, 
                        lengthscales: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        """Matern 3/2 kernel with ARD"""
        X1_scaled = X1 / lengthscales
        X2_scaled = X2 / lengthscales
        
        X1_norm = torch.sum(X1_scaled ** 2, dim=1, keepdim=True)
        X2_norm = torch.sum(X2_scaled ** 2, dim=1)
        X1X2 = torch.matmul(X1_scaled, X2_scaled.T)
        
        sqdist = X1_norm + X2_norm - 2 * X1X2
        dist = torch.sqrt(torch.clamp(sqdist, min=1e-12))
        sqrt3 = np.sqrt(3.0)
        K = variance * (1 + sqrt3 * dist) * torch.exp(-sqrt3 * dist)
        
        return K
    
    def _rbf_time_kernel(self, t1: torch.Tensor, t2: torch.Tensor, 
                         time_lengthscale: torch.Tensor) -> torch.Tensor:
        """RBF kernel for time dimension in separable kernel"""
        t1_expanded = t1.unsqueeze(1)
        t2_expanded = t2.unsqueeze(0)
        time_diff = (t1_expanded - t2_expanded) / time_lengthscale
        K_t = torch.exp(-0.5 * time_diff ** 2)
        return K_t
    
    def _compute_kernel_separable(self, X1: torch.Tensor, X2: torch.Tensor,
                                   t1: torch.Tensor, t2: torch.Tensor,
                                   lengthscales: torch.Tensor, time_lengthscale: torch.Tensor,
                                   variance: torch.Tensor) -> torch.Tensor:
        """Compute separable kernel: k((Z_i, t_i), (Z_j, t_j)) = k_Z(Z_i, Z_j) * k_t(t_i, t_j)"""
        k_Z = self._matern32_kernel(X1, X2, lengthscales, 1.0)
        
        k_t = self._rbf_time_kernel(t1, t2, time_lengthscale)
        
        K = variance * k_Z * k_t
        return K
    
    def _compute_log_likelihood(self, log_hazard: torch.Tensor, times: torch.Tensor, 
                               events: torch.Tensor) -> torch.Tensor:
        """
        Compute survival likelihood
        
        For piecewise constant hazard model, approximate cumulative hazard as Λ(t) ≈ λ(t) * t
        """
        hazard = torch.exp(log_hazard)
        
        
        cumulative_hazard = hazard * times
        
        log_likelihood = events * log_hazard - cumulative_hazard
        
        log_likelihood = torch.clamp(log_likelihood, min=-100, max=100)
        
        return torch.sum(log_likelihood)
    
    def _compute_piecewise_cumulative_hazard(self, log_hazard: torch.Tensor, times: torch.Tensor,
                                            events: torch.Tensor, pieces: torch.Tensor,
                                            breakpoints: np.ndarray) -> torch.Tensor:
        """
        Compute cumulative hazard using proper piecewise integration
        
        For piecewise constant hazard with breakpoints t_0, t_1, ..., t_K:
            if patient's time t falls in interval [t_k, t_{k+1}):
                Λ(t) = Σ_{j=0}^{k-1} λ_j * (t_{j+1} - t_j) + λ_k * (t - t_k)
        """
        cumulative_hazard = torch.zeros_like(times)
        
        for i in range(len(times)):
            t = times[i].item()
            piece_idx = pieces[i].item()
            hazard_i = torch.exp(log_hazard[i])
            
            cumulative = 0.0
            for j in range(piece_idx):
                if j < len(breakpoints) - 1:
                    interval_length = breakpoints[j + 1] - breakpoints[j]
                    cumulative += hazard_i.item() * interval_length
            
            if piece_idx < len(breakpoints) - 1:
                interval_start = breakpoints[piece_idx]
                interval_end = min(breakpoints[piece_idx + 1], t)
                partial_length = max(0, interval_end - interval_start)
                cumulative += hazard_i.item() * partial_length
            
            cumulative_hazard[i] = cumulative
        
        return cumulative_hazard
    
    def _compute_ranking_loss(self, predicted_risk: torch.Tensor, times: torch.Tensor,
                             events: torch.Tensor) -> torch.Tensor:
        """
        Ranking loss: penalize when higher risk patients have longer survival times
        
        For each pair (i, j) where:
            Patient i has event (events[i] == 1)
            times[i] < times[j] (i dies before j)
        
        Want: risk_i > risk_j
        Penalty: log(1 + exp(risk_j - risk_i)) if risk_j > risk_i
        """
        event_mask = events == 1
        if event_mask.sum() < 2:
            return torch.tensor(0.0, device=predicted_risk.device)
        
        event_indices = torch.where(event_mask)[0]
        
        loss = 0.0
        n_pairs = 0
        
        for i_idx, i in enumerate(event_indices):
            t_i = times[i].item()
            risk_i = predicted_risk[i]
            
            mask = times > t_i
            if mask.sum() == 0:
                continue
            
            risk_j = predicted_risk[mask]
            risk_diff = risk_j - risk_i
            
            wrong_orderings = risk_diff > 0
            if wrong_orderings.sum() > 0:
                loss += torch.log(1 + torch.exp(risk_diff[wrong_orderings])).sum()
                n_pairs += wrong_orderings.sum().item()
        
        if n_pairs > 0:
            return loss / n_pairs
        else:
            return torch.tensor(0.0, device=predicted_risk.device)
    
    def fit(self, Z: np.ndarray, times: np.ndarray, events: np.ndarray,
            epochs: int = 1000, lr: float = 0.01, scheduler_start_epoch: int = 0):
        """
        Fit the model using variational inference
        
        Args:
            Z: Latent omics features (n_samples, n_latent_dim)
            times: Survival/censoring times
            events: Event indicators (1 = event, 0 = censored)
            epochs: Number of training epochs
            lr: Learning rate
            scheduler_start_epoch: Epoch to start learning rate scheduler (0 = start immediately, 2000 = start after epoch 2000)
        """
        print(f"Fitting variational GP survival model on {self.device}...")
        
        Z_tensor = torch.FloatTensor(Z).to(self.device)
        times_tensor = torch.FloatTensor(times).to(self.device)
        events_tensor = torch.FloatTensor(events).to(self.device)
        
        time_mean = times_tensor.mean().item()
        time_std = times_tensor.std().item() + 1e-8
        times_normalized = (times_tensor - time_mean) / time_std
        self.time_mean = time_mean
        self.time_std = time_std
        
        self.Z_train = Z_tensor
        self.times = times_tensor
        self.times_normalized = times_normalized
        self.events = events_tensor
        self.latent_dim = Z.shape[1]
        n_samples = len(Z)
        
        breakpoints = self._create_baseline_hazard_pieces(times)
        pieces = self._assign_to_pieces(times, breakpoints)
        self.breakpoints = breakpoints
        self.pieces = torch.LongTensor(pieces).to(self.device)
        n_intervals = len(breakpoints) - 1
        
        print(f"Using {n_intervals} piecewise constant intervals for baseline hazard")
        print(f"Latent dimension (input): {self.latent_dim}")
        print(f"Inducing points: {self.n_inducing}")
        
        Z_tensor_transformed = Z_tensor
        neural_input_dim = self.latent_dim
        self.neural_cox_main_effects = NeuralCoxMainEffects(
            input_dim=neural_input_dim,
            hidden_dims=[512, 256, 128, 64],
            dropout=0.2,
        ).to(self.device)
        print(f"Neural Cox Main Effects: {neural_input_dim} -> 1 (4 hidden layers: 512, 256, 128, 64)")
        
        if self.n_inducing < n_samples:
            indices = np.random.choice(n_samples, self.n_inducing, replace=False)
            Z_inducing = Z_tensor_transformed[indices].clone().detach().requires_grad_(False)
        else:
            Z_inducing = Z_tensor_transformed.clone().detach().requires_grad_(False)
        
        
        baseline_init = []
        for j in range(n_intervals):
            interval_start = breakpoints[j]
            interval_end = breakpoints[j + 1]
            mask = (times >= interval_start) & (times < interval_end) & (events == 1)
            n_events = mask.sum()
            n_at_risk = ((times >= interval_start) & (events == 0) | mask).sum()
            if n_at_risk > 0:
                hazard_est = n_events / (n_at_risk * (interval_end - interval_start + 1e-6))
                baseline_init.append(np.log(hazard_est + 1e-6))
            else:
                baseline_init.append(-3.0)
        baseline_init = np.array(baseline_init)
        
        mu_baseline = nn.Parameter(torch.FloatTensor(baseline_init).to(self.device))
        log_sigma_baseline = nn.Parameter(torch.ones(n_intervals, device=self.device) * (-2.0))
        
        linear_input_dim = self.latent_dim
        try:
            event_mask = events == 1
            if event_mask.sum() > 5:
                event_mean = Z[event_mask].mean(axis=0)
                no_event_mean = Z[~event_mask].mean(axis=0)
                simple_coef = (event_mean - no_event_mean) * 0.1
            else:
                simple_coef = np.random.randn(linear_input_dim) * 0.1
        except:
            simple_coef = np.zeros(linear_input_dim)
        mu_beta = nn.Parameter(torch.FloatTensor(simple_coef).to(self.device))
        log_sigma_beta = nn.Parameter(torch.ones(linear_input_dim, device=self.device) * (-2.0))
        
        mu_gp_var = nn.Parameter(torch.tensor(0.693, device=self.device))
        log_sigma_gp_var = nn.Parameter(torch.tensor(-1.5, device=self.device))
        
        if self.ard:
            mu_lengthscales = nn.Parameter(torch.zeros(self.latent_dim, device=self.device))
            log_sigma_lengthscales = nn.Parameter(torch.ones(self.latent_dim, device=self.device) * (-2.0))
        else:
            mu_lengthscales = nn.Parameter(torch.tensor(0.0, device=self.device))
            log_sigma_lengthscales = nn.Parameter(torch.tensor(-2.0, device=self.device))
        
        mu_time_lengthscale = nn.Parameter(torch.tensor(0.0, device=self.device))
        log_sigma_time_lengthscale = nn.Parameter(torch.tensor(-2.0, device=self.device))
        
        
        try:
            event_mask = events == 1
            if event_mask.sum() > 5 and (~event_mask).sum() > 5:
                Z_inducing_np = Z_inducing.detach().cpu().numpy()
                Z_np = Z
                mu_u_init = torch.zeros(self.n_inducing, device=self.device)
                
                for i in range(self.n_inducing):
                    inducing_pt = Z_inducing_np[i]
                    distances = np.linalg.norm(Z_np - inducing_pt, axis=1)
                    k = min(20, len(Z_np) // 4)
                    nearest_idx = np.argpartition(distances, k)[:k]
                    nearest_distances = distances[nearest_idx]
                    
                    weights = 1.0 / (nearest_distances + 1e-6)
                    event_weights = weights * events[nearest_idx]
                    total_weights = weights.sum()
                    
                    if total_weights > 0:
                        event_rate = event_weights.sum() / total_weights
                        mu_u_init[i] = (event_rate - 0.5) * 2.0
                    else:
                        mu_u_init[i] = 0.0
            else:
                mu_u_init = torch.randn(self.n_inducing, device=self.device) * 0.1
        except:
            mu_u_init = torch.randn(self.n_inducing, device=self.device) * 0.1
        
        mu_u = nn.Parameter(mu_u_init)
        
        L_u_tril = None
        L_u_mask = None
        L_u_raw = None
        log_diag_u = None
        log_sigma_u = None
        flow_param_list = []
        
        if self.vi_family == 'cholesky':
            L_u_tril = nn.Parameter(torch.zeros(self.n_inducing, self.n_inducing, device=self.device))
            with torch.no_grad():
                L_u_tril.data = torch.eye(self.n_inducing, device=self.device) * 0.1
            L_u_mask = torch.tril(torch.ones(self.n_inducing, self.n_inducing, device=self.device))
        elif self.vi_family == 'lowrank':
            rank_u = min(10, self.n_inducing // 2)
            L_u_raw = nn.Parameter(torch.randn(self.n_inducing, rank_u, device=self.device) * 0.1)
            log_diag_u = nn.Parameter(torch.ones(self.n_inducing, device=self.device) * (-4.0))
        else:
            raise ValueError(f"Unknown vi_family: {self.vi_family}. Must be one of: 'lowrank', 'cholesky'")
        
        Z_inducing_param = nn.Parameter(Z_inducing)
        
        times_normalized_np = times_normalized.detach().cpu().numpy()
        time_quantiles = np.linspace(0, 1, self.n_inducing)
        t_inducing_init = np.quantile(times_normalized_np, time_quantiles)
        t_inducing_param = nn.Parameter(torch.FloatTensor(t_inducing_init).to(self.device))
        
        params = [mu_baseline, log_sigma_baseline, mu_gp_var, log_sigma_gp_var,
                  mu_lengthscales, log_sigma_lengthscales, mu_time_lengthscale, log_sigma_time_lengthscale,
                  mu_u, Z_inducing_param, t_inducing_param]
        
        if self.vi_family == 'cholesky':
            params.extend([L_u_tril])
        elif self.vi_family == 'lowrank':
            params.extend([L_u_raw, log_diag_u])
        
        params.extend([mu_beta, log_sigma_beta])
        
        neural_params = list(self.neural_cox_main_effects.parameters())
        neural_lr = lr * 2.0
        
        params_with_lr = [
            {'params': params, 'lr': lr},
            {'params': neural_params, 'lr': neural_lr, 'weight_decay': 1e-4}
        ]
        optimizer = optim.Adam(params_with_lr)
        print(f"Using separate learning rates: GP params={lr:.6f}, Neural Cox={neural_lr:.6f} (with weight decay)")
        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=50, verbose=False)
        except TypeError:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=50)
        
        print(f"\nTraining for {epochs} epochs...")
        kl_warmup_epochs = min(500, epochs // 4)
        
        use_amp = self.device == 'cuda' and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        if use_amp:
            print("Using automatic mixed precision (AMP)")
        
        best_elbo = float('-inf')
        
        for epoch in range(epochs):
            Z_batch = Z_tensor
            times_batch = times_tensor
            events_batch = events_tensor
            pieces_batch = self.pieces
            n_batch = n_samples
            optimizer.zero_grad()
            
            eps_baseline = torch.randn(n_intervals, device=self.device)
            log_baseline = mu_baseline + torch.exp(log_sigma_baseline) * eps_baseline
            
            eps_var = torch.randn(1, device=self.device).item()
            log_gp_var = mu_gp_var + torch.exp(log_sigma_gp_var) * eps_var
            gp_variance = torch.clamp(torch.exp(log_gp_var), min=0.5, max=50.0)
            
            eps_length = torch.randn(self.latent_dim if self.ard else 1, device=self.device)
            if self.ard:
                log_lengthscales = mu_lengthscales + torch.exp(log_sigma_lengthscales) * eps_length
                lengthscales = torch.exp(log_lengthscales)
                lengthscales = torch.clamp(lengthscales, min=0.1, max=10.0)
            else:
                log_lengthscale = mu_lengthscales + torch.exp(log_sigma_lengthscales) * eps_length
                lengthscales = torch.exp(log_lengthscale).repeat(self.latent_dim)
                lengthscales = torch.clamp(lengthscales, min=0.1, max=10.0)
            
            eps_time_length = torch.randn(1, device=self.device)
            log_time_lengthscale = mu_time_lengthscale + torch.exp(log_sigma_time_lengthscale) * eps_time_length
            time_lengthscale = torch.exp(log_time_lengthscale)
            time_lengthscale = torch.clamp(time_lengthscale, min=0.1, max=10.0)
            
            log_q_u = None
            Sigma_u = None
            diag_u = None
            if self.vi_family == 'cholesky':
                L_u = L_u_tril * L_u_mask
                L_u = L_u + torch.diag(torch.exp(torch.clamp(torch.diag(L_u), min=-10, max=10))) * 1e-6
                Sigma_u = L_u @ L_u.T + torch.eye(self.n_inducing, device=self.device) * 1e-6
                eps_u = torch.randn(self.n_inducing, device=self.device)
                try:
                    L_Sigma = torch.linalg.cholesky(Sigma_u)
                    u = mu_u + L_Sigma @ eps_u
                except:
                    u = mu_u + L_u @ eps_u
            elif self.vi_family == 'lowrank':
                rank_u = L_u_raw.shape[1]
                diag_u = torch.exp(log_diag_u) + 1e-6
                Sigma_u = L_u_raw @ L_u_raw.T + torch.diag(diag_u)
                eps_u = torch.randn(self.n_inducing, device=self.device)
                try:
                    L_Sigma = torch.linalg.cholesky(Sigma_u)
                    u = mu_u + L_Sigma @ eps_u
                except:
                    u = mu_u + torch.sqrt(diag_u) * eps_u
            
            Z_batch_transformed = Z_batch
            
            times_batch_normalized = self.times_normalized
            K_mm = self._compute_kernel_separable(
                Z_inducing_param, Z_inducing_param, 
                t_inducing_param, t_inducing_param,
                lengthscales, time_lengthscale, gp_variance
            )
            K_mm = (K_mm + K_mm.T) / 2.0
            
            base_jitter = 1e-4
            K_mm = K_mm + torch.eye(self.n_inducing, device=self.device) * base_jitter
            
            L_mm = None
            K_mm_inv = None
            for attempt in range(4):
                try:
                    L_mm = torch.linalg.cholesky(K_mm)
                    K_mm_inv = torch.cholesky_inverse(L_mm)
                    break
                except:
                    if attempt < 3:
                        jitter = base_jitter * (10.0 ** attempt)
                        K_mm = K_mm + torch.eye(self.n_inducing, device=self.device) * jitter
                    else:
                        try:
                            eigenvals, eigenvecs = torch.linalg.eigh(K_mm)
                            eigenvals = torch.clamp(eigenvals, min=1e-6)
                            K_mm = eigenvecs @ torch.diag(eigenvals) @ eigenvecs.T
                            K_mm = (K_mm + K_mm.T) / 2.0
                            K_mm = K_mm + torch.eye(self.n_inducing, device=self.device) * 1e-4
                            L_mm = torch.linalg.cholesky(K_mm)
                            K_mm_inv = torch.cholesky_inverse(L_mm)
                        except:
                            K_mm = torch.diag(torch.diag(K_mm)) + torch.eye(self.n_inducing, device=self.device) * 1e-3
                            L_mm = torch.linalg.cholesky(K_mm)
                            K_mm_inv = torch.cholesky_inverse(L_mm)
                        break
            
            if L_mm is None or K_mm_inv is None:
                raise RuntimeError("Failed to compute Cholesky decomposition of K_mm after all fallbacks")
            
            K_nm = self._compute_kernel_separable(
                Z_batch_transformed, Z_inducing_param,
                times_batch_normalized, t_inducing_param,
                lengthscales, time_lengthscale, gp_variance
            )
            K_nn = self._compute_kernel_separable(
                Z_batch_transformed, Z_batch_transformed,
                times_batch_normalized, times_batch_normalized,
                lengthscales, time_lengthscale, gp_variance
            )
            
            f_mean = K_nm @ K_mm_inv @ u
            
            f_cov = K_nn - K_nm @ K_mm_inv @ K_nm.T
            f_cov = f_cov + torch.eye(n_batch, device=self.device) * 1e-4
            
            try:
                f_dist = MultivariateNormal(f_mean, covariance_matrix=f_cov)
                f = f_dist.rsample()
            except:
                f_var = torch.clamp(torch.diag(f_cov), min=1e-6)
                f = f_mean + torch.randn_like(f_mean) * torch.sqrt(f_var)
            
            log_hazard_base = log_baseline[pieces_batch]
            
            neural_hazard = self.neural_cox_main_effects(Z_batch)
            
            log_hazard_linear = torch.zeros(n_batch, device=self.device)
            
            log_hazard = log_hazard_base + neural_hazard + f
            log_hazard = torch.clamp(log_hazard, min=-8, max=2)
            
            log_likelihood = self._compute_log_likelihood(log_hazard, times_batch, events_batch)
            
            if torch.isnan(log_likelihood) or torch.isinf(log_likelihood):
                log_likelihood = torch.tensor(-1e6, device=self.device)
                print(f"Warning: NaN/Inf in likelihood at epoch {epoch+1}, skipping")
                continue
            
            kl_baseline = 0.5 * torch.sum(
                torch.exp(2 * log_sigma_baseline) + mu_baseline ** 2 - 1 - 2 * log_sigma_baseline
            ) / 4.0
            
            kl_gp_var = 0.5 * (torch.exp(2 * log_sigma_gp_var) + mu_gp_var ** 2)
            
            if self.ard:
                kl_lengthscales = 0.5 * torch.sum(
                    torch.exp(2 * log_sigma_lengthscales) + mu_lengthscales ** 2
                )
            else:
                kl_lengthscales = 0.5 * (torch.exp(2 * log_sigma_lengthscales) + mu_lengthscales ** 2)
            
            kl_time_lengthscale = 0.5 * (
                torch.exp(2 * log_sigma_time_lengthscale) + mu_time_lengthscale ** 2 - 1 - 2 * log_sigma_time_lengthscale
            )
            
            kl_kernel_weights = torch.tensor(0.0, device=self.device)
            
            kl_beta = 0.5 * torch.sum(
                torch.exp(2 * log_sigma_beta) + mu_beta ** 2
            ) / 4.0 + 0.1 * torch.sum(mu_beta ** 2)
            
            try:
                K_mm_inv = torch.cholesky_inverse(L_mm)
                if self.vi_family == 'cholesky':
                    kl_u = 0.5 * (torch.trace(K_mm_inv @ Sigma_u) + 
                                 mu_u @ K_mm_inv @ mu_u - self.n_inducing)
                    try:
                        logdet_K_mm = 2 * torch.sum(torch.log(torch.diag(L_mm)))
                        logdet_Sigma_u = torch.logdet(Sigma_u + torch.eye(self.n_inducing, device=self.device) * 1e-6)
                        kl_u = kl_u + 0.5 * (logdet_K_mm - logdet_Sigma_u)
                    except:
                        pass
                else:
                    kl_u = 0.5 * (torch.trace(K_mm_inv @ Sigma_u) + 
                                 mu_u @ K_mm_inv @ mu_u - self.n_inducing)
                    try:
                        logdet_K_mm = 2 * torch.sum(torch.log(torch.diag(L_mm)))
                        logdet_Sigma_u = torch.logdet(Sigma_u + torch.eye(self.n_inducing, device=self.device) * 1e-6)
                        kl_u = kl_u + 0.5 * (logdet_K_mm - logdet_Sigma_u)
                    except:
                        pass
            except:
                if self.vi_family == 'cholesky':
                    kl_u = 0.5 * (mu_u @ mu_u + torch.trace(Sigma_u))
                else:
                    kl_u = 0.5 * (mu_u @ mu_u + torch.sum(diag_u))
            
            if epoch < kl_warmup_epochs:
                kl_weight = epoch / kl_warmup_epochs
            else:
                kl_weight = 1.0
            
            total_kl = kl_baseline + kl_gp_var + kl_lengthscales + kl_time_lengthscale + kl_beta + kl_u + kl_kernel_weights
            elbo = log_likelihood - kl_weight * total_kl
            
            loss = -elbo
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
                optimizer.step()
            
            if epoch >= scheduler_start_epoch:
                scheduler.step(elbo.item())
            
            if (epoch + 1) % 100 == 0:
                total_kl_val = kl_baseline.item() + kl_gp_var.item() + kl_lengthscales.item() + kl_time_lengthscale.item() + kl_beta.item() + kl_u.item()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1}/{epochs}: ELBO = {elbo.item():.2f}, "
                     f"LL = {log_likelihood.item():.2f}, KL = {total_kl_val:.2f}, "
                     f"KL_weight = {kl_weight:.3f}, LR = {current_lr:.6f}")
            
            if elbo.item() > best_elbo:
                best_elbo = elbo.item()
                self.variational_params = {
                    'mu_baseline': mu_baseline.clone().detach(),
                    'log_sigma_baseline': log_sigma_baseline.clone().detach(),
                    'mu_gp_var': mu_gp_var.clone().detach(),
                    'log_sigma_gp_var': log_sigma_gp_var.clone().detach(),
                    'mu_lengthscales': mu_lengthscales.clone().detach(),
                    'log_sigma_lengthscales': log_sigma_lengthscales.clone().detach(),
                    'mu_time_lengthscale': mu_time_lengthscale.clone().detach(),
                    'log_sigma_time_lengthscale': log_sigma_time_lengthscale.clone().detach(),
                    'mu_u': mu_u.clone().detach(),
                    'Z_inducing': Z_inducing_param.clone().detach(),
                    't_inducing': t_inducing_param.clone().detach()
                }
                self.variational_params['mu_beta'] = mu_beta.clone().detach()
                self.variational_params['log_sigma_beta'] = log_sigma_beta.clone().detach()
                self.variational_params['neural_cox_main_effects_state_dict'] = self.neural_cox_main_effects.state_dict()
                if self.vi_family == 'cholesky':
                    self.variational_params['L_u_tril'] = L_u_tril.clone().detach()
                elif self.vi_family == 'lowrank':
                    self.variational_params['L_u_raw'] = L_u_raw.clone().detach()
                    self.variational_params['log_diag_u'] = log_diag_u.clone().detach()
        
        print(f"\nTraining complete! Best ELBO: {best_elbo:.2f}")
        self.model = {}
        if len(self.variational_params) > 0:
            for key in self.variational_params:
                if key in ['mu_baseline', 'log_sigma_baseline', 'mu_gp_var', 'log_sigma_gp_var',
                          'mu_lengthscales', 'log_sigma_lengthscales', 'mu_time_lengthscale', 'log_sigma_time_lengthscale',
                          'mu_u', 'Z_inducing', 't_inducing',
                          'mu_beta', 'log_sigma_beta', 'L_u_tril', 'L_u_raw', 'log_diag_u']:
                    param_val = self.variational_params[key]
                    if isinstance(param_val, torch.Tensor):
                        self.model[key] = param_val.clone().to(self.device)
                    else:
                        self.model[key] = param_val
        else:
            self.model = {'mu_baseline': mu_baseline, 'log_sigma_baseline': log_sigma_baseline,
                         'mu_gp_var': mu_gp_var, 'log_sigma_gp_var': log_sigma_gp_var,
                         'mu_lengthscales': mu_lengthscales, 'log_sigma_lengthscales': log_sigma_lengthscales,
                         'mu_u': mu_u, 'Z_inducing': Z_inducing_param}
            if self.vi_family == 'cholesky':
                self.model['L_u_tril'] = L_u_tril
            elif self.vi_family == 'lowrank':
                self.model['L_u_raw'] = L_u_raw
                self.model['log_diag_u'] = log_diag_u
            self.model['mu_beta'] = mu_beta
            self.model['log_sigma_beta'] = log_sigma_beta
        
        return self
    
    def predict_survival(self, Z_new: np.ndarray, times: np.ndarray,
                        n_samples: int = 100) -> np.ndarray:
        """Predict survival probabilities using variational posterior"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        print(f"Predicting survival for {Z_new.shape[0]} new samples on {self.device}...")
        
        Z_new_tensor = torch.FloatTensor(Z_new).to(self.device)
        
        times_array = np.array(times)
        pieces_new = self._assign_to_pieces(times_array, self.breakpoints)
        pieces_new_tensor = torch.LongTensor(pieces_new).to(self.device)
        
        n_new = Z_new_tensor.shape[0]
        n_times = len(times)
        survival_probs = np.zeros((n_new, n_times, n_samples))
        
        mu_baseline = self.model['mu_baseline']
        mu_gp_var = self.model['mu_gp_var']
        mu_lengthscales = self.model['mu_lengthscales']
        mu_time_lengthscale = self.model['mu_time_lengthscale']
        mu_u = self.model['mu_u']
        Z_inducing = self.model['Z_inducing']
        t_inducing = self.model['t_inducing']
        
        times_tensor = torch.FloatTensor(times_array).to(self.device)
        times_normalized = (times_tensor - self.time_mean) / self.time_std
        
        if self.vi_family == 'cholesky':
            L_u_tril = self.model['L_u_tril']
            L_u_mask = torch.tril(torch.ones(self.n_inducing, self.n_inducing, device=self.device))
        elif self.vi_family == 'lowrank':
            L_u_raw = self.model['L_u_raw']
            log_diag_u = self.model['log_diag_u']
        else:
            raise ValueError(f"Unknown vi_family: {self.vi_family}. Must be one of: 'lowrank', 'cholesky'")
        
        for i in range(n_samples):
            eps_baseline = torch.randn(len(mu_baseline), device=self.device)
            log_baseline = mu_baseline + torch.exp(self.model['log_sigma_baseline']) * eps_baseline
            
            eps_var = torch.randn(1, device=self.device).item()
            log_gp_var = mu_gp_var + torch.exp(self.model['log_sigma_gp_var']) * eps_var
            gp_variance = torch.clamp(torch.exp(log_gp_var), min=0.5, max=50.0)
            
            eps_length = torch.randn(self.latent_dim, device=self.device)
            log_lengthscales = mu_lengthscales + torch.exp(self.model['log_sigma_lengthscales']) * eps_length
            lengthscales = torch.clamp(torch.exp(log_lengthscales), min=0.1, max=10.0)
            
            eps_time_length = torch.randn(1, device=self.device)
            log_time_lengthscale = mu_time_lengthscale + torch.exp(self.model['log_sigma_time_lengthscale']) * eps_time_length
            time_lengthscale = torch.clamp(torch.exp(log_time_lengthscale), min=0.1, max=10.0)
            
            if self.vi_family == 'cholesky':
                L_u = L_u_tril * L_u_mask
                L_u = L_u + torch.diag(torch.exp(torch.clamp(torch.diag(L_u), min=-10, max=10))) * 1e-6
                Sigma_u = L_u @ L_u.T + torch.eye(self.n_inducing, device=self.device) * 1e-6
                try:
                    L_Sigma = torch.linalg.cholesky(Sigma_u)
                    eps_u = torch.randn(self.n_inducing, device=self.device)
                    u = mu_u + L_Sigma @ eps_u
                except:
                    u = mu_u + L_u @ torch.randn(self.n_inducing, device=self.device)
            elif self.vi_family == 'lowrank':
                diag_u = torch.exp(log_diag_u) + 1e-6
                rank_u = L_u_raw.shape[1]
                Sigma_u = L_u_raw @ L_u_raw.T + torch.diag(diag_u)
                try:
                    L_Sigma = torch.linalg.cholesky(Sigma_u)
                    eps_u = torch.randn(self.n_inducing, device=self.device)
                    u = mu_u + L_Sigma @ eps_u
                except:
                    u = mu_u + torch.sqrt(diag_u) * torch.randn(self.n_inducing, device=self.device)
            else:
                raise ValueError(f"Unknown vi_family: {self.vi_family}. Must be one of: 'lowrank', 'cholesky'")
            
            K_mm = self._compute_kernel_separable(
                Z_inducing, Z_inducing,
                t_inducing, t_inducing,
                lengthscales, time_lengthscale, gp_variance
            )
            K_mm = (K_mm + K_mm.T) / 2.0
            base_jitter = 1e-4
            K_mm = K_mm + torch.eye(self.n_inducing, device=self.device) * base_jitter
            
            try:
                L_mm = torch.linalg.cholesky(K_mm)
                K_mm_inv = torch.cholesky_inverse(L_mm)
            except:
                eigenvals, eigenvecs = torch.linalg.eigh(K_mm)
                eigenvals = torch.clamp(eigenvals, min=1e-6)
                K_mm = eigenvecs @ torch.diag(eigenvals) @ eigenvecs.T
                K_mm = (K_mm + K_mm.T) / 2.0
                K_mm = K_mm + torch.eye(self.n_inducing, device=self.device) * 1e-3
                L_mm = torch.linalg.cholesky(K_mm)
                K_mm_inv = torch.cholesky_inverse(L_mm)
            
            f_new_mean = torch.zeros(n_new, n_times, device=self.device)
            
            for t_idx in range(n_times):
                t_norm_value = times_normalized[t_idx].item()
                t_norm_expanded = torch.full((n_new,), t_norm_value, device=self.device, dtype=torch.float32)
                K_new_m_t = self._compute_kernel_separable(
                    Z_new_tensor, Z_inducing,
                    t_norm_expanded, t_inducing,
                    lengthscales, time_lengthscale, gp_variance
                )
                
                f_new_mean[:, t_idx] = K_new_m_t @ K_mm_inv @ u
            
            f_new = f_new_mean
            
            log_hazard_base_per_time = log_baseline[pieces_new_tensor]
            
            with torch.no_grad():
                neural_hazard = self.neural_cox_main_effects(Z_new_tensor)
            neural_hazard_expanded = neural_hazard.unsqueeze(1).expand(-1, n_times)
            
            log_hazard_linear_expanded = torch.zeros(n_new, n_times, device=self.device)
            
            log_hazard = (log_hazard_base_per_time.unsqueeze(0) + 
                        neural_hazard_expanded + 
                        f_new)
            
            log_hazard = torch.clamp(log_hazard, min=-8, max=2)
            hazard = torch.exp(log_hazard)
            
            time_tensor = torch.FloatTensor(times_array).to(self.device)
            time_diffs = torch.diff(torch.cat([torch.tensor([0.0], device=self.device), time_tensor]))
            
            cumulative_hazard = torch.cumsum(hazard * time_diffs.unsqueeze(0), dim=1)
            survival = torch.exp(-cumulative_hazard)
            
            survival_probs[:, :, i] = survival.detach().cpu().numpy()
        
        return survival_probs
    
    def get_feature_relevance(self) -> np.ndarray:
        """Get ARD lengthscales (inverse relevance) from variational posterior"""
        if self.model is None or not self.ard:
            raise ValueError("Model must be trained with ARD=True first")
        
        mean_lengthscales = torch.exp(self.model['mu_lengthscales']).detach().cpu().numpy()
        relevance = 1.0 / (mean_lengthscales + 1e-8)
        
        return relevance
    
    def get_posterior_summary(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        summaries = []
        
        mu_b = self.model['mu_baseline'].detach().cpu().numpy()
        sigma_b = torch.exp(self.model['log_sigma_baseline']).detach().cpu().numpy()
        for i, (m, s) in enumerate(zip(mu_b, sigma_b)):
            summaries.append({
                'parameter': f'log_baseline_hazard[{i}]',
                'mean': m,
                'std': s,
                'q2.5': m - 1.96 * s,
                'q97.5': m + 1.96 * s
            })
        
        mu_v = self.model['mu_gp_var'].detach().cpu().item()
        sigma_v = torch.exp(self.model['log_sigma_gp_var']).detach().cpu().item()
        summaries.append({
            'parameter': 'gp_variance',
            'mean': np.exp(mu_v),
            'std': np.exp(mu_v) * sigma_v,
            'q2.5': np.exp(mu_v - 1.96 * sigma_v),
            'q97.5': np.exp(mu_v + 1.96 * sigma_v)
        })
        
        mu_l = self.model['mu_lengthscales'].detach().cpu().numpy()
        sigma_l = torch.exp(self.model['log_sigma_lengthscales']).detach().cpu().numpy()
        for i, (m, s) in enumerate(zip(mu_l, sigma_l)):
            summaries.append({
                'parameter': f'lengthscales[{i}]',
                'mean': np.exp(m),
                'std': np.exp(m) * s,
                'q2.5': np.exp(m - 1.96 * s),
                'q97.5': np.exp(m + 1.96 * s)
            })
        
        return pd.DataFrame(summaries)

