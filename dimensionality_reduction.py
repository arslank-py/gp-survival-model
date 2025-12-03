"""
Dimensionality reduction for multiomics data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import warnings


class VAE(nn.Module):
    """Variational autoencoder for multiomics dimensionality reduction"""
    
    def __init__(self, input_dim: int, latent_dim: int = 50, hidden_dims: list = None):
        super(VAE, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]  
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        decoder_layers = []
        decoder_dims = hidden_dims[::-1]
        prev_dim = latent_dim
        for h_dim in decoder_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent representation"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class VAEReducer:
    """VAE reduction wrapper"""
    
    def __init__(self, n_components: int = 50, hidden_dims: list = None, 
                 epochs: int = 100, batch_size: int = 256, device: str = None,
                 beta: float = 0.1, random_seed: int = 42, early_stopping: bool = True,
                 survival_weight: float = 0.1):
        self.n_components = n_components
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.beta = beta
        self.survival_weight = survival_weight
        self.random_seed = random_seed
        self.early_stopping = early_stopping
        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted = False
        self.survival_head = None
    
    def _loss_function(self, recon_x, x, mu, logvar, beta: float = 0.1,
                      survival_loss: torch.Tensor = None):
        """VAE loss: reconstruction + KL divergence + survival loss"""
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kl_loss
        
        if survival_loss is not None:
            total_loss = total_loss + self.survival_weight * survival_loss
        
        return total_loss
    
    def _cox_partial_log_likelihood(self, z: torch.Tensor, times: torch.Tensor, 
                                    events: torch.Tensor, survival_head: nn.Module) -> torch.Tensor:
        """
        Compute Cox partial log-likelihood from latent representation
        
        Args:
            z: Latent representations (batch_size, latent_dim)
            times: Survival times (batch_size,)
            events: Event indicators (batch_size,)
            survival_head: Linear layer to predict risk scores from z
        
        Returns:
            Negative partial log-likelihood (normalized by batch size)
        """
        risk_scores = survival_head(z).squeeze()
        
        sorted_indices = torch.argsort(times)
        sorted_times = times[sorted_indices]
        sorted_events = events[sorted_indices]
        sorted_risk = risk_scores[sorted_indices]
        
        event_mask = sorted_events == 1
        
        if event_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device, requires_grad=True)
        
        event_risk = sorted_risk[event_mask]
        event_indices = torch.where(event_mask)[0]
        
        n = len(sorted_times)
        event_risk_sum = torch.zeros_like(event_risk)
        
        for i, event_idx in enumerate(event_indices):
            event_time = sorted_times[event_idx]
            at_risk_mask = sorted_times >= event_time
            risk_sum = torch.logsumexp(sorted_risk[at_risk_mask], dim=0)
            event_risk_sum[i] = risk_sum
        
        cox_loss = -torch.sum(event_risk - event_risk_sum) / max(event_mask.sum().item(), 1)
        
        return cox_loss
    
    def fit_transform(self, X: np.ndarray, times: np.ndarray, events: np.ndarray) -> np.ndarray:
        """
        Train supervised VAE and transform data to latent space
        
        Args:
            X: Feature matrix (n_samples, n_features)
            times: Survival times (n_samples,)
            events: Event indicators (n_samples,)
        """
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model = VAE(
            input_dim=X.shape[1],
            latent_dim=self.n_components,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        if times is None or events is None:
            raise ValueError("times and events must be provided")
        self.survival_head = nn.Linear(self.n_components, 1).to(self.device)
        times = np.asarray(times)
        events = np.asarray(events)
        times_tensor = torch.as_tensor(times, dtype=torch.float32, device=self.device)
        events_tensor = torch.as_tensor(events, dtype=torch.float32, device=self.device)
        
        optimizer_params = list(self.model.parameters()) + list(self.survival_head.parameters())
        optimizer = optim.Adam(optimizer_params, lr=1e-3)
        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=20, verbose=False)
        except TypeError:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=20)
        self.model.train()
        
        use_amp = self.device == 'cuda' and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        if use_amp:
            print("Using automatic mixed precision (AMP)")
        
        model_compiled = False
        original_model = self.model
        try:
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model, mode='reduce-overhead')
                with torch.no_grad():
                    dummy_input = torch.randn(1, X_tensor.shape[1], device=self.device)
                    _ = self.model(dummy_input)
                print("Model compiled with torch.compile for faster training")
                model_compiled = True
        except Exception as e:
            self.model = original_model  
            if 'Triton' in str(e) or 'triton' in str(e).lower() or 'TritonMissing' in str(type(e)):
                print("Note: torch.compile requires Triton")
            pass 
        
        dataset = torch.utils.data.TensorDataset(X_tensor, times_tensor, events_tensor)
        
        generator = torch.Generator()
        generator.manual_seed(self.random_seed)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=generator,
            num_workers=0, pin_memory=False  # Data already on GPU, no need to pin
        )
        
        print(f"Training Supervised VAE for {self.epochs} epochs...")
        print(f"  Survival loss weight: {self.survival_weight}")
        best_loss = float('inf')
        patience_counter = 0
        early_stop_patience = max(50, self.epochs // 5)  # Early stop if no improvement
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]
                batch_times = batch[1]
                batch_events = batch[2]
                
                optimizer.zero_grad()
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        recon, mu, logvar = self.model(x)
                        survival_loss = self._cox_partial_log_likelihood(mu, batch_times, batch_events, self.survival_head)
                        loss = self._loss_function(
                            recon, x, mu, logvar,
                            beta=self.beta,
                            survival_loss=survival_loss
                        )
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    params = list(self.model.parameters()) + list(self.survival_head.parameters())
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    recon, mu, logvar = self.model(x)
                    survival_loss = self._cox_partial_log_likelihood(mu, batch_times, batch_events, self.survival_head)
                    loss = self._loss_function(
                        recon, x, mu, logvar,
                        beta=self.beta,
                        survival_loss=survival_loss
                    )
                    loss.backward()
                    params = list(self.model.parameters()) + list(self.survival_head.parameters())
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            if self.early_stopping:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        print(f"Early stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)")
                        break
            
            if (epoch + 1) % 20 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        self.model.eval()
        with torch.no_grad():
            mu, _ = self.model.encode(X_tensor)
            Z = mu.cpu().numpy()
        
        self.is_fitted = True
        print(f"VAE: {X.shape[1]} features -> {Z.shape[1]} latent dimensions")
        
        return Z
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted VAE"""
        if not self.is_fitted:
            raise ValueError("VAE must be fitted before transform")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            mu, _ = self.model.encode(X_tensor)
            Z = mu.cpu().numpy()
        
        return Z


def reduce_dimensionality(X: np.ndarray, method: str = 'vae', 
                         n_components: int = 50, random_seed: int = 42, 
                         times: np.ndarray = None,
                         events: np.ndarray = None, **kwargs) -> Tuple[np.ndarray, object]:
    """
    Convenience function for supervised VAE
    
    Args:
        X: Feature matrix (n_samples, n_features)
        method: 'vae' (only supported method)
        n_components: Number of latent dimensions
        times: Survival times (n_samples,)
        events: Event indicators (n_samples,)
        **kwargs: Additional arguments for reducer
    
    Returns:
        Z: Latent representations (n_samples, n_components)
        reducer: Fitted reducer object
    """
    if method.lower() != 'vae':
        raise ValueError(f"Unknown method: {method}. Only 'vae' is supported.")
    
    if times is None or events is None:
        raise ValueError("times and events must be provided")
    
    if 'random_seed' not in kwargs:
        kwargs['random_seed'] = random_seed
    
    reducer = VAEReducer(n_components=n_components, **kwargs)
    Z = reducer.fit_transform(X, times=times, events=events)
    return Z, reducer

