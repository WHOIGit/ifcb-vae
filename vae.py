import json
import sys
from typing import List 

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, random_split

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt


def create_synthetic_data(
    n_samples: int,
    n_taxa: int,
    n_env_covariates: int,
    n_latent_dims: int,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates synthetic plankton data matching your scale (0 to ~100s).
    Includes co-occurrence patterns suitable for testing attention mechanisms.
    """
    torch.manual_seed(seed)
    
    # 1. Generate correlated environmental data
    env_correlation = torch.randn(n_env_covariates, n_env_covariates) * 0.3
    env_cov = torch.mm(env_correlation, env_correlation.t())
    env_cov = env_cov + torch.eye(n_env_covariates) * 2.0
    env_mean = torch.zeros(n_env_covariates)
    mvn = torch.distributions.MultivariateNormal(env_mean, env_cov)
    env_data = mvn.sample((n_samples,))
    
    # 2. Map to latent space
    truth_mapping_net = nn.Sequential(
        nn.Linear(n_env_covariates, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, n_latent_dims),
        nn.Tanh()
    )
    
    with torch.no_grad():
        true_latent_positions = truth_mapping_net(env_data)
    
    # 3. Create strong co-occurrence patterns (important for attention)
    # Define clear ecological groups
    n_groups = min(5, n_taxa // 3)  # More distinct groups
    
    # Hard assignment to primary group (for stronger patterns)
    primary_group = torch.randint(0, n_groups, (n_taxa,))
    
    # But allow secondary associations
    group_membership = torch.zeros(n_taxa, n_groups)
    group_membership[range(n_taxa), primary_group] = 1.0
    # Add some secondary associations
    for i in range(n_taxa):
        if torch.rand(1) < 0.3:  # 30% have secondary group
            secondary = torch.randint(0, n_groups, (1,))
            if secondary != primary_group[i]:
                group_membership[i, secondary] = 0.5
    
    # Normalize
    group_membership = group_membership / group_membership.sum(dim=1, keepdim=True)
    
    # Group centers in latent space
    group_centers = (torch.rand(n_groups, n_latent_dims) * 2.0) - 1.0
    
    # Species niches based on groups
    niche_centers = torch.matmul(group_membership, group_centers)
    niche_centers += torch.randn(n_taxa, n_latent_dims) * 0.2  # Individual variation
    
    # Niche widths - some specialists, some generalists
    niche_widths = torch.exp(torch.randn(n_taxa, n_latent_dims) * 0.4) + 0.5
    
    # 4. Strong co-occurrence patterns through interactions
    interaction_matrix = torch.zeros(n_taxa, n_taxa)
    
    # Strong positive interactions within groups
    for g in range(n_groups):
        group_mask = (primary_group == g)
        group_indices = torch.where(group_mask)[0]
        for i in range(len(group_indices)):
            for j in range(i+1, len(group_indices)):
                # Positive within-group association
                interaction_matrix[group_indices[i], group_indices[j]] = 0.5
                interaction_matrix[group_indices[j], group_indices[i]] = 0.5
    
    # Negative interactions between certain group pairs (e.g., competition)
    for g1 in range(n_groups):
        for g2 in range(g1+1, n_groups):
            if torch.rand(1) < 0.4:  # 40% chance of group competition
                mask1 = (primary_group == g1)
                mask2 = (primary_group == g2)
                indices1 = torch.where(mask1)[0]
                indices2 = torch.where(mask2)[0]
                for i1 in indices1:
                    for i2 in indices2:
                        interaction_matrix[i1, i2] = -0.3
                        interaction_matrix[i2, i1] = -0.3
    
    # 5. Calculate abundances with your scale in mind
    dists = true_latent_positions.unsqueeze(1) - niche_centers.unsqueeze(0)
    scaled_dists = dists / (niche_widths.unsqueeze(0) + 1e-6)
    env_response = torch.exp(-0.5 * scaled_dists.pow(2).sum(dim=2))
    
    # 6. Apply interactions to create co-occurrence
    abundances = env_response.clone()
    
    # Stronger interaction effects for clearer patterns
    for _ in range(3):
        interaction_effects = torch.matmul(abundances, interaction_matrix)
        abundances = env_response * torch.exp(0.5 * interaction_effects)  # Exponential for stronger effect
        abundances = torch.clamp(abundances, min=0, max=1)
    
    # 7. Scale to match your data range (0 to 100s)
    # Different max concentrations for different species, but in your range
    max_concentrations = torch.exp(torch.randn(n_taxa) * 0.7 + 3.5)  # Roughly 10 to 100
    
    # Some dominant species
    dominance = torch.randn(n_taxa)
    max_concentrations[dominance > 1.5] *= 2.0  # Some species up to ~200
    max_concentrations[dominance < -1.5] *= 0.2  # Some species max ~5
    
    # Ensure we stay in reasonable range
    max_concentrations = torch.clamp(max_concentrations, min=1, max=500)
    
    base_concentrations = abundances * max_concentrations.unsqueeze(0)
    
    # 8. Add correlated noise for co-occurring species
    # Base noise
    noise = torch.randn(n_samples, n_taxa) * 0.2
    
    # Add group-correlated noise
    for g in range(n_groups):
        group_mask = (primary_group == g)
        group_noise = torch.randn(n_samples, 1) * 0.3
        noise[:, group_mask] += group_noise
    
    noise = torch.exp(noise)
    noisy_concentrations = base_concentrations * noise
    
    # 9. Zero-inflation that preserves co-occurrence
    # Base zero probability
    base_zero_prob = 0.4  # 40% zeros on average
    
    # Group-based zero patterns (species in same group tend to be absent together)
    zero_noise = torch.randn(n_samples, n_taxa) * 0.3
    for g in range(n_groups):
        group_mask = (primary_group == g)
        # Correlated absence
        group_zero_effect = torch.randn(n_samples, 1) * 0.5
        zero_noise[:, group_mask] += group_zero_effect
    
    # Convert to probabilities
    zero_prob = torch.sigmoid(base_zero_prob + zero_noise)
    
    # Apply zeros
    zero_mask = torch.rand_like(noisy_concentrations) < zero_prob
    concentration_data = noisy_concentrations * (~zero_mask).float()
    
    # 10. Detection limit appropriate for your scale
    detection_limit = 0.5  # Reasonable for data up to 100s
    concentration_data[concentration_data < detection_limit] = 0
    
    # 11. Ensure we stay in your range
    concentration_data = torch.clamp(concentration_data, min=0, max=1000)
    
    return env_data, concentration_data, true_latent_positions

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Assuming the function from the previous step is available
# from your_previous_file import create_synthetic_data

# --- 1. The Unified Neural Network Architecture ---

import torch
import torch.nn as nn
from typing import List

class SeparableVAE(nn.Module):
    def __init__(
        self,
        n_taxa: int,
        n_env_covariates: int,
        k_salient_dims: int,  # Dimensions predictable from environment
        k_private_dims: int,  # Dimensions for everything else
        encoder_hidden_sizes: List[int],
        decoder_hidden_sizes: List[int],
        regressor_hidden_sizes: List[int],
        dropout_p: float = 0.3
    ):
        super().__init__()
        
        # --- Regressor for Salient (Environment-Predicted) Latent Space ---
        regressor_layers = []
        in_size = n_env_covariates
        for h_size in regressor_hidden_sizes:
            regressor_layers.append(nn.Linear(in_size, h_size))
            regressor_layers.append(nn.ReLU())
            in_size = h_size
        self.regressor = nn.Sequential(*regressor_layers)
        # The regressor predicts the parameters of the salient latent space
        self.regressor_mu = nn.Linear(in_size, k_salient_dims)
        self.regressor_logvar = nn.Linear(in_size, k_salient_dims)

        # --- Encoder for Private (Concentration-Only) Latent Space ---
        encoder_layers = []
        in_size = n_taxa
        for h_size in encoder_hidden_sizes:
            encoder_layers.append(nn.Linear(in_size, h_size))
            encoder_layers.append(nn.BatchNorm1d(h_size))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(p=dropout_p))
            in_size = h_size
        self.private_encoder = nn.Sequential(*encoder_layers)
        # The encoder predicts the parameters of the private latent space
        self.private_mu = nn.Linear(in_size, k_private_dims)
        self.private_logvar = nn.Linear(in_size, k_private_dims)
        
        # --- Decoder ---
        # The decoder takes BOTH latent spaces as input
        decoder_input_dim = k_salient_dims + k_private_dims
        decoder_layers = []
        in_size = decoder_input_dim
        for h_size in decoder_hidden_sizes:
            decoder_layers.append(nn.Linear(in_size, h_size))
            decoder_layers.append(nn.BatchNorm1d(h_size))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(p=dropout_p))
            in_size = h_size
        decoder_layers.append(nn.Linear(in_size, n_taxa))
        decoder_layers.append(nn.Hardtanh(min_val=-1.0, max_val=8.0))
        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_conc, x_env):
        # 1. Get parameters for the salient space from the environment
        h_reg = self.regressor(x_env)
        mu_salient = self.regressor_mu(h_reg)
        logvar_salient = self.regressor_logvar(h_reg)
        
        # 2. Get parameters for the private space from the concentrations
        h_enc = self.private_encoder(x_conc)
        mu_private = self.private_mu(h_enc)
        logvar_private = self.private_logvar(h_enc)

        # 3. Sample from both latent distributions
        z_salient = self.reparameterize(mu_salient, logvar_salient)
        z_private = self.reparameterize(mu_private, logvar_private)

        # 4. Concatenate the latent vectors to feed into the decoder
        z_combined = torch.cat([z_salient, z_private], dim=1)
        
        # 5. Reconstruct
        reconstructed_x = self.decoder(z_combined)
        
        # Return all parameters for the loss function
        return reconstructed_x, mu_salient, logvar_salient, mu_private, logvar_private

# --- 2. The Combined Loss Function ---

def loss_function(reconstructed_x, x, mu_salient, logvar_salient, mu_private, logvar_private, beta_salient, beta_private):
    # Reconstruction Loss (unchanged)
    recon_loss = nn.functional.mse_loss(reconstructed_x, x, reduction='mean')
    
    # KL Divergence for the SALIENT space
    kl_salient = -0.5 * torch.mean(torch.sum(1 + logvar_salient - mu_salient.pow(2) - logvar_salient.exp(), dim=1))
    
    # KL Divergence for the PRIVATE space
    kl_private = -0.5 * torch.mean(torch.sum(1 + logvar_private - mu_private.pow(2) - logvar_private.exp(), dim=1))

    # Total loss with separate betas for each latent space
    total_loss = recon_loss + beta_salient * kl_salient + beta_private * kl_private
    
    return total_loss

# --- 3. Helper Functions for Inference ---

@torch.no_grad()
def reconstruct_from_latent(model, z_latent):
    """Takes a latent vector and reconstructs concentrations."""
    model.eval()
    log_conc_rec = model.decoder(z_latent)
    # Inverse transform for log(x+1) is exp(y)-1
    conc_rec = torch.expm1(log_conc_rec)
    conc_rec = torch.clamp(conc_rec, min=0)  # Ensure non-negative
    return conc_rec

@torch.no_grad()
def predict_from_environment(model, x_env):
    """Takes environmental data and predicts full concentrations."""
    model.eval()
    # Predict latent space from environment
    z_pred = model.regressor(x_env)
    # Reconstruct from the predicted latent space
    return reconstruct_from_latent(model, z_pred)


# --- Helper function for R-squared calculation ---
def calculate_r2_score(y_true, y_pred):
    """Calculates an overall R-squared score for multi-output predictions."""
    return r2_score(y_true, y_pred, multioutput='variance_weighted')


# --- 1. Evaluate Predictive Accuracy from Environment ---
@torch.no_grad()
def evaluate_predictive_accuracy(model, test_loader, device):
    """Evaluates how well the model predicts concentrations from environmental data."""
    model.eval()
    all_true_conc = []
    all_pred_conc = []

    for x_conc_log_batch, x_env_batch in test_loader:
        x_conc_log_batch = x_conc_log_batch.to(device)
        x_env_batch = x_env_batch.to(device)
        # Get true concentrations by inverting the log(x+1) transform
        true_conc = torch.expm1(x_conc_log_batch)
        all_true_conc.append(true_conc)
        
        # Get predicted concentrations using the helper function
        pred_conc = predict_from_environment(model, x_env_batch)
        all_pred_conc.append(pred_conc)

    # Combine all batches into two large tensors
    y_true = torch.cat(all_true_conc).cpu().numpy()
    y_pred = torch.cat(all_pred_conc).cpu().numpy()

    # Calculate and print R-squared score
    r2 = calculate_r2_score(y_true, y_pred)
    print(f"--> Predictive Accuracy (Environment -> Concentrations) R²: {r2:.4f}")

    # ... (rest of the function for plotting is unchanged)
    non_rare_taxa = np.where(np.mean(y_true, axis=0) > 0.1)[0]
    if len(non_rare_taxa) == 0:
        print("No non-rare taxa found with mean concentration > 0.1")
        return
    
    taxon = non_rare_taxa[0]
    #print(f"Evaluating predictive accuracy for Taxon {taxon}...")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true[:, taxon], y_pred[:, taxon], alpha=0.3)
    plt.plot([y_true[:, taxon].min(), y_true[:, taxon].max()], [y_true[:, taxon].min(), y_true[:, taxon].max()], '--', color='red', lw=2)
    plt.title(f"Predictive Accuracy for Taxon {taxon}")
    plt.xlabel("True Concentration")
    plt.ylabel("Predicted Concentration")
    plt.grid(True)
    #plt.show()


# --- 2. Evaluate VAE Reconstruction Quality ---
@torch.no_grad()
def evaluate_reconstruction_quality(model, test_loader, device):
    """Evaluates how well the VAE reconstructs concentration data."""
    model.eval()
    all_true_conc = []
    all_rec_conc = []

    for x_conc_log_batch, x_env_batch in test_loader:
        x_conc_log_batch = x_conc_log_batch.to(device)
        x_env_batch = x_env_batch.to(device)
        # Get true concentrations
        true_conc = torch.expm1(x_conc_log_batch)
        all_true_conc.append(true_conc)
        
        # Get reconstructed log concentrations from the VAE part of the model
        rec_x_log, _, _, _ = model(x_conc_log_batch, x_env_batch)
        
        # Invert the log(x+1) transform
        rec_conc = torch.expm1(rec_x_log)
        all_rec_conc.append(rec_conc)

    y_true = torch.cat(all_true_conc).cpu().numpy()
    y_rec = torch.cat(all_rec_conc).cpu().numpy()
    
    r2 = calculate_r2_score(y_true, y_rec)
    print(f"--> VAE Reconstruction Quality R²: {r2:.4f}")


def diagnose_vae_issues(model, train_loader, val_loader, device):
    """
    Comprehensive diagnostics for SupervisedVAE performance issues.
    """
    model.eval()
    print("🔍 DIAGNOSING VAE ISSUES...")
    print("="*60)
    
    with torch.no_grad():
        x_conc_batch, x_env_batch = next(iter(train_loader))
        x_conc_batch = x_conc_batch.to(device)
        x_env_batch = x_env_batch.to(device)
        
        rec_x, mu, logvar, z_pred = model(x_conc_batch, x_env_batch)
        
        print("1. DATA PREPROCESSING CHECK")
        print("-" * 30)
        true_conc = torch.expm1(x_conc_batch)
        rec_conc = torch.expm1(rec_x)
        
        print(f"Input log concentration range: {x_conc_batch.min():.3f} to {x_conc_batch.max():.3f}")
        print(f"True concentration range: {true_conc.min():.3f} to {true_conc.max():.3f}")

        print(f"Reconstructed concentration range: {rec_conc.min():.3f} to {rec_conc.max():.3f}")
        print(f"Zero proportion in true data: {(true_conc < 0.1).float().mean():.3f}")
        print(f"Zero proportion in reconstructed: {(rec_conc < 0.1).float().mean():.3f}")
        
        # 2. CHECK LATENT SPACE
        print("\n2. LATENT SPACE CHECK")
        print("-" * 30)
        print(f"Latent mu range: {mu.min():.3f} to {mu.max():.3f}")
        print(f"Latent std range: {torch.exp(0.5 * logvar).min():.3f} to {torch.exp(0.5 * logvar).max():.3f}")
        print(f"KL divergence per sample: {(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).mean():.3f}")
        
        # Check if latent space is collapsed
        latent_var = torch.var(mu, dim=0)
        print(f"Latent dimension variances: {latent_var.tolist()}")
        collapsed_dims = (latent_var < 0.01).sum().item()
        print(f"Collapsed latent dimensions (var < 0.01): {collapsed_dims}/{len(latent_var)}")
        
        # 3. CHECK ENVIRONMENT -> LATENT PREDICTION
        print("\n3. ENVIRONMENT -> LATENT PREDICTION")
        print("-" * 30)
        env_pred_error = torch.mean((z_pred - mu) ** 2, dim=0)
        print(f"Per-dimension prediction error: {env_pred_error.tolist()}")
        print(f"Mean prediction error: {env_pred_error.mean():.4f}")
        
        # 4. CHECK RECONSTRUCTION QUALITY
        print("\n4. RECONSTRUCTION QUALITY")
        print("-" * 30)
        recon_error = torch.mean((rec_x - x_conc_batch) ** 2, dim=0)
        print(f"Per-taxa reconstruction error (log space): {recon_error.mean():.4f}")
        
        # Convert to concentration space for meaningful metrics
        true_conc_np = true_conc.cpu().numpy()
        rec_conc_np = rec_conc.cpu().numpy()
        
        # Overall R2
        r2_overall = r2_score(true_conc_np.flatten(), rec_conc_np.flatten())
        print(f"Overall R² (concentration space): {r2_overall:.4f}")
        
        # Per-taxa R2
        r2_per_taxa = []
        for i in range(true_conc_np.shape[1]):
            if np.var(true_conc_np[:, i]) > 1e-6:  # Only if there's variance
                r2_taxa = r2_score(true_conc_np[:, i], rec_conc_np[:, i])
                r2_per_taxa.append(r2_taxa)
        
        if r2_per_taxa:
            print(f"Mean per-taxa R²: {np.mean(r2_per_taxa):.4f}")
            print(f"Best taxa R²: {np.max(r2_per_taxa):.4f}")
            print(f"Worst taxa R²: {np.min(r2_per_taxa):.4f}")
    
    return {
        'latent_collapse': collapsed_dims,
        'prediction_error': env_pred_error.mean().item(),
        'reconstruction_r2': r2_overall,
        'per_taxa_r2': r2_per_taxa
    }

@torch.no_grad()
def generate_full_report(model, data_loader, device):
    """
    Runs a comprehensive evaluation for the SeparableVAE and returns all key metrics.
    """
    model.eval()
    
    # Buffers to store results from all batches
    all_true_conc = []
    all_rec_conc = []
    all_mu_s, all_logvar_s = [], []
    all_mu_p, all_logvar_p = [], []
    all_recon_error = []

    for x_conc_log_batch, x_env_batch in data_loader:
        x_conc_log_batch = x_conc_log_batch.to(device)
        x_env_batch = x_env_batch.to(device)

        # Forward pass with the new model
        rec_x_log, mu_s, logvar_s, mu_p, logvar_p = model(x_conc_log_batch, x_env_batch)

        # --- Store values for later computation ---
        all_true_conc.append(torch.expm1(x_conc_log_batch))
        all_rec_conc.append(torch.expm1(rec_x_log))
        all_mu_s.append(mu_s)
        all_logvar_s.append(logvar_s)
        all_mu_p.append(mu_p)
        all_logvar_p.append(logvar_p)
        all_recon_error.append(nn.functional.mse_loss(rec_x_log, x_conc_log_batch).item())

    # Concatenate all batch results into single numpy arrays
    y_true = torch.cat(all_true_conc).cpu().numpy()
    y_rec = torch.cat(all_rec_conc).cpu().numpy()
    mu_s_full, logvar_s_full = torch.cat(all_mu_s).cpu().numpy(), torch.cat(all_logvar_s).cpu().numpy()
    mu_p_full, logvar_p_full = torch.cat(all_mu_p).cpu().numpy(), torch.cat(all_logvar_p).cpu().numpy()

    # --- Calculate all metrics ---
    
    # 1. Reconstruction Metrics
    recon_metrics = {
        "overall_r2": r2_score(y_true, y_rec, multioutput='variance_weighted'),
        "mean_per_taxa_r2": np.mean(r2_score(y_true, y_rec, multioutput='raw_values')),
        "median_per_taxa_r2": np.median(r2_score(y_true, y_rec, multioutput='raw_values')),
        "mean_log_space_error": np.mean(all_recon_error)
    }

    # 2. Latent Space Diagnostics Function
    def get_latent_space_diags(mu, logvar):
        kl_divergence = -0.5 * np.mean(np.sum(1 + logvar - mu**2 - np.exp(logvar), axis=1))
        variance = np.var(mu, axis=0)
        return {
            "mean_kl_divergence_per_sample": float(kl_divergence),
            "latent_dimension_variance": variance.tolist(),
            "collapsed_dimensions": int(np.sum(variance < 0.01))
        }

    # --- Assemble the final report dictionary ---
    report = {
        "reconstruction_quality": {k: float(v) for k, v in recon_metrics.items()},
        "salient_space_diagnostics": get_latent_space_diags(mu_s_full, logvar_s_full),
        "private_space_diagnostics": get_latent_space_diags(mu_p_full, logvar_p_full)
    }

    return json.dumps(report, indent=4)

# --- Modified main block to include a test set and final evaluation ---

if __name__ == '__main__':
    # --- Device Configuration ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("MPS not found, using CPU")

    # --- Hyperparameters ---
    N_SAMPLES = 5000
    N_TAXA = 50
    N_ENV_COVARIATES = 5
    
    # Latent space sizes for the SeparableVAE architecture
    K_SALIENT_DIMS = 10   # Dimensions for environment-predictable patterns
    K_PRIVATE_DIMS = 20   # Dimensions for all other patterns

    # Model architecture sizes
    ENCODER_SIZES = [512, 256, 128]
    DECODER_SIZES = [256, 512, 768]
    REGRESSOR_SIZES = [128, 128, 64]

    # Regularization for each latent space
    BETA_SALIENT = 0.0001
    BETA_PRIVATE = 0.005

    # --- Training Parameters ---
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 2048
    EPOCHS = 1000

    # --- Data Preparation ---
    env_data, conc_data, true_latents = create_synthetic_data(
        N_SAMPLES, N_TAXA, N_ENV_COVARIATES, K_SALIENT_DIMS + K_PRIVATE_DIMS
    )
    
    # Use the stable log(x+1) transform on concentration data
    conc_data_log = torch.log1p(conc_data)

    # --- 1. Split data indices BEFORE creating datasets ---
    dataset_size = len(conc_data_log)
    indices = list(range(dataset_size))
    np.random.shuffle(indices) # Shuffle indices once

    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]
    
    # Separate the environmental data for scaling
    env_data_train = env_data[train_indices]
    env_data_val = env_data[val_indices]
    env_data_test = env_data[test_indices]
    
    # --- 2. Create and fit the scaler ONLY on the training set ---
    scaler = StandardScaler()
    scaler.fit(env_data_train)
    
    # --- 3. Apply the FITTED scaler to all data splits ---
    env_data_train_scaled = torch.tensor(scaler.transform(env_data_train), dtype=torch.float32)
    env_data_val_scaled = torch.tensor(scaler.transform(env_data_val), dtype=torch.float32)
    env_data_test_scaled = torch.tensor(scaler.transform(env_data_test), dtype=torch.float32)
    
    # --- 4. Create Datasets and DataLoaders ---
    # Create the full dataset
    full_dataset = TensorDataset(conc_data_log, env_data) # Note: env_data will be replaced below
    
    # Re-combine the scaled environmental data with the concentration data
    train_dataset = TensorDataset(conc_data_log[train_indices], env_data_train_scaled)
    val_dataset = TensorDataset(conc_data_log[val_indices], env_data_val_scaled)
    test_dataset = TensorDataset(conc_data_log[test_indices], env_data_test_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # --- Model Initialization ---
    model = SeparableVAE(
        n_taxa=N_TAXA,
        n_env_covariates=N_ENV_COVARIATES,
        k_salient_dims=K_SALIENT_DIMS,
        k_private_dims=K_PRIVATE_DIMS,
        encoder_hidden_sizes=ENCODER_SIZES,
        decoder_hidden_sizes=DECODER_SIZES,
        regressor_hidden_sizes=REGRESSOR_SIZES,
        dropout_p=0.3
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=20)
    
    # --- Training Loop ---
    print("Starting model training...")
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_train_loss = 0
        for x_conc_batch, x_env_batch in train_loader:
            x_conc_batch = x_conc_batch.to(device)
            x_env_batch = x_env_batch.to(device)

            optimizer.zero_grad()
            rec_x, mu_s, logvar_s, mu_p, logvar_p = model(x_conc_batch, x_env_batch)
            loss = loss_function(rec_x, x_conc_batch, mu_s, logvar_s, mu_p, logvar_p, BETA_SALIENT, BETA_PRIVATE)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_conc_batch, x_env_batch in val_loader:
                x_conc_batch = x_conc_batch.to(device)
                x_env_batch = x_env_batch.to(device)
                rec_x, mu_s, logvar_s, mu_p, logvar_p = model(x_conc_batch, x_env_batch)
                loss = loss_function(rec_x, x_conc_batch, mu_s, logvar_s, mu_p, logvar_p, BETA_SALIENT, BETA_PRIVATE)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
             print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # --- Final Evaluation ---

    report = generate_full_report(model, test_loader, device)

    with open('vae_report.json', 'w') as f:
        f.write(report)
