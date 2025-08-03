import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import multioutput
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Data transformation
class PowerRootStandardScaler(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible scaler that applies a root transformation (1/root)
    followed by standardization (zero mean, unit variance), and inverts both.

    Parameters:
        root (float): Degree of the root transformation (e.g., 2 for sqrt, 3 for cbrt)
    """
    def __init__(self, root=3):
        assert root > 0, "Root must be a positive number."
        self.root = root
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X_root = np.sign(X) * np.abs(X) ** (1 / self.root)
        self.scaler.fit(X_root)
        return self

    def transform(self, X):
        X_root = np.sign(X) * np.abs(X) ** (1 / self.root)
        return self.scaler.transform(X_root)

    def inverse_transform(self, X_scaled):
        X_root = self.scaler.inverse_transform(X_scaled)
        return np.sign(X_root) * np.abs(X_root) ** self.root

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

# Load data
def load_data(path, taxonomy_path='ifcb_taxonomy.csv'):
    """
    Loads and preprocesses IFCB data using taxonomy and prevalence filtering.
    Returns raw environmental features and filtered taxa abundances.

    Parameters:
        path: str — Path to CSV with abundance data
        taxonomy_path: str — Path to taxonomy CSV

    Returns:
        X_env: pd.DataFrame — Environmental predictors
        Y_filtered: pd.DataFrame — Taxa abundance matrix (untransformed)
        taxa_columns: list — List of taxa column names
    """

    df = pd.read_csv(path)
    taxonomy = pd.read_csv(taxonomy_path)
    df['sample_time'] = pd.to_datetime(df['sample_time'])

    # --- Taxa column selection using taxonomy ---
    cols_to_drop = [
        'nanoplankton_mix'
        # optionally drop unknown/misc groups
    ]
    taxa_labels = taxonomy['Annotations'].unique()
    taxa_cols = [col for col in df.columns if col in taxa_labels and col not in cols_to_drop]
    dataset = df[taxa_cols].copy()

    # --- Prevalence filtering ---
    presence = (dataset > dataset.quantile(0.01, axis=0)).astype(int)
    prevalence = presence.sum(axis=0) / presence.shape[0]
    keep_mask = (prevalence > 0) #& (prevalence <= 0.8) 
    Y_filtered = dataset.loc[:, keep_mask]

    # --- Remove rows with zero total abundance ---
    Y_filtered = Y_filtered[Y_filtered.sum(axis=1) > 0].reset_index(drop=True)

    # --- Align metadata with kept samples ---
    df = df.loc[Y_filtered.index].reset_index(drop=True)

    # --- Temporal features ---
    df = add_temporal_cycles(df, time_col='sample_time')

    # --- Environmental predictors ---
    env_columns = ['latitude', 'longitude', 'temperature', 'salinity', 'doy']
    X_env = df[env_columns].copy().reset_index(drop=True)

    print("Environmental variables shape:", X_env.shape)
    print("Filtered taxa matrix shape:", Y_filtered.shape)

    return X_env, Y_filtered, Y_filtered.columns.tolist(), df

def add_temporal_cycles(df, time_col='sample_time'):
    """
    Adds cyclic temporal features to the DataFrame based on sample_time.

    Features added:
        - day_sin, day_cos: encodes time of day
        - year_sin, year_cos: encodes time of year

    Parameters:
        df: pandas DataFrame with a datetime column
        time_col: name of the datetime column (timezone-aware is okay)

    Returns:
        df_with_cycles: original DataFrame with 4 new columns
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # Time of day (in seconds)
    seconds_in_day = 24 * 60 * 60
    time_sec = df[time_col].dt.hour * 3600 + df[time_col].dt.minute * 60 + df[time_col].dt.second
    df['day_sin'] = np.sin(2 * np.pi * time_sec / seconds_in_day)
    df['day_cos'] = np.cos(2 * np.pi * time_sec / seconds_in_day)

    # Day of year
    day_of_year = df[time_col].dt.dayofyear
    df['year_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['year_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

    return df

# Variational Autoencoder (VAE) Model
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, hidden_dims=(128, 64)):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(last_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            # encoder_layers.append(nn.Dropout(p=0.1))  # Add 10% dropout
            last_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mu = nn.Linear(last_dim, latent_dim)
        self.fc_logvar = nn.Linear(last_dim, latent_dim)

        # Decoder
        decoder_layers = []
        last_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(last_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            # decoder_layers.append(nn.Dropout(p=0.2))  # Add 20% dropout
            last_dim = h_dim
        decoder_layers.append(nn.Linear(last_dim, input_dim))  # output layer
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=0.1, feature_weights=None, labels=None):
    # Reconstruction loss (MSE)
    recon_loss_good = F.mse_loss(recon_x[labels > 0], x[labels > 0], reduction='mean')

    if recon_x[labels == 0].shape[0] == 0:
        recon_loss_bad = torch.tensor(0.0, device=recon_x.device)
    else:
        recon_loss_bad = F.mse_loss(recon_x[labels <= 0], x[labels <= 0], reduction='mean')

    # Feature weights for reconstruction loss
    if feature_weights is not None:
        recon_loss_good = (recon_loss_good * feature_weights).mean()
        recon_loss_bad = (recon_loss_bad * (1 - feature_weights)).mean()

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # centering the latent space
    latent_centering_loss = torch.mean(mu.pow(2))
    alpha = 0
    gamma = 0.5 * (recon_loss_good.item() / (recon_loss_bad.item() + 1e-8))

    # FIXME sanity check

    # return recon_loss + beta * kl_loss, recon_loss, kl_loss
    return recon_loss_good + (beta * kl_loss) + (alpha * latent_centering_loss) + (gamma * recon_loss_bad), \
              recon_loss_good, recon_loss_bad, kl_loss

def train_vae(model, dataloader, optimizer, device, beta=0.01, epochs=500, feature_weights=None):
    model.train()
    model.to(device)
    feature_weights = feature_weights.to(device)
    def sigmoid_annealing(epoch, total_epochs=10000, max_beta=1.0, steepness=5):
        warmup = 100
        if epoch < warmup:
            return 0
        else:
            epoch -= warmup
            total_epochs -= warmup
        midpoint = total_epochs / 3
        return max_beta / (1 + np.exp(-steepness * (epoch - midpoint) / total_epochs))
    training_curve = []
    for epoch in range(epochs):
        beta = sigmoid_annealing(epoch, epochs, max_beta=0.01, steepness=100)
        total_loss = 0
        for batch, labels in dataloader:
            batch = batch.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, recon_loss_good, recon_loss_bad, kl_loss = vae_loss(
                recon, batch, mu, logvar, beta=beta,
                feature_weights=feature_weights, labels=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon_loss = recon_loss_good.item()
            total_bad_loss = recon_loss_bad.item()
            total_kl_loss = kl_loss.item()
            training_curve.append({
                'epoch': epoch,
                'beta': beta,
                'loss': total_loss,
                'recon_loss_good': total_recon_loss,
                'recon_loss_bad': total_bad_loss,
                'kl_loss': total_kl_loss
            })
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}, Beta: {beta:.4f}, Loss: {total_loss:.4f}, Recon Loss: {total_recon_loss:.4f}, Bad Recon Loss: {total_bad_loss:.4f}, KL Loss: {total_kl_loss:.4f}", flush=True)

    df = pd.DataFrame.from_records(training_curve)
    df.to_csv('vae_training_curve.csv', index=False)
    from plot_training_curves import plot_training_curve
    plot_training_curve(df)

    return model

def visualize_latent_space(vae, Y_scaled, X_env, color_var='salinity', scaler=None,
                           method='pca', perplexity=30, random_state=42, device='cpu'):
    """
    Visualize latent space of a trained VAE using t-SNE, PCA, or UMAP.

    Parameters:
    - vae: trained VAE model
    - Y_scaled: pandas DataFrame of scaled concentration data (input to VAE)
    - X_env: pandas DataFrame of environmental variables
    - color_var: column name in X_env to color points by
    - method: 'tsne', 'pca', or 'umap'
    - perplexity: only used for t-SNE
    - random_state: reproducibility
    - device: 'cpu' or 'cuda'
    """
    vae.eval()
    vae.to(device)

    # plot the taxax concentrations as a violin plot
    plt.figure(figsize=(12, 6))
    plt.violinplot(scaler.inverse_transform(Y_scaled.values), showmeans=True, showmedians=True)
    plt.title('Taxa Concentrations Distribution')
    plt.xlabel('Taxa')
    plt.ylabel('Scaled Concentration')
    plt.tight_layout()
    # plt.show()
    plt.close()

    # Step 1: Encode all samples into latent space
    X_tensor = torch.tensor(Y_scaled.values, dtype=torch.float32)
    mu_list = []
    
    with torch.no_grad():
        for i in range(0, X_tensor.shape[0], 128):
            batch = X_tensor[i:i+128].to(device)
            mu, _ = vae.encode(batch)
            mu_list.append(mu.cpu().numpy())

    latent_mu = np.vstack(mu_list)

    # Step 2: Dimensionality reduction
    for method in ['tsne', 'pca', 'umap']:
        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=perplexity)
        elif method == 'pca':
            reducer = PCA(n_components=2)
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2)
            except ImportError:
                raise ImportError("UMAP is not installed. Please install it with 'pip install umap-learn'.")
        else:
            raise ValueError("Method must be 'tsne' or 'pca'.")

        latent_2d = reducer.fit_transform(latent_mu)

        # Step 3: Plot
        color_vals = X_env.loc[Y_scaled.index, color_var]

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
                            c=color_vals, cmap='viridis', s=20, alpha=0.8)
        plt.colorbar(scatter, label=f'{color_var}')
        plt.title(f'{method.upper()} of VAE Latent Space (colored by {color_var})')
        plt.xlabel(f'{method.upper()} dim 1')
        plt.ylabel(f'{method.upper()} dim 2')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()

    # now make a box plot of the latent space
    plt.figure(figsize=(8, 6))
    plt.boxplot(latent_mu)
    plt.title('Latent Space Distribution')
    plt.xlabel('Latent Dimensions')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.show()

    # now choose a latent dimention z1
    # and construct a transect through z1 from its min to its max over n steps where n is 256
    n_steps = 100
    for d in range(latent_mu.shape[1]):
        z_mean = np.tile(np.mean(latent_mu, axis=0), (n_steps, 1))
        # compute the first and third quantile of the latent_mu for this dimension
        q5 = np.percentile(latent_mu[:, d], 5)
        q95 = np.percentile(latent_mu[:, d], 95)
        z_mean[:, d] = np.linspace(q5, q95, n_steps)
        # now decode these latent vectors
        z_transect_tensor = torch.tensor(z_mean, dtype=torch.float32).to(device)
        # decode
        recon_transect = decode_latent_means(z_transect_tensor, vae, device=device)
        recon_transect = scaler.inverse_transform(recon_transect)
        recon_transect = pd.DataFrame(recon_transect, columns=Y_scaled.columns)
        # clamp the recon_transect to be non-negative
        recon_transect = np.clip(recon_transect, 0, None)
        # now make a stacked plot of the transect with a diferent color for each taxon
        plt.figure(figsize=(10, 6))
        x_values = z_mean[:, d]
        bar_width = np.min(np.diff(x_values))  # Minimum spacing between consecutive x values
        for i, taxon in enumerate(Y_scaled.columns):
            plt.bar(x_values, recon_transect[taxon], bottom=recon_transect.iloc[:, :i].sum(axis=1), label=taxon, width=bar_width)
        plt.title('Transect through Latent Space (z{})'.format(d + 1))
        plt.xlabel('Latent Dimension z{}'.format(d + 1))
        plt.ylabel('Reconstructed Concentration')
        plt.grid(True)
        plt.show()

# Encoder and Decoder Functions
def encode_latent_means(Y_scaled, vae, device='cpu'):
    """
    Encode scaled concentrations into latent mean vectors.

    Parameters:
        Y_scaled (np.ndarray): Scaled concentration data (n_samples, n_taxa)
        vae (torch.nn.Module): Trained VAE model
        device (str): 'cpu', 'cuda', or 'mps'

    Returns:
        np.ndarray: Latent mean vectors (n_samples, latent_dim)
    """
    Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32, device=device)

    vae.eval()
    vae.to(device)

    with torch.no_grad():
        mu, _ = vae.encode(Y_tensor)

    return mu.cpu().numpy()

def decode_latent_means(z_pred, vae, device='cpu'):
    """
    Decode latent vectors into predicted concentrations (in standardized space).

    Parameters:
        z_pred (np.ndarray): Latent vectors of shape (n_samples, latent_dim)
        vae (torch.nn.Module): Trained VAE model
        device (str): 'cpu', 'cuda', or 'mps'

    Returns:
        np.ndarray: Predicted concentrations (n_samples, n_taxa), still in scaled space
    """
    z_tensor = torch.tensor(z_pred, dtype=torch.float32, device=device)

    vae.eval()
    with torch.no_grad():
        recon = vae.decode(z_tensor).cpu().numpy()

    return recon

# Data splitting for regression tasks
def split_regression_data(
    X_env, Y_conc, Z_target,
    val_size=0.2,
    test_size=0.2,
    random_state=42,
    holdout_last=False
):
    """
    Splits data for time series ML:
    - Test set is either the first or last test_size fraction (chronological hold-out controlled by holdout_last)
    - Train/val split is random (unless val_size=0), then re-sorted chronologically

    Parameters:
        X_env: pandas DataFrame of environmental covariates (n_samples, n_features)
        Y_conc: pandas DataFrame of concentrations
        Z_target: numpy array of VAE latent means (n_samples, latent_dim)
        val_size: float, fraction of data (excluding test) used for validation
        test_size: float, fraction of data used for test
        random_state: int, seed for reproducibility
        holdout_last: bool, if True, test set is taken from the end; else from the beginning

    Returns:
        Dict with train/val/test splits and their indices (sorted chronologically)
    """
    n = len(X_env)
    all_indices = np.arange(n)

    # Step 1: Chronological test set
    test_len = int(n * test_size)
    if holdout_last:
        idx_test = all_indices[-test_len:]
        idx_remain = all_indices[:-test_len]
    else:
        idx_test = all_indices[:test_len]
        idx_remain = all_indices[test_len:]

    # Step 2: Random train/val split
    if val_size > 0:
        val_ratio = val_size / (1 - test_size)
        idx_train, idx_val = train_test_split(
            idx_remain, test_size=val_ratio, random_state=random_state
        )
        idx_train = np.sort(idx_train)
        idx_val = np.sort(idx_val)
    else:
        idx_train = np.sort(idx_remain)
        idx_val = np.array([], dtype=int)  # empty index array

    # Helper function to select rows by index
    def select(arr, idx):
        if isinstance(arr, pd.DataFrame):
            return arr.iloc[idx]
        elif isinstance(arr, np.ndarray):
            return arr[idx]
        else:
            raise TypeError("Unsupported array type.")

    return {
        'X_train': select(X_env, idx_train),
        'X_val': select(X_env, idx_val),
        'X_test': select(X_env, idx_test),

        'Z_train': select(Z_target, idx_train),
        'Z_val': select(Z_target, idx_val),
        'Z_test': select(Z_target, idx_test),

        'Y_train': select(Y_conc, idx_train),
        'Y_val': select(Y_conc, idx_val),
        'Y_test': select(Y_conc, idx_test),

        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test
    }

# Train Random Forest Regressor
def train_regressor(X_train, Z_train, use_grid_search=False):
    """
    Trains a multi-output Random Forest Regressor to predict VAE latent means.

    Parameters:
        X_train: pandas DataFrame of shape (n_samples, n_features)
        Z_train: numpy array of shape (n_samples, latent_dim)
        use_grid_search: bool, whether to perform hyperparameter tuning

    Returns:
        Trained MultiOutputRegressor instance
    """
    base_rf = RandomForestRegressor(n_jobs=-1, random_state=42)
    rf_model = MultiOutputRegressor(base_rf)

    if use_grid_search:
        param_grid = {
            'estimator__n_estimators': [200],
            'estimator__max_depth': [None, 20]
        }

        grid_search = GridSearchCV(
            rf_model,
            param_grid,
            cv=3,
            scoring='r2',
            verbose=2,
            n_jobs=-1
        )

        grid_search.fit(X_train, Z_train)
        print("Best parameters:", grid_search.best_params_)
        print("Best cross-validated R² score:", grid_search.best_score_)
        return grid_search.best_estimator_
    else:
        rf_model.fit(X_train, Z_train)
        return rf_model

# prepare taxon weights for loss function
# inverse prevalence weights
def compute_taxon_weights(y):
    """
    Compute weights for each taxon based on inverse prevalence and dynamic range scaling.
    Assign zero weights to all-zero taxa.
    """
    eps = 1e-8  # Small constant to prevent division by zero
    mean_concentrations = np.mean(y, axis=0)
    all_zero_mask = (mean_concentrations == 0)

    # Assign zero weights to all-zero taxa
    weights = np.zeros_like(mean_concentrations)
    non_zero_mask = ~all_zero_mask

    # Compute prevalence (fraction of non-zero samples for each taxon)
    prevalence = np.sum(y > 0, axis=0) / y.shape[0]

    # Combine prevalence and mean concentration to compute weights
    # Higher prevalence and higher mean concentration result in lower weights
    combined_metric = prevalence[non_zero_mask] * mean_concentrations[non_zero_mask]
    weights[non_zero_mask] = 1 / (prevalence[non_zero_mask] + eps)

    # Normalize weights for non-zero taxa to sum to 1
    weights[non_zero_mask] /= np.sum(weights[non_zero_mask])

    # return torch.tensor(weights, dtype=torch.float32)

    # return torch.tensor(weights, dtype=torch.float32)

    # Debugging: Print weights and all-zero taxa
    print("All-zero taxa:", all_zero_mask)
    print("Taxon weights:", weights)

    # Plot weights
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(weights)), weights)
    plt.title('Taxon Weights for VAE Loss')
    plt.xlabel('Taxa')
    plt.ylabel('Weight')
    plt.xticks(rotation=90)
    plt.yscale('log')  # Log scale for better visibility
    plt.tight_layout()
    # plt.show()
    plt.close()
    print("Taxon weights:", weights)

    return torch.tensor(weights, dtype=torch.float32)

def generate_implausible_data_by_shuffling(real_data):
    """
    Generate implausible data by shuffling taxa within each sample.

    Parameters:
        real_data (np.ndarray): Real data matrix (n_samples, n_taxa)

    Returns:
        np.ndarray: Implausible data matrix with shuffled taxa
    """
    implausible_data = real_data.copy()
    for i in range(real_data.shape[0]):
        implausible_data[i] = np.random.permutation(real_data[i])
    return implausible_data

class IdentityScaler:
    """
    Identity scaler that does nothing.
    Used to skip scaling in this example.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X

    def inverse_transform(self, X_scaled):
        return X_scaled
        
# Train and Evaluate the VAE Workflow
def train_all(path='ifcb_count_clean.csv'):
    # === 0. Load and Prepare Data ===
    X, Y, taxa, _ = load_data(path)

    # === 1. Preprocess Taxa Abundance Data ===
    # scaler = PowerRootStandardScaler(root=3)  # Custom scaler

    scaler = IdentityScaler()
    Y_scaled = scaler.fit_transform(Y.values)  # Scaled input for VAE

    # === 2. Train/Val/Test Split ===
    # Dummy latent matrix is used for splitting but not needed for training yet
    Z_dummy = np.zeros((len(Y), 1))

    # Returns dictionary with X, Y, and index splits
    splits = split_regression_data(X, Y_scaled, Z_dummy, val_size=0, test_size=0.15, random_state=42)

    # === 3. Train Variational Autoencoder (VAE) ===
    # Prepare training data
    Y_train = splits['Y_train']
    # Y_implausible = generate_implausible_data_by_shuffling(Y_train)
    Y_labels = np.ones(Y_train.shape[0])  # Labels for real data
    implausible = True  # Set to True to generate implausible data
    if implausible:
        # generate implausible data by shuffling taxa within each sample
        Y_implausible = generate_implausible_data_by_shuffling(Y_train)[:Y_train.shape[0] // 2]  # Take half of the training data
        Y_implausible_labels = np.zeros(Y_implausible.shape[0])  # Labels for implausible data
        # Combine real and implausible labels
        Y_labels = np.concatenate((Y_labels, Y_implausible_labels))
        Y_train = np.vstack((Y_train, Y_implausible))
        # now shuffle the data and labels together
        indices = np.arange(Y_train.shape[0])
        np.random.shuffle(indices)
        Y_train = Y_train[indices]
        Y_labels = Y_labels[indices]

    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    Y_labels_tensor = torch.tensor(Y_labels, dtype=torch.float32)
    train_dataset = TensorDataset(Y_train_tensor, Y_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)

    w = compute_taxon_weights(splits['Y_train'])

    # Initialize VAE
    vae = VAE(input_dim=Y.shape[1], latent_dim=3)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-3)

    # Detect device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Train model
    train_vae(vae, train_loader, optimizer, beta=0.3, epochs=2000, feature_weights=w, device=device)

    # Save trained model
    torch.save(vae.state_dict(), 'vae_model.pth')
    print("✅ VAE model saved.")

    # === 4. Encode Latent Representations ===
    Z_train = encode_latent_means(splits['Y_train'], vae, device=device)
    Z_val   = encode_latent_means(splits['Y_val'],   vae, device=device)
    Z_test  = encode_latent_means(splits['Y_test'],  vae, device=device)

    # === 5. Train Random Forest on Environmental Covariates ===
    rf_model = train_regressor(splits['X_train'], Z_train)

    # Save RF model
    torch.save(rf_model, 'rf_model.pth')
    print("✅ Random Forest model saved.")

    # === 6. Predict & Decode ===
    # Predict latent variables from environmental covariates
    Z_test_pred = rf_model.predict(splits['X_test'])
    print("Predicted latent means shape:", Z_test_pred.shape)

    # Decode latent vectors into scaled concentration predictions
    Y_test_pred_scaled = decode_latent_means(Z_test_pred, vae, device=device)

    # Inverse scale to get concentrations in original units
    Y_test_pred = scaler.inverse_transform(Y_test_pred_scaled)
    Y_test_pred = pd.DataFrame(Y_test_pred, columns=taxa)
    print("Predicted concentrations shape:", Y_test_pred.shape)

    # === 7. Recover Ground Truth Test Set ===
    Y_test_true = scaler.inverse_transform(splits['Y_test'])
    print("True concentrations shape:", Y_test_true.shape)

    # === 8. Visualize Predictions vs Ground Truth ===
    plt.figure(figsize=(12, 6))

    # True concentrations
    plt.subplot(1, 2, 1)
    plt.violinplot(Y_test_true, showmeans=True, showmedians=True)
    plt.title('True Concentrations (Test Set)')
    plt.xlabel('Taxa')
    plt.ylabel('Concentration')

    # Predicted concentrations
    plt.subplot(1, 2, 2)
    plt.violinplot(Y_test_pred.values, showmeans=True, showmedians=True)
    plt.title('Predicted Concentrations (Test Set)')
    plt.xlabel('Taxa')
    plt.ylabel('Concentration')

    plt.tight_layout()
    plt.show()

    # === 9. Evaluate Model Performance ===
    r2_scores = r2_score(Y_test_true, Y_test_pred, multioutput='raw_values')       # per taxon
    mean_r2 = r2_score(Y_test_true, Y_test_pred, multioutput='uniform_average')    # overall

    print("R² scores per taxon:", r2_scores)
    print("Mean R² score:", mean_r2)

# Visualize latent space
def vis_all(path='ifcb_count_clean.csv'):
    """
    Main function to run the entire training and evaluation pipeline.
    """
    # Load data and preprocess
    X_env, Y, taxa_columns, _ = load_data(path)

    # Apply transformation and scaling
    # scaler = PowerRootStandardScaler(root=3)  # Custom scaler
    scaler = IdentityScaler()  # Use identity scaler for this example
    Y_scaled = pd.DataFrame(
        scaler.fit_transform(Y),
        columns=taxa_columns,
        index=Y.index
    )

    # Visualize latent space
    vae = VAE(input_dim=Y_scaled.shape[1], latent_dim=3)
    vae.load_state_dict(torch.load('vae_model.pth'))
    #v isualize_latent_space(vae, Y_scaled, X_env, color_var='salinity', scaler=scaler, method='pca', device='mps' if torch.backends.mps.is_available() else 'cpu')
    visualize_latent_space(vae, Y_scaled, X_env, color_var='salinity', scaler=scaler, method='pca', device='mps' if torch.backends.mps.is_available() else 'cpu')

if __name__ == "__main__":
    print("Starting VAE training and visualization...")
    # Run the training and evaluation pipeline
    filepath = 'data/oleander/ifcb_count_clean.csv'
    train_all(path=filepath)
    vis_all(path=filepath)
