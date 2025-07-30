import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import multioutput
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Load data
def load_data(path):
    df = pd.read_csv(path)

    # Step 2: Split into environmental and taxa columns
    # Identify start of taxa columns — assumed to start at 'Acanthoica'
    taxa_start_col = 'Acanthoica'
    taxa_columns = df.columns[df.columns.get_loc(taxa_start_col):]

    # env columns are
    sample_time = pd.to_datetime(df['sample_time'])
    df = add_temporal_cycles(df, time_col='sample_time')
    env_columns = ['latitude', 'longitude', 'temperature', 'salinity', 'day_sin', 'day_cos', 'year_sin', 'year_cos']
    # Extract DataFrames
    X_env = df[env_columns].copy()
    Y_raw = df[taxa_columns].copy()

    # Step 3: Compute concentrations (handle missing or zero ml_analyzed)
    # ml_analyzed = df['ml_analyzed'].replace(0, np.nan)  # Avoid division by zero
    # Y_conc = Y_raw.div(ml_analyzed, axis=0)

    # # Optional: Drop samples with missing volume
    # valid_samples = ml_analyzed.notna()
    # Y_conc = Y_conc[valid_samples]
    # X_env = X_env[valid_samples]
    Y_conc = Y_raw.copy() # Input data is already in concentration units

    # Step 4: Apply asinh transform (works well for 0s and large values)
    Y_trans = np.arcsinh(Y_conc)

    # Step 5: Standardize each taxon (column-wise)
    # Optionally, filter out columns with too many zeros (e.g., <5% non-zero)
    min_nonzero_fraction = 0.025
    nonzero_fraction = (Y_conc > 0).mean()
    kept_taxa = nonzero_fraction[nonzero_fraction > min_nonzero_fraction].index
    # remove taxa with very high mean concentration
    mean_concentration = Y_conc[kept_taxa].mean()
    kept_taxa = kept_taxa[mean_concentration < 5]  #

    Y_filtered = Y_trans[kept_taxa]

    # Scale each taxon column (StandardScaler by default)
    scaler = StandardScaler()
    Y_scaled = pd.DataFrame(
        scaler.fit_transform(Y_filtered),
        columns=Y_filtered.columns,
        index=Y_filtered.index
    )

    # Final output
    print("Environmental variables shape:", X_env.shape)
    print("Transformed + filtered taxa shape:", Y_scaled.shape)

    return X_env, Y_scaled, kept_taxa.tolist(), scaler

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

def vae_loss(recon_x, x, mu, logvar, beta=0.1):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def train_vae(model, dataloader, optimizer, device, beta=0.01, epochs=500):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon_loss = recon_loss.item()
            total_kl_loss = kl_loss.item()
        if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Recon Loss: {total_recon_loss:.4f}, KL Loss: {total_kl_loss:.4f}")

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
    plt.violinplot(np.sinh(Y_scaled.values), showmeans=True, showmedians=True)
    plt.title('Taxa Concentrations Distribution')
    plt.xlabel('Taxa')
    plt.ylabel('Scaled Concentration')
    plt.tight_layout()
    #plt.show()
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
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    elif method == 'pca':
        reducer = PCA(n_components=2)
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

    # now make a violin plot of the latent space
    plt.figure(figsize=(8, 6))
    plt.violinplot(latent_mu, showmeans=True, showmedians=True)
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
        z_mean[:, d] = np.linspace(latent_mu[:, d].min(), latent_mu[:, d].max(), n_steps)
        # now decode these latent vectors
        z_transect_tensor = torch.tensor(z_mean, dtype=torch.float32).to(device)
        # decode
        recon_transect = decode_and_inverse_transform(z_transect_tensor, vae, scaler, Y_scaled.columns, device=device)
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

def decode_and_inverse_transform(z_pred, vae, scaler, taxa_columns, device='cpu'):
    """
    Decode latent vectors and invert preprocessing to get predicted concentrations.
    
    Parameters:
        z_pred: np.ndarray or torch.Tensor of shape (n_samples, latent_dim)
        vae: trained VAE model
        scaler: fitted StandardScaler used during preprocessing
        taxa_columns: list of taxa column names (used to return a DataFrame)
        device: 'cpu' or 'cuda' based on where the model lives
        
    Returns:
        concentrations_pred: pd.DataFrame of shape (n_samples, n_taxa) in original units
    """
    # Ensure tensor input
    if not isinstance(z_pred, torch.Tensor):
        z_pred = torch.tensor(z_pred, dtype=torch.float32)

    z_pred = z_pred.to(device)

    # Decode
    vae.eval()
    with torch.no_grad():
        recon_scaled = vae.decode(z_pred).cpu().numpy()

    # Inverse scale
    recon_transformed = scaler.inverse_transform(recon_scaled)

    # Inverse asinh transform → sinh to recover original scale
    recon_concentration = np.sinh(recon_transformed)

    # Format as DataFrame
    concentrations_pred = pd.DataFrame(recon_concentration, columns=taxa_columns)

    return concentrations_pred

# regressor

def encode_latent_means(Y_scaled, vae_model, device='mps'):
    """
    Encodes scaled concentration data into VAE latent mean vectors (mu).
    
    Parameters:
        Y_scaled: pandas DataFrame or numpy array of shape (n_samples, n_taxa)
                  This should already be asinh-transformed and standardized.
        vae_model: trained VAE instance
        device: 'cpu' or 'cuda'

    Returns:
        Z_mu: np.ndarray of shape (n_samples, latent_dim)
    """
    # Ensure model is in eval mode
    vae_model.eval()
    vae_model.to(device)

    # Convert input to torch.Tensor
    Y_tensor = torch.tensor(Y_scaled.values, dtype=torch.float32).to(device)

    # Pass through encoder
    with torch.no_grad():
        mu, _ = vae_model.encode(Y_tensor)

    # Move result to CPU and convert to numpy
    Z_mu = mu.cpu().numpy()
    return Z_mu

from sklearn.model_selection import GridSearchCV, train_test_split

def split_regression_data(X_env, Y_conc, Z_target, test_size=0.2, val_size=0.1, random_state=42):
    """
    Splits X_env, Y_conc, and Z_target into train/val/test sets.

    Parameters:
        X_env: pandas DataFrame of env covariates (n_samples, n_features)
        Z_target: numpy array of VAE latent means (n_samples, latent_dim)
        Y_conc: pandas DataFrame of concentrations (original or scaled)
        test_size: fraction for final test set
        val_size: fraction of remaining training set for validation
        random_state: seed

    Returns:
        Dict with train/val/test sets for X_env, Z_target, and Y_conc
    """
    # First split: train+val vs. test
    X_temp, X_test, Z_temp, Z_test, Y_temp, Y_test = train_test_split(
        X_env, Z_target, Y_conc, test_size=test_size, #random_state=random_state
    )

    # Second split: train vs. val
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, Z_train, Z_val, Y_train, Y_val = train_test_split(
        X_temp, Z_temp, Y_temp, test_size=val_fraction, #random_state=random_state
    )

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'Z_train': Z_train,
        'Z_val': Z_val,
        'Z_test': Z_test,
        'Y_train': Y_train,
        'Y_val': Y_val,
        'Y_test': Y_test
    }


from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

def train_regressor(X_train, Z_train):
    """
    Trains a multi-output Random Forest Regressor to predict VAE latent means.

    Parameters:
        X_train: pandas DataFrame of shape (n_samples, n_features)
        Z_train: numpy array of shape (n_samples, latent_dim)

    Returns:
        Trained MultiOutputRegressor instance
    """
    param_grid = {
        'estimator__n_estimators': [200],
        'estimator__max_depth': [None, 20]
    }

    base_rf = RandomForestRegressor(n_jobs=-1)
    rf_model = MultiOutputRegressor(base_rf)

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

def train_all(path='oleander_data.csv'):
    X, Y, taxa, scaler = load_data(path)

    # 1. Split the data FIRST (no VAE yet!)
    Z_dummy = np.zeros((len(Y), 1))  # dummy placeholder
    splits = split_regression_data(X, Y, Z_dummy, test_size=0.2, val_size=0.1)

    # 2. Train VAE on Y_train only
    Y_train = splits['Y_train']
    X_tensor = torch.tensor(Y_train.values, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)

    vae = VAE(input_dim=Y.shape[1], latent_dim=8)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    train_vae(vae, dataloader, optimizer, epochs=2000, device='mps' if torch.backends.mps.is_available() else 'cpu')

    torch.save(vae.state_dict(), 'vae_model.pth')
    print("VAE model saved.")

    # 3. Encode all splits using trained VAE
    Z_train = encode_latent_means(splits['Y_train'], vae)
    Z_val   = encode_latent_means(splits['Y_val'], vae)
    Z_test  = encode_latent_means(splits['Y_test'], vae)

    # 4. Train RF using clean split
    rf_model = train_regressor(splits['X_train'], Z_train)

    torch.save(rf_model, 'rf_model.pth')
    print("Random Forest model saved.")

    # 5. Predict on validation set
    Z_val_pred = rf_model.predict(splits['X_val'])
    print("Predicted latent means shape:", Z_val_pred.shape)
    # Decode the predictions back to concentrations
    Y_val_pred = decode_and_inverse_transform(
        Z_val_pred, vae, scaler, taxa_columns=Y.columns.tolist(), device='mps' if torch.backends.mps.is_available() else 'cpu'
    )
    print("Predicted concentrations shape:", Y_val_pred.shape)
    Y_val_true_scaled = splits['Y_val']
    Y_val_true = np.sinh(scaler.inverse_transform(Y_val_true_scaled))
    print("True concentrations shape:", Y_val_true.shape)
    # make two side by side violin plots of the true and predicted concentrations
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.violinplot(Y_val_true, showmeans=True, showmedians=True)
    plt.title('True Concentrations (validation set)')
    plt.xlabel('Taxa')
    plt.ylabel('Concentration')
    plt.subplot(1, 2, 2)
    plt.violinplot(Y_val_pred.values, showmeans=True, showmedians=True)
    plt.title('Predicted Concentrations (validation set)')
    plt.xlabel('Taxa')
    plt.ylabel('Concentration')
    plt.tight_layout()
    plt.show()
    # now compute R^2 scores per taxon
    from sklearn.metrics import r2_score
    r2_scores = r2_score(Y_val_true, Y_val_pred, multioutput='raw_values')
    print("R² scores per taxon:", r2_scores)
    print("Mean R² score:", r2_score(Y_val_true, Y_val_pred, multioutput='uniform_average'))


def vis_all(path='oleander_data.csv'):
    """
    Main function to run the entire training and evaluation pipeline.
    """
    # Load data and preprocess
    X_env, Y_scaled, taxa_columns, scaler = load_data(path)

    # Visualize latent space
    vae = VAE(input_dim=Y_scaled.shape[1], latent_dim=8)
    vae.load_state_dict(torch.load('vae_model.pth'))
    visualize_latent_space(vae, Y_scaled, X_env, color_var='salinity', scaler=scaler, method='pca', device='mps' if torch.backends.mps.is_available() else 'cpu')

if __name__ == "__main__":
    # Run the training and evaluation pipeline
    train_all(path='oleander_data.csv')
    vis_all(path='oleander_data.csv')
