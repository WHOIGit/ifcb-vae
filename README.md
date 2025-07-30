# Variational Autoencoder for IFCB Counts

This repository contains a Variational Autoencoder (VAE) implementation designed to analyze and reconstruct phytoplankton concentration data collected by Imaging FlowCytobot (IFCB). The pipeline includes data preprocessing, VAE training, latent space visualization, and regressor training to predict latent data from environmental covariates, which can then be decoded to reconstruct predicted concentrations.

## Features

- **Data Preprocessing**:
  - Handles missing or zero values in the dataset.
  - Applies an asinh transformation for numerical stability.
  - Standardizes taxa concentrations and filters out taxa with insufficient data.

- **Variational Autoencoder (VAE)**:
  - Encodes taxa concentration data into a latent space.
  - Reconstructs the original data from the latent space.
  - Supports customizable latent dimensions and hidden layer configurations.

- **Visualization**:
  - Visualizes the latent space using PCA or t-SNE.
  - Generates violin plots for taxa concentrations and latent space distributions.
  - Creates stacked bar plots to analyze reconstructed data along transects in the latent space.

- **Regressor Training**:
  - Trains a Random Forest Regressor to predict latent representations from environmental variables.
  - Evaluates regressor performance using R² scores.
  - Validates reconstructed predictions against the original concentration data.

## Usage

### 1. Data Preprocessing
The `load_data` function prepares the input data by:
- Splitting environmental and taxa columns.
- Computing concentrations and applying transformations.
- Standardizing and filtering taxa data.

### 2. Training the VAE
The `train_vae` function trains the VAE on the preprocessed taxa data. Customize the latent dimensions and hidden layers in the `VAE` class.

### 3. Visualizing the Latent Space
The `visualize_latent_space` function provides insights into the latent space by:
- Reducing dimensions using PCA or t-SNE.
- Plotting distributions and transects through the latent space.

### 4. Training the Regressor
The `train_regressor` function trains a Random Forest Regressor to predict latent representations from environmental variables. Evaluate the model using R² scores.

### 5. Decoding and Reconstruction
The `decode_and_inverse_transform` function decodes latent vectors and applies inverse transformations to reconstruct taxa concentrations.

## Example Workflow

1. Preprocess the data:
   ```python
   X_env, Y_scaled, taxa_columns, scaler = load_data('oleander_data.csv')
   ```

2. Train the VAE:
   ```python
   vae = VAE(input_dim=Y_scaled.shape[1], latent_dim=8)
   train_vae(vae, dataloader, optimizer, epochs=2000, device='cpu')
   ```

3. Visualize the latent space:
   ```python
   visualize_latent_space(vae, Y_scaled, X_env, color_var='temperature', method='pca')
   ```

4. Train the regressor:
   ```python
   rf_model = train_regressor(X_train, Z_train)
   ```

5. Decode and reconstruct data:
   ```python
   Y_pred = decode_and_inverse_transform(Z_pred, vae, scaler, taxa_columns)
   ```

## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn

Install dependencies using:
```bash
pip install -r requirements.txt
```
