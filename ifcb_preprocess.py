import pandas as pd
import numpy as np
import os
import re

def import_google_sheet(share_url, save_path=None):
    """
    Download a specific tab from a Google Sheets share URL as CSV.

    Args:
        share_url (str): Full Google Sheets share URL.
        save_path (str, optional): Path to save the CSV locally.

    Returns:
        pd.DataFrame: Loaded CSV as a DataFrame.
    """
    # Extract spreadsheet ID
    id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', share_url)
    if not id_match:
        raise ValueError("Spreadsheet ID not found in URL.")
    spreadsheet_id = id_match.group(1)

    # Extract GID (tab ID)
    gid_match = re.search(r'[?&]gid=(\d+)', share_url)
    gid = gid_match.group(1) if gid_match else '0'  # default to first tab

    # Build export URL
    export_url = f'https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}'

    try:
        df = pd.read_csv(export_url)
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Saved taxonomy to: {save_path}")
        else:
            print("Loaded taxonomy from Google Sheets.")
        return df
    except Exception as e:
        print(f"Error downloading or parsing CSV: {e}")
        return None



def load_data(workdir, data_type):
    """
    Load metadata, raw class data (count or carbon), and taxonomy.
    """
    meta_path = os.path.join(workdir, 'ifcb_metadata.csv')
    raw_path = os.path.join(workdir, f'ifcb_{data_type}_raw.csv')
    tax_path = os.path.join(workdir, 'ifcb_taxonomy.csv')

    print(f"\nLoading data for '{data_type}'...")
    meta = pd.read_csv(meta_path)
    raw = pd.read_csv(raw_path)
    tax = pd.read_csv(tax_path)

    return meta, raw, tax

def preprocess(meta, raw):
    """
    Merge and filter metadata and class data.
    Add date and space-time grid columns.
    """
    data_cols = raw.columns[1:].tolist()
    df = pd.merge(meta, raw[['pid'] + data_cols], on='pid', how='left')

    # Filter unwanted rows
    df = df[df['skip'] == 0]
    df = df[df['sample_type'].isin(['Normal', np.nan])]
    df['ml_analyzed'] = df['ml_analyzed'].replace(0, np.nan)
    df = df.dropna(subset=['sample_time', 'longitude', 'latitude', 'ml_analyzed']).reset_index(drop=True)

    # Date/time components
    df['sample_time'] = pd.to_datetime(df['sample_time'])
    df['year'] = df['sample_time'].dt.year
    df['month'] = df['sample_time'].dt.month
    df['day'] = df['sample_time'].dt.day
    df['week'] = df['sample_time'].dt.isocalendar().week
    df['doy'] = df['sample_time'].dt.dayofyear
    df['season'] = df['month'].map(
        lambda m: 'JFM' if m in [1, 2, 3] else
                  'AMJ' if m in [4, 5, 6] else
                  'JAS' if m in [7, 8, 9] else
                  'OND'
    )

    # Spatial-temporal grid columns
    decimal_places = 2
    df['x'] = (df['longitude'].round(decimal_places) * 10**decimal_places).astype(int)
    df['y'] = (df['latitude'].round(decimal_places) * 10**decimal_places).astype(int)
    df['t'] = ((df['sample_time'] - df['sample_time'].min()) / pd.Timedelta(minutes=1)).round().astype(int)

    return df, data_cols

def aggregate_cast_data(df, data_cols):
    """
    Aggregate replicate cast samples if present.
    """
    if 'cast' not in df['sample_type'].unique():
        print("No cast data found.")
        return df

    print("Aggregating cast data...")
    cast_data = df[df['sample_type'] == 'cast']
    other_data = df[df['sample_type'] != 'cast']

    agg_dict = {
        col: 'sum' if col in data_cols or col == 'ml_analyzed' else 'first'
        for col in cast_data.columns if col not in ['cruise', 'cast', 'depth']
    }

    cast_agg = cast_data.groupby(['cruise', 'cast', 'depth'], as_index=False).agg(agg_dict)
    df = pd.concat([cast_agg, other_data], axis=0, ignore_index=True)
    return df.sort_values('sample_time').reset_index(drop=True)

def normalize_and_collapse(df, taxonomy, data_cols, data_type):
    """
    Normalize by volume analyzed and collapse annotations to taxonomy labels.
    """
    raw_annotations = taxonomy['Annotations'].tolist()
    valid_annotations = [col for col in raw_annotations if col in df.columns]
    column_map = dict(zip(taxonomy['Annotations'], taxonomy['Label']))

    df_norm = (
        df[valid_annotations]
        .div(df['ml_analyzed'], axis=0)
        .rename(columns=column_map)
        .T.groupby(level=0).sum().T
    )

    df_norm[['x', 'y', 't']] = df[['x', 'y', 't']].values
    grouped = df_norm.groupby(['x', 'y', 't']).mean()

    if data_type == 'count':
        grouped = grouped.round(0).astype(int)

    return grouped.reset_index(drop=True)

def process_dataset(workdir, data_type):
    """
    Run full pipeline for one dataset: count or carbon.
    """
    meta, raw, tax = load_data(workdir, data_type)
    df, data_cols = preprocess(meta, raw)
    df = aggregate_cast_data(df, data_cols)
    df_norm = normalize_and_collapse(df, tax, data_cols, data_type)

    # Clean and export
    df = df.drop(columns=tax['Annotations'].tolist(), errors='ignore')
    df = pd.concat([df, df_norm], axis=1)

    output_path = os.path.join(workdir, f'ifcb_{data_type}_clean.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned {data_type} data to: {output_path}")

def process_all(workdir):
    """
    Process both count and carbon datasets.
    """
    print(f"Processing all IFCB data in: {workdir}")

    # Check that necessary files exist
    expected_files = [
        'ifcb_metadata.csv',
        'ifcb_taxonomy.csv',
        'ifcb_count_raw.csv',
        'ifcb_carbon_raw.csv'
    ]
    missing = [f for f in expected_files if not os.path.exists(os.path.join(workdir, f))]
    if missing:
        print(f"Error: missing file(s) in {workdir}: {missing}")
        return

    # Process both datasets
    for dtype in ['count', 'carbon']:
        process_dataset(workdir, dtype)


if __name__ == "__main__":
    workdir = 'data/oleander'
    taxonomy_file = os.path.join(workdir, 'ifcb_taxonomy.csv')

    # Download taxonomy file if missing
    if not os.path.exists(taxonomy_file):
        print("ifcb_taxonomy.csv not found, downloading from Google Sheets...")
        url = "https://docs.google.com/spreadsheets/d/1dkjfPrBFH9t-8Ymh9xHTdAFKwuxMGpGmjn_eeslIZBw/edit?pli=1&gid=1521292620#gid=1521292620"
        taxonomy = import_google_sheet(url, save_path=taxonomy_file)

    # Run full processing
    process_all(workdir)
