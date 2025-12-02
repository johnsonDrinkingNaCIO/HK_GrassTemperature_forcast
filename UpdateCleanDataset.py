import pandas as pd
import numpy as np
import os
import glob
def load_all_csvs_combined(folder_path='DataSets'):
  csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
  if not csv_files:
    print(f"No CSV files found in {folder_path}")
    return pd.DataFrame()

  print(f"Found {len(csv_files)} CSV files:")
  for file in csv_files:
    print(os.path.basename(file))


  for f in csv_files:
    missing_count = 0
    df = pd.read_csv(f, skiprows=3, header=None)
    df = df.iloc[:-3]
    df = df.apply(pd.to_numeric, errors='coerce')

    # CHANGED: Just count missing after
    missing_count = df.iloc[:, 3].isna().sum()

    # CHANGED: Use pandas interpolate instead of manual loop
    df.iloc[:, 3] = df.iloc[:, 3].interpolate(method='linear').ffill().bfill()

    

    print(f"Missing values in column 4: {missing_count}")
    print(f"shape:{df.shape}")
    print(df.head())

    return df


all_data = load_all_csvs_combined()