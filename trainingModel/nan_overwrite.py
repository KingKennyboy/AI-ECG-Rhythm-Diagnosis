import pandas as pd

def process_csv(csv_file_path, expected_columns=None):
    first_row = pd.read_csv(csv_file_path, nrows=1)

    has_header = False
    if expected_columns:
        has_header = all(column in first_row.columns for column in expected_columns)
    else:
        has_header = not first_row.applymap(lambda x: pd.to_numeric(x, errors='coerce')).dropna().empty

    if has_header:
        df = pd.read_csv(csv_file_path)
    else:
        df = pd.read_csv(csv_file_path, header=None)

    if not has_header and expected_columns:
        df.columns = expected_columns
    
    df.fillna(df.mean(), inplace=True)
    
    df.to_csv(csv_file_path, index=False, header=has_header or expected_columns)

