import pandas as pd
from sklearn.model_selection import train_test_split


def merge_target_labels(input_df, output_df, labels_to_merge, target):
    input_df[target] = input_df[target].replace(labels_to_merge)

    input_df.to_csv(output_df, index=False)

    print(f"Modified CSV has been saved to {output_df}.")


def split_dataset(dataset, train_df_csv,test_df_csv):
    df = pd.read_csv(dataset)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv(train_df_csv, index=False)

    test_df.to_csv(test_df_csv, index=False)

