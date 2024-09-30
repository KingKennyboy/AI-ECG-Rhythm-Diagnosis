import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class ECGDataLoader(Dataset):
    def __init__(self, train_data, test_data, ecg_folder, ecg_col_name, labels_col_name, seq_len, num_features, flag):
        self.train_data = train_data
        self.test_data = test_data
        self.ecg_path = ecg_folder
        self.ecg_col_name = ecg_col_name
        self.labels = labels_col_name
        self.seq_len = seq_len
        self.num_features = num_features
        self.flag = flag
        self.data = self._load_and_normalize_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X, y = self.data[idx]
        return X, y

    def mean_and_std(self, train):
        mean = np.mean(train, axis=(0, 1), keepdims=True)

        std = np.std(train, axis=(0, 1), keepdims=True)

        return mean, std

    def normalized_data(self, data, mean, std):
        return (data - mean) / std

    def load_data(self, data):
        ecg_data = 0

        df = pd.read_csv(data)

        for _, row in df.iterrows():
            ecg_file_path = self.ecg_path + row[self.ecg_col_name] + '.csv'

            ecg_data = pd.read_csv(ecg_file_path, header=None).values

        return df, ecg_data

    def _load_and_normalize_data(self):
        data = []
        df_train, train_data = self.load_data(self.train_data)

        mean, std = self.mean_and_std(train_data)

        if self.flag == 'TRAIN':
            df = df_train

        else:
            df = pd.read_csv(self.test_data)

        label_to_position = {label: i for i, label in enumerate(df[self.labels].unique())}

        for _, row in df.iterrows():
            ecg_file_path = self.ecg_path + row[self.ecg_col_name] + '.csv'

            label_index = label_to_position[row[self.labels]]

            ecg_data = pd.read_csv(ecg_file_path, header=None).values

            if np.isnan(ecg_data).any():
                print('null')
                continue
            normalized_data = self.normalized_data(ecg_data, mean, std)

            X = torch.tensor(normalized_data, dtype=torch.float32)

            y = torch.tensor(label_index, dtype=torch.long)

            data.append((X, y))

        return data

