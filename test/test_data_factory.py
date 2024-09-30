import unittest
from unittest.mock import MagicMock
from trainingModel.timesnet_diff_method.data_factory import collate_fn_wrapper
import sys
import torch
sys.path.insert(0, "")


class TestDataProvider(unittest.TestCase):
    def setUp(self):
        self.args_mock = MagicMock()
        self.args_mock.data = 'ECG'
        self.args_mock.batch_size = 10
        self.args_mock.freq = 50
        self.args_mock.num_workers = 4
        self.args_mock.task_name = 'classification'
        self.args_mock.train_data = 'trainingModel/train_dataset.csv'
        self.args_mock.test_data = 'trainingModel/test_dataset.csv'
        self.args_mock.root_path = 'trainingModel/ECGDataProcessed/'
        self.args_mock.filename = 'FileName'
        self.args_mock.target = 'Rhythm'
        self.args_mock.seq_len = 500
        self.args_mock.num_features = 12


class TestCollateFnWrapper(unittest.TestCase):
    def setUp(self):

        self.data = [
            (torch.randn(5, 3), torch.tensor([1])),
            (torch.randn(10, 3), torch.tensor([2])),
        ]
        self.max_len = 8

    def test_wrapper_output_dimensions(self):
        wrapper = collate_fn_wrapper(self.max_len)
        X, targets, padding_masks = wrapper(self.data)

        self.assertEqual(X.size(), (2, self.max_len, 3))
        self.assertEqual(targets.size(), (2, 1))
        self.assertEqual(padding_masks.size(), (2, self.max_len))


        if padding_masks.dim() == 2:
            padding_masks = padding_masks.unsqueeze(-1)

        self.assertTrue(torch.all((X[:, 5:, :] == 0) | (padding_masks[:, 5:].expand_as(X[:, 5:, :]) == 1)), "Padding not applied correctly")

    def test_wrapper_correctly_passes_max_len(self):
        wrapper = collate_fn_wrapper(self.max_len)

        X, targets, padding_masks = wrapper(self.data)

        actual_length = (X[1].nonzero(as_tuple=True)[0].max().item() + 1)

        expected_length = min(10, self.max_len)

        self.assertEqual(actual_length, expected_length, "Max length clipping error")


if __name__ == '__main__':
    unittest.main()
