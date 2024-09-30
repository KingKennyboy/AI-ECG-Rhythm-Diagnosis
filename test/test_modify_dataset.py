import pandas as pd
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainingModel.modify_dataset import merge_target_labels


class TestMergeLabels(unittest.TestCase):
    def setUp(self):
        self.input_data = {
            'ID': [1, 2, 3, 4, 5],
            'Category': ['A', 'B', 'C', 'D', 'E']
        }
        self.input_df = pd.DataFrame(self.input_data)

        self.labels_to_merge = {'A': 'X', 'B': 'Y', 'C': 'Z'}

    def test_merge_labels(self):
        expected_output_data = {
            'ID': [1, 2, 3, 4, 5],
            'Category': ['X', 'Y', 'Z', 'D', 'E']
        }

        expected_output_df = pd.DataFrame(expected_output_data)

        output_file = "test_output.csv"

        merge_target_labels(self.input_df, output_file, self.labels_to_merge, 'Category')

        result_df = pd.read_csv(output_file)

        self.assertTrue(result_df.equals(expected_output_df))

    def test_merge_labels_empty_input(self):
        empty_df = pd.DataFrame(columns=['ID', 'Category'])

        output_file = "test_output_empty.csv"

        merge_target_labels(empty_df, output_file, self.labels_to_merge, 'Category')

        result_df = pd.read_csv(output_file)

        self.assertTrue(result_df.empty)

    def test_merge_labels_no_matching_labels(self):
        labels_to_merge = {'X': 'A', 'Y': 'B', 'Z': 'C'}

        output_file = "test_output_no_matching.csv"

        merge_target_labels(self.input_df, output_file, labels_to_merge, 'Category')

        result_df = pd.read_csv(output_file)

        self.assertTrue(result_df.equals(self.input_df))

    def tearDown(self):
        import os
        for file_name in ["test_output.csv", "test_output_empty.csv", "test_output_no_matching.csv"]:
            if os.path.exists(file_name):
                os.remove(file_name)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
