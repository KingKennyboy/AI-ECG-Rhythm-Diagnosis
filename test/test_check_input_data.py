import unittest
from trainingModel.timesnet_diff_method.check_input_data import check_csv_columns, is_csv


class TestCSVFunctions(unittest.TestCase):
    def test_check_csv_columns(self):
        data1 = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12],
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
        self.assertTrue(check_csv_columns(data1))

        data2 = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        self.assertFalse(check_csv_columns(data2))

    def test_is_csv(self):
        data1 = iter([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
        self.assertTrue(is_csv(data1))

        data2 = iter([])
        self.assertFalse(is_csv(data2))

        data3 = "not iterable"
        self.assertFalse(is_csv(data3))


if __name__ == '__main__':
    unittest.main()

