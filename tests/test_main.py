import unittest

from src.main import save_options, primary_checks


class TestMain(unittest.TestCase):
    def test_save_and_move_success(self):
        """
         Test that it saves and runs an experiment sucessfully
         """
        args = {'MinMaxScaler_copy': False,
                'MinMaxScaler_feature_range_high': 1,
                'MinMaxScaler_feature_range_low': 0,
                'classifier': 'NaiveBayes',
                'dataset_files': ['data_0inches.npy', 'labels_0inches.npy'],
                'run_options': False,
                'save_options': True,
                'scaler': 'MinMaxScaler',
                'split_method': 'train_test_split',
                'train_test_split_test_size': 0.2}

        result = save_options(args)
        self.assertIsNotNone(result)

    def test_primary_checks_success(self):
        """
         Test that it makes all primary checks
         """
        args = {'MinMaxScaler_copy': False,
                'MinMaxScaler_feature_range_high': 1,
                'MinMaxScaler_feature_range_low': 0,
                'classifier': 'NaiveBayes',
                'dataset_files': ['data_0inches.npy', 'labels_0inches.npy'],
                'plots_number': 4,
                'run_options': False,
                'save_options': False,
                'scaler': 'MinMaxScaler',
                'split_method': 'train_test_split',
                'train_test_split_test_size': 0.2
                }

        result = primary_checks(args)
        self.assertIsNotNone(result)

    def test_primary_checks_save_options_success(self):
        """
         Test that it makes all primary checks
         """
        args = {'MinMaxScaler_copy': False,
                'MinMaxScaler_feature_range_high': 1,
                'MinMaxScaler_feature_range_low': 0,
                'classifier': 'NaiveBayes',
                'dataset_files': ['data_0inches.npy', 'labels_0inches.npy'],
                'plots_number': 4,
                'run_options': False,
                'save_options': True,
                'scaler': 'MinMaxScaler',
                'split_method': 'train_test_split',
                'train_test_split_test_size': 0.2
                }

        result = primary_checks(args)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
