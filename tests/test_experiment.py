import unittest
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from src.experiment import run_experiment


class TestExperiment(unittest.TestCase):
    def test_parameters_kfold_success(self):
        """
         Test that it run an experiment sucessfully
         """
        scaler = MinMaxScaler(copy=False, feature_range=(-1, 1))
        classifier = GaussianNB()
        split = {'split_method': 'kfolds', 'n_splits': 3}
        data = 'data_0inches.npy'
        labels = 'labels_0inches.npy'
        plots = {'plots_number': 3}

        result = run_experiment(scaler, classifier, split, data, labels, plots)
        self.assertIsNotNone(result)

    def test_parameters_test_train_split_0inches_success(self):
        """
         Test that it run an experiment sucessfully
         """
        scaler = MinMaxScaler(copy=False, feature_range=(-1, 1))
        classifier = GaussianNB()
        split = {'split_method': 'train_test_split', 'test_size': 0.2}

        data = 'data_0inches.npy'
        labels = 'labels_0inches.npy'
        plots = {'plots_number': 3}

        result = run_experiment(scaler, classifier, split, data, labels, plots)
        self.assertIsNotNone(result)
    def test_parameters_test_train_split_6inches_success(self):
        """
         Test that it run an experiment sucessfully
         """
        scaler = MinMaxScaler(copy=False, feature_range=(-1, 1))
        classifier = GaussianNB()
        split = {'split_method': 'train_test_split', 'test_size': 0.2}

        data = 'data_6inches.npy'
        labels = 'labels_6inches.npy'
        plots = {'plots_number': 3}

        result = run_experiment(scaler, classifier, split, data, labels, plots)
        self.assertIsNotNone(result)

    def test_parameters_kfold_fail(self):
        """
         Test that it run an experiment unsucessfully
         """
        scaler = MinMaxScaler(copy=False, feature_range=(-1, 1))
        classifier = GaussianNB()
        split = {'split_method': 'kfolds', 'n_splits': 'fail'}

        data = 'data_0inches.npy'
        labels = 'labels_0inches.npy'
        plots = {'plots_number': 3}

        result = run_experiment(scaler, classifier, split, data, labels, plots)
        self.assertEqual(result, 1)

    def test_parameters_test_train_split_fail(self):
        """
         Test that it run an experiment unsucessfully
         """
        scaler = MinMaxScaler(copy=False, feature_range=(-1, 1))
        classifier = GaussianNB()
        split = {'split_method': 'train_test_split', 'test_size': 'fail'}

        data = 'data_0inches.npy'
        labels = 'labels_0inches.npy'

        plots = {'plots_number': 3}

        result = run_experiment(scaler, classifier, split, data, labels, plots)
        self.assertEqual(result, 1)

    def test_parameters_no_split_method_fail(self):
        """
         Test that it run an experiment unsucessfully
         """
        scaler = MinMaxScaler(copy=False, feature_range=(-1, 1))
        classifier = GaussianNB()
        split = {}

        data = 'data_0inches.npy'
        labels = 'labels_0inches.npy'

        plots = {'plots_number': 3}

        result = run_experiment(scaler, classifier, split, data, labels, plots)
        self.assertEqual(result, 1)

    def test_parameters_wrong_split_method_fail(self):
        """
         Test that it run an experiment unsucessfully
         """
        scaler = MinMaxScaler(copy=False, feature_range=(-1, 1))
        classifier = GaussianNB()
        split = {'split_method': 'my_split', 'test_size': 'fail'}

        data = 'data_0inches.npy'
        labels = 'labels_0inches.npy'

        plots = {'plots_number': 3}

        result = run_experiment(scaler, classifier, split, data, labels, plots)
        self.assertEqual(result, 1)


if __name__ == '__main__':
    unittest.main()
