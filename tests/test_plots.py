import unittest
import numpy

from src.ploting import plot_observations


class TestPlot(unittest.TestCase):
    def test_plot_observations(self):
        """
         Test that it export a plot
         """
        X = numpy.load('dataset/data_0inches.npy')
        number_of_plots = 3
        result = plot_observations(X, number_of_plots)
        self.assertTrue(result)

# TODO check more cases

if __name__ == '__main__':
    unittest.main()
